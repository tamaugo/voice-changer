//! Training jobs.
//!
//! A pure-Rust RVC trainer is not realistic for this scaffold. The generator,
//! discriminator, ContentVec features, RMVPE pitch, and faiss index are
//! already implemented in the MIT-licensed RVC WebUI, and that tree is what
//! knows how to emit a checkpoint the live model can load. Reimplementing it
//! in candle or tch would fork the checkpoint format.
//!
//! This crate spawns `python/rvc_train.py`, which runs that CLI in a
//! subprocess. The script is original. It does not import RVC. Dry-run mode
//! exercises the same progress protocol without a GPU or an RVC checkout.

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum TrainError {
    #[error("voice name is empty or has path characters")]
    BadName,
    #[error("dataset directory does not exist: {0}")]
    MissingDataset(PathBuf),
    #[error("training script not found. Looked for python/rvc_train.py")]
    MissingScript,
    #[error("failed to start training: {0}")]
    Spawn(String),
}

#[derive(Debug, Clone)]
pub struct TrainRequest {
    pub python: String,
    pub script: PathBuf,
    pub rvc_root: Option<PathBuf>,
    pub name: String,
    pub dataset: PathBuf,
    pub epochs: u32,
    pub batch_size: u32,
    pub sample_rate: u32,
    pub gpu: u32,
    pub version: String,
    pub f0: bool,
    pub voices_dir: PathBuf,
    pub dry_run: bool,
    pub workers: u32,
    /// Override the dry-run pause between stages. `None` keeps the script default.
    pub sleep_secs: Option<f32>,
}

impl TrainRequest {
    pub fn validate(&self) -> Result<(), TrainError> {
        if !valid_name(&self.name) {
            return Err(TrainError::BadName);
        }
        if !self.dataset.is_dir() {
            return Err(TrainError::MissingDataset(self.dataset.clone()));
        }
        if !self.script.is_file() {
            return Err(TrainError::MissingScript);
        }
        Ok(())
    }
}

pub fn valid_name(name: &str) -> bool {
    let name = name.trim();
    !name.is_empty()
        && name.len() <= 64
        && name
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' || ch == ' ')
}

#[derive(Debug, Clone)]
pub enum TrainEvent {
    Progress { fraction: f32, message: String },
    Log(String),
    Voice(PathBuf),
    Error(String),
    Finished { code: i32 },
}

pub fn parse_line(line: &str) -> Option<TrainEvent> {
    let line = line.trim();
    if line.is_empty() {
        return None;
    }
    if let Some(rest) = line.strip_prefix("PROGRESS ") {
        let mut parts = rest.splitn(2, ' ');
        let fraction = parts
            .next()
            .and_then(|text| text.parse::<f32>().ok())
            .unwrap_or(0.0);
        let message = parts.next().unwrap_or("").to_string();
        return Some(TrainEvent::Progress {
            fraction: fraction.clamp(0.0, 1.0),
            message,
        });
    }
    if let Some(rest) = line.strip_prefix("LOG ") {
        return Some(TrainEvent::Log(rest.to_string()));
    }
    if let Some(rest) = line.strip_prefix("VOICE ") {
        return Some(TrainEvent::Voice(PathBuf::from(rest)));
    }
    if let Some(rest) = line.strip_prefix("ERROR ") {
        return Some(TrainEvent::Error(rest.to_string()));
    }
    Some(TrainEvent::Log(line.to_string()))
}

pub struct TrainHandle {
    child: Option<Child>,
    events: Arc<Mutex<Vec<TrainEvent>>>,
    readers: Vec<JoinHandle<()>>,
}

impl TrainHandle {
    pub fn spawn(request: TrainRequest) -> Result<Self, TrainError> {
        request.validate()?;
        let mut command = Command::new(&request.python);
        command
            .arg(&request.script)
            .arg("--name")
            .arg(request.name.trim())
            .arg("--dataset")
            .arg(&request.dataset)
            .arg("--epochs")
            .arg(request.epochs.to_string())
            .arg("--batch-size")
            .arg(request.batch_size.to_string())
            .arg("--sample-rate")
            .arg(request.sample_rate.to_string())
            .arg("--gpu")
            .arg(request.gpu.to_string())
            .arg("--version")
            .arg(&request.version)
            .arg("--voices-dir")
            .arg(&request.voices_dir)
            .arg("--workers")
            .arg(request.workers.to_string())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env("PYTHONUNBUFFERED", "1");
        if request.f0 {
            command.arg("--f0");
        } else {
            command.arg("--no-f0");
        }
        if request.dry_run {
            command.arg("--dry-run");
        }
        if let Some(sleep) = request.sleep_secs {
            command.arg("--sleep").arg(sleep.to_string());
        }
        if let Some(root) = &request.rvc_root {
            command.arg("--rvc-root").arg(root);
        }
        let mut child = command
            .spawn()
            .map_err(|err| TrainError::Spawn(err.to_string()))?;
        let events = Arc::new(Mutex::new(Vec::new()));
        let mut readers = Vec::new();
        if let Some(stdout) = child.stdout.take() {
            readers.push(spawn_reader(stdout, Arc::clone(&events)));
        }
        if let Some(stderr) = child.stderr.take() {
            readers.push(spawn_reader(stderr, Arc::clone(&events)));
        }
        Ok(Self {
            child: Some(child),
            events,
            readers,
        })
    }

    /// Events produced since the last call.
    pub fn poll(&mut self) -> Vec<TrainEvent> {
        let mut events = self.events.lock().expect("train events").split_off(0);
        if let Some(child) = self.child.as_mut() {
            if let Ok(Some(status)) = child.try_wait() {
                events.push(TrainEvent::Finished {
                    code: status.code().unwrap_or(-1),
                });
                self.child = None;
            }
        }
        events
    }

    pub fn running(&mut self) -> bool {
        let Some(child) = self.child.as_mut() else {
            return false;
        };
        matches!(child.try_wait(), Ok(None) | Err(_))
    }
}

impl Drop for TrainHandle {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        for reader in self.readers.drain(..) {
            let _ = reader.join();
        }
    }
}

fn spawn_reader<R>(pipe: R, events: Arc<Mutex<Vec<TrainEvent>>>) -> JoinHandle<()>
where
    R: std::io::Read + Send + 'static,
{
    std::thread::spawn(move || {
        let reader = BufReader::new(pipe);
        for line in reader.lines() {
            let Ok(line) = line else { break };
            if let Some(event) = parse_line(&line) {
                events.lock().expect("train events").push(event);
            }
        }
    })
}

/// Candidates for the trainer script, preferring the source tree.
pub fn script_candidates() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    paths.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../python/rvc_train.py"));
    if let Ok(cwd) = std::env::current_dir() {
        paths.push(cwd.join("python/rvc_train.py"));
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            paths.push(dir.join("rvc_train.py"));
            paths.push(dir.join("python/rvc_train.py"));
        }
    }
    paths
}

pub fn find_script() -> Option<PathBuf> {
    script_candidates().into_iter().find(|path| path.is_file())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_progress_protocol() {
        match parse_line("PROGRESS 0.25 extract f0").unwrap() {
            TrainEvent::Progress { fraction, message } => {
                assert!((fraction - 0.25).abs() < f32::EPSILON);
                assert_eq!(message, "extract f0");
            }
            other => panic!("unexpected {other:?}"),
        }
        match parse_line("VOICE voices/ada/voice.toml").unwrap() {
            TrainEvent::Voice(path) => assert_eq!(path, PathBuf::from("voices/ada/voice.toml")),
            other => panic!("unexpected {other:?}"),
        }
        assert!(matches!(
            parse_line("ERROR missing torch"),
            Some(TrainEvent::Error(_))
        ));
        assert!(matches!(parse_line("plain log"), Some(TrainEvent::Log(_))));
        assert!(parse_line("   ").is_none());
    }

    #[test]
    fn rejects_path_names() {
        assert!(valid_name("Ada"));
        assert!(valid_name("ada_2"));
        assert!(!valid_name(""));
        assert!(!valid_name("../ada"));
        assert!(!valid_name("a/b"));
    }

    #[test]
    fn script_exists_in_the_source_tree() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../python/rvc_train.py");
        assert!(path.is_file(), "missing {}", path.display());
    }
}
