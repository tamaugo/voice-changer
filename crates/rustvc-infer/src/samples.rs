//! Clips committed as `voices/<Name>/samples/*`.

use std::fs;
use std::path::{Path, PathBuf};

pub const AUDIO_EXTENSIONS: &[&str] = &["wav", "flac", "mp3", "ogg", "m4a", "aac"];
pub const MAX_RECOMMENDED_BYTES: u64 = 50 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SampleClip {
    pub path: PathBuf,
    pub file_name: String,
    pub bytes: u64,
    pub over_limit: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepoVoice {
    pub name: String,
    pub directory: PathBuf,
    pub samples: Vec<SampleClip>,
    pub trained: bool,
    /// `source.toml` is present, so clips can be downloaded onto this PC.
    pub has_source: bool,
}

pub fn is_audio_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            AUDIO_EXTENSIONS
                .iter()
                .any(|allowed| ext.eq_ignore_ascii_case(allowed))
        })
        .unwrap_or(false)
}

/// Find `configured` by walking `roots` and their parents.
/// An absolute `configured` path is returned as given.
pub fn resolve_voices_dir(configured: &Path, roots: &[PathBuf]) -> PathBuf {
    if configured.is_absolute() {
        return configured.to_path_buf();
    }
    for root in roots {
        let mut dir = Some(root.as_path());
        for _ in 0..8 {
            let Some(current) = dir else { break };
            let candidate = current.join(configured);
            if candidate.is_dir() {
                return candidate;
            }
            dir = current.parent();
        }
    }
    roots
        .first()
        .map(|root| root.join(configured))
        .unwrap_or_else(|| configured.to_path_buf())
}

pub fn scan_repo_voices(root: &Path) -> Vec<RepoVoice> {
    let Ok(entries) = fs::read_dir(root) else {
        return Vec::new();
    };
    let mut voices = Vec::new();
    for entry in entries.flatten() {
        let directory = entry.path();
        if !directory.is_dir() {
            continue;
        }
        let Some(name) = entry.file_name().into_string().ok() else {
            continue;
        };
        if name.starts_with('.') {
            continue;
        }
        let samples = list_samples(&directory.join("samples"));
        let has_source = directory.join("source.toml").is_file();
        if samples.is_empty() && !has_source {
            continue;
        }
        voices.push(RepoVoice {
            trained: has_trained_model(&directory),
            has_source,
            name,
            directory,
            samples,
        });
    }
    voices.sort_by(|left, right| {
        right.trained.cmp(&left.trained).reverse().then_with(|| {
            left.name
                .to_ascii_lowercase()
                .cmp(&right.name.to_ascii_lowercase())
        })
    });
    voices
}

fn list_samples(dir: &Path) -> Vec<SampleClip> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut clips = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() || !is_audio_file(&path) {
            continue;
        }
        let bytes = entry.metadata().map(|meta| meta.len()).unwrap_or(0);
        let file_name = entry.file_name().to_string_lossy().into_owned();
        clips.push(SampleClip {
            over_limit: bytes > MAX_RECOMMENDED_BYTES,
            path,
            file_name,
            bytes,
        });
    }
    clips.sort_by(|left, right| {
        left.file_name
            .to_ascii_lowercase()
            .cmp(&right.file_name.to_ascii_lowercase())
    });
    clips
}

fn has_trained_model(dir: &Path) -> bool {
    if dir.join("synthesizer.onnx").is_file() || dir.join("model.pth").is_file() {
        return true;
    }
    let Ok(text) = fs::read_to_string(dir.join("voice.toml")) else {
        return false;
    };
    if !text.lines().any(|line| {
        let line = line.trim();
        line == "trained = true" || line.starts_with("trained = true")
    }) {
        return false;
    }
    model_path_from_toml(&text)
        .map(|relative| dir.join(relative).is_file())
        .unwrap_or(false)
}

fn model_path_from_toml(text: &str) -> Option<&str> {
    for key in ["model", "content_model"] {
        for line in text.lines() {
            let line = line.trim();
            let Some(rest) = line.strip_prefix(key) else {
                continue;
            };
            let rest = rest.trim().strip_prefix('=')?.trim().trim_matches('"');
            if !rest.is_empty() {
                return Some(rest);
            }
        }
    }
    None
}

/// Copy `files` into an empty `dest` directory. The trainer reads that folder.
pub fn stage_selected(files: &[PathBuf], dest: &Path) -> std::io::Result<usize> {
    if dest.exists() {
        fs::remove_dir_all(dest)?;
    }
    fs::create_dir_all(dest)?;
    let mut copied = 0;
    for file in files {
        let name = file.file_name().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "sample has no file name")
        })?;
        fs::copy(file, dest.join(name))?;
        copied += 1;
    }
    Ok(copied)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav(dir: &Path, name: &str) {
        fs::create_dir_all(dir).unwrap();
        fs::write(dir.join(name), b"RIFF").unwrap();
    }

    #[test]
    fn untrained_voice_lists_audio_only() {
        let root = std::env::temp_dir().join(format!("rustvc-scan-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let samples = root.join("Ada").join("samples");
        wav(&samples, "b.wav");
        wav(&samples, "a.WAV");
        fs::write(samples.join("notes.txt"), b"no").unwrap();
        let voices = scan_repo_voices(&root);
        assert_eq!(voices.len(), 1);
        assert_eq!(voices[0].name, "Ada");
        assert!(!voices[0].trained);
        assert_eq!(
            voices[0]
                .samples
                .iter()
                .map(|clip| clip.file_name.as_str())
                .collect::<Vec<_>>(),
            vec!["a.WAV", "b.wav"]
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn model_file_marks_the_voice_trained() {
        let root = std::env::temp_dir().join(format!("rustvc-scan-trained-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let ada = root.join("Ada");
        wav(&ada.join("samples"), "clip.wav");
        fs::write(ada.join("synthesizer.onnx"), b"onnx").unwrap();
        let fresh = root.join("Bea");
        wav(&fresh.join("samples"), "clip.wav");
        fs::write(
            fresh.join("voice.toml"),
            "name = \"Bea\"\ntrained = true\nmodel = \"missing.pth\"\n",
        )
        .unwrap();
        let voices = scan_repo_voices(&root);
        assert_eq!(voices.len(), 2);
        let ada = voices.iter().find(|voice| voice.name == "Ada").unwrap();
        let bea = voices.iter().find(|voice| voice.name == "Bea").unwrap();
        assert!(ada.trained);
        assert!(!bea.trained);
        assert_eq!(voices[0].name, "Bea");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn source_manifest_lists_a_voice_with_no_samples() {
        let root = std::env::temp_dir().join(format!("rustvc-scan-source-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let rem = root.join("Rem");
        fs::create_dir_all(&rem).unwrap();
        fs::write(rem.join("source.toml"), "name = \"Rem\"\n").unwrap();
        let voices = scan_repo_voices(&root);
        assert_eq!(voices.len(), 1);
        assert!(voices[0].has_source);
        assert!(voices[0].samples.is_empty());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn resolve_walks_up_from_the_exe_dir() {
        let root = std::env::temp_dir().join(format!("rustvc-resolve-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("voices")).unwrap();
        let exe = root.join("target").join("release");
        fs::create_dir_all(&exe).unwrap();
        let found = resolve_voices_dir(Path::new("voices"), &[exe]);
        assert_eq!(found, root.join("voices"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn stage_selected_copies_only_those_files() {
        let root = std::env::temp_dir().join(format!("rustvc-stage-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let samples = root.join("samples");
        wav(&samples, "keep.wav");
        wav(&samples, "skip.wav");
        let dest = root.join("dataset");
        let n = stage_selected(&[samples.join("keep.wav")], &dest).unwrap();
        assert_eq!(n, 1);
        assert!(dest.join("keep.wav").is_file());
        assert!(!dest.join("skip.wav").exists());
        let _ = fs::remove_dir_all(&root);
    }
}
