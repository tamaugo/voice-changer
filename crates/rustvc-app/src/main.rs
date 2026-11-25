//! voice-changer desktop app.
//!
//! The window has four actions: pick a voice, train a voice, settings, and
//! close. Audio starts on launch when a capture and playback device exist.

mod ui;

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};

use rustvc_train::{find_script, TrainEvent, TrainHandle, TrainRequest};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("voice-changer\n  --self-test     run the offline audio, pitch, and training checks\n  --list-devices  print capture and playback devices");
        return;
    }
    if args.iter().any(|arg| arg == "--list-devices") {
        let devices = rustvc_audio::list_devices();
        println!("inputs: {}", devices.inputs.join(", "));
        println!("outputs: {}", devices.outputs.join(", "));
        println!(
            "default input: {}",
            devices.default_input.unwrap_or_default()
        );
        println!(
            "default output: {}",
            devices.default_output.unwrap_or_default()
        );
        return;
    }
    if args.iter().any(|arg| arg == "--self-test") {
        if let Err(err) = self_test() {
            eprintln!("self-test failed: {err}");
            std::process::exit(1);
        }
        println!("self-test passed");
        return;
    }
    if let Err(err) = ui::run() {
        eprintln!("failed to open the window: {err}");
        std::process::exit(1);
    }
}

fn self_test() -> Result<(), String> {
    rustvc_audio::self_test()?;
    println!("audio loop: passthrough and resample ok");
    rustvc_infer::self_test()?;
    println!("inference hook: pitch bins and passthrough ok");
    dry_run_training()?;
    println!("training flow: dry run wrote a voice manifest");
    Ok(())
}

fn dry_run_training() -> Result<(), String> {
    let script = find_script().ok_or("python/rvc_train.py is missing")?;
    let python = if command_ok("python3") {
        "python3"
    } else if command_ok("python") {
        "python"
    } else {
        return Err("python3 is not on PATH".into());
    };
    let root = std::env::temp_dir().join(format!("rustvc-selftest-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    let dataset = root.join("dataset");
    let voices = root.join("voices");
    fs::create_dir_all(&dataset).map_err(|err| err.to_string())?;
    write_silence_wav(&dataset.join("clip.wav"))?;
    let mut job = TrainHandle::spawn(TrainRequest {
        python: python.to_string(),
        script,
        rvc_root: None,
        name: "SelfTest".into(),
        dataset,
        epochs: 1,
        batch_size: 1,
        sample_rate: 40_000,
        gpu: 0,
        version: "v2".into(),
        f0: true,
        voices_dir: voices.clone(),
        dry_run: true,
        workers: 1,
        sleep_secs: Some(0.0),
    })
    .map_err(|err| err.to_string())?;

    let started = Instant::now();
    let mut saw_progress = false;
    loop {
        for event in job.poll() {
            match event {
                TrainEvent::Progress { .. } => saw_progress = true,
                TrainEvent::Error(message) => return Err(message),
                TrainEvent::Finished { code } if code != 0 => {
                    return Err(format!("trainer exited {code}"));
                }
                TrainEvent::Finished { .. } => {
                    let manifest = voices.join("SelfTest").join("voice.toml");
                    if !manifest.is_file() {
                        return Err(format!("missing {}", manifest.display()));
                    }
                    if !saw_progress {
                        return Err("trainer finished without a PROGRESS line".into());
                    }
                    let _ = fs::remove_dir_all(&root);
                    return Ok(());
                }
                TrainEvent::Log(_) | TrainEvent::Voice(_) => {}
            }
        }
        if started.elapsed() > Duration::from_secs(30) {
            return Err("training dry run timed out".into());
        }
        thread::sleep(Duration::from_millis(20));
    }
}

fn command_ok(program: &str) -> bool {
    Command::new(program)
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn write_silence_wav(path: &PathBuf) -> Result<(), String> {
    let samples = 160u32;
    let data_len = samples * 2;
    let mut bytes = Vec::with_capacity(44 + data_len as usize);
    bytes.extend(b"RIFF");
    bytes.extend(&(36 + data_len).to_le_bytes());
    bytes.extend(b"WAVEfmt ");
    bytes.extend(&16u32.to_le_bytes());
    bytes.extend(&1u16.to_le_bytes());
    bytes.extend(&1u16.to_le_bytes());
    bytes.extend(&16_000u32.to_le_bytes());
    bytes.extend(&32_000u32.to_le_bytes());
    bytes.extend(&2u16.to_le_bytes());
    bytes.extend(&16u16.to_le_bytes());
    bytes.extend(b"data");
    bytes.extend(&data_len.to_le_bytes());
    bytes.resize(bytes.len() + data_len as usize, 0);
    fs::write(path, bytes).map_err(|err| err.to_string())
}
