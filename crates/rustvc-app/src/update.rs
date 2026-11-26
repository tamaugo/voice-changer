//! One click: check GitHub, download the installer, refresh voices, restart.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

use serde::Deserialize;

use crate::brand::{self, GITHUB_REPO, INSTALLER_ASSET};
use crate::setup::{data_dir, download_file};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReleaseAsset {
    pub version: String,
    pub name: String,
    pub url: String,
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpdatePlan {
    pub download: Option<ReleaseAsset>,
    pub already_current: bool,
    pub note: String,
}

pub enum UpdateEvent {
    Progress { fraction: f32, message: String },
    Log(String),
    Voices(PathBuf),
    Restart,
    Finished(Result<(), String>),
}

pub struct UpdateHandle {
    rx: Receiver<UpdateEvent>,
}

impl UpdateHandle {
    pub fn poll(&self) -> Option<UpdateEvent> {
        match self.rx.try_recv() {
            Ok(event) => Some(event),
            Err(_) => None,
        }
    }
}

pub fn start(local_version: String, voices_root: PathBuf, exe: PathBuf) -> UpdateHandle {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let result = run(&local_version, &voices_root, &exe, &tx);
        let _ = tx.send(UpdateEvent::Finished(result));
    });
    UpdateHandle { rx }
}

fn run(
    local_version: &str,
    voices_root: &Path,
    exe: &Path,
    tx: &Sender<UpdateEvent>,
) -> Result<(), String> {
    send_progress(tx, 0.05, "Checking GitHub");
    let release = fetch_latest_release()?;
    let plan = plan_update(local_version, release.as_ref());
    send_log(tx, &plan.note);
    let mut installer = None;
    if let Some(asset) = &plan.download {
        send_progress(tx, 0.1, "Downloading installer");
        let dest = data_dir().join("downloads").join(INSTALLER_ASSET);
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent).map_err(|err| err.to_string())?;
        }
        download_file(&asset.url, &dest, |fraction, _| {
            send_progress(tx, 0.1 + fraction * 0.6, "Downloading installer");
        })?;
        installer = Some(dest);
    }
    send_progress(tx, 0.75, "Updating voices");
    match refresh_voices(voices_root) {
        Ok(dir) => {
            send_log(tx, &format!("voices: {}", dir.display()));
            let _ = tx.send(UpdateEvent::Voices(dir));
        }
        Err(err) => send_log(tx, &format!("voices: {err}")),
    }
    if let Some(installer) = installer {
        if cfg!(windows) {
            let helper = write_windows_helper(
                &data_dir().join("update"),
                std::process::id(),
                &installer,
                exe,
                git_root(voices_root).as_deref(),
            )?;
            send_progress(tx, 0.95, "Installing and restarting");
            Command::new("cmd")
                .args(["/C", "start", "", &helper.to_string_lossy()])
                .spawn()
                .map_err(|err| err.to_string())?;
            let _ = tx.send(UpdateEvent::Restart);
        } else {
            send_log(
                tx,
                &format!(
                    "Installer downloaded to {}. It runs on Windows.",
                    installer.display()
                ),
            );
        }
    }
    send_progress(tx, 1.0, "Finished");
    Ok(())
}

pub fn fetch_latest_release() -> Result<Option<ReleaseAsset>, String> {
    let url = format!("https://api.github.com/repos/{GITHUB_REPO}/releases/latest");
    let response = ureq::get(&url)
        .set("User-Agent", "Lark")
        .set("Accept", "application/vnd.github+json")
        .call();
    match response {
        Ok(response) => {
            let body = response.into_string().map_err(|err| err.to_string())?;
            parse_latest_release(&body).map(Some)
        }
        Err(ureq::Error::Status(404, _)) => Ok(None),
        Err(err) => Err(err.to_string()),
    }
}

pub fn plan_update(local_version: &str, release: Option<&ReleaseAsset>) -> UpdatePlan {
    let Some(release) = release else {
        return UpdatePlan {
            download: None,
            already_current: false,
            note: format!("No GitHub Release yet. This copy is {local_version}."),
        };
    };
    if version_is_newer(&release.version, local_version) {
        UpdatePlan {
            download: Some(release.clone()),
            already_current: false,
            note: format!("Updating {} to {}.", local_version, release.version),
        }
    } else {
        UpdatePlan {
            download: None,
            already_current: true,
            note: format!("{} is current ({}).", brand::APP_NAME, local_version),
        }
    }
}

pub fn parse_latest_release(body: &str) -> Result<ReleaseAsset, String> {
    let parsed: ReleaseJson = serde_json::from_str(body).map_err(|err| err.to_string())?;
    let asset = parsed
        .assets
        .into_iter()
        .find(|asset| asset.name.eq_ignore_ascii_case(INSTALLER_ASSET))
        .ok_or_else(|| format!("{INSTALLER_ASSET} is not attached to {}", parsed.tag_name))?;
    Ok(ReleaseAsset {
        version: parsed.tag_name.trim_start_matches('v').to_string(),
        name: asset.name,
        url: asset.browser_download_url,
        bytes: asset.size,
    })
}

pub fn version_is_newer(remote: &str, local: &str) -> bool {
    let remote = parts(remote);
    let local = parts(local);
    remote > local
}

fn parts(version: &str) -> Vec<u32> {
    let version = version.trim().trim_start_matches('v');
    let mut numbers: Vec<u32> = version
        .split(|c: char| !c.is_ascii_digit())
        .filter(|part| !part.is_empty())
        .filter_map(|part| part.parse().ok())
        .collect();
    while numbers.len() < 3 {
        numbers.push(0);
    }
    numbers.truncate(3);
    numbers
}

pub fn git_root(start: &Path) -> Option<PathBuf> {
    let mut dir = Some(start);
    while let Some(current) = dir {
        if current.join(".git").exists() {
            return Some(current.to_path_buf());
        }
        dir = current.parent();
    }
    None
}

pub fn refresh_voices(voices_root: &Path) -> Result<PathBuf, String> {
    if let Some(root) = git_root(voices_root) {
        git(&["-C", &root.to_string_lossy(), "pull", "--ff-only"])?;
        git(&["-C", &root.to_string_lossy(), "lfs", "pull"])?;
        return Ok(voices_root.to_path_buf());
    }
    let dest = data_dir().join("checkout");
    if dest.join(".git").exists() {
        git(&["-C", &dest.to_string_lossy(), "pull", "--ff-only"])?;
        git(&["-C", &dest.to_string_lossy(), "lfs", "pull"])?;
    } else {
        if dest.exists() {
            fs::remove_dir_all(&dest).map_err(|err| err.to_string())?;
        }
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent).map_err(|err| err.to_string())?;
        }
        git(&[
            "clone",
            "--depth",
            "1",
            &format!("https://github.com/{GITHUB_REPO}.git"),
            &dest.to_string_lossy(),
        ])?;
        git(&["-C", &dest.to_string_lossy(), "lfs", "pull"])?;
    }
    Ok(dest.join("voices"))
}

fn git(args: &[&str]) -> Result<(), String> {
    let status = Command::new("git")
        .args(args)
        .status()
        .map_err(|err| err.to_string())?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("git {} exited {status}", args.join(" ")))
    }
}

pub fn write_windows_helper(
    dir: &Path,
    pid: u32,
    installer: &Path,
    exe: &Path,
    voices_git: Option<&Path>,
) -> Result<PathBuf, String> {
    fs::create_dir_all(dir).map_err(|err| err.to_string())?;
    let script = dir.join("restart.bat");
    let mut body = format!(
        "@echo off\r\n:wait\r\ntasklist /FI \"PID eq {pid}\" | find \"{pid}\" >nul\r\nif not errorlevel 1 (\r\n  timeout /t 1 /nobreak >nul\r\n  goto wait\r\n)\r\n"
    );
    if let Some(repo) = voices_git {
        body.push_str(&format!(
            "git -C \"{}\" pull --ff-only\r\ngit -C \"{}\" lfs pull\r\n",
            repo.display(),
            repo.display()
        ));
    }
    body.push_str(&format!(
        "start /wait \"\" \"{}\" /VERYSILENT /SUPPRESSMSGBOXES /NORESTART\r\nstart \"\" \"{}\"\r\n",
        installer.display(),
        exe.display()
    ));
    fs::write(&script, body).map_err(|err| err.to_string())?;
    Ok(script)
}

fn send_progress(tx: &Sender<UpdateEvent>, fraction: f32, message: &str) {
    let _ = tx.send(UpdateEvent::Progress {
        fraction,
        message: message.to_string(),
    });
}

fn send_log(tx: &Sender<UpdateEvent>, line: &str) {
    let _ = tx.send(UpdateEvent::Log(line.to_string()));
}

#[derive(Deserialize)]
struct ReleaseJson {
    tag_name: String,
    assets: Vec<AssetJson>,
}

#[derive(Deserialize)]
struct AssetJson {
    name: String,
    browser_download_url: String,
    #[serde(default)]
    size: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newer_versions_compare_numeric_parts() {
        assert!(version_is_newer("0.2.1", "0.2.0"));
        assert!(version_is_newer("v0.3.0", "0.2.9"));
        assert!(!version_is_newer("0.2.0", "0.2.0"));
        assert!(!version_is_newer("0.2", "0.2.0"));
        assert!(!version_is_newer("0.1.9", "0.2.0"));
    }

    #[test]
    fn release_json_picks_the_installer_asset() {
        let body = r#"{
            "tag_name": "v0.2.1",
            "assets": [
                {"name": "notes.txt", "browser_download_url": "https://example.test/notes", "size": 4},
                {"name": "Lark-Setup.exe", "browser_download_url": "https://example.test/Lark-Setup.exe", "size": 9}
            ]
        }"#;
        let release = parse_latest_release(body).unwrap();
        assert_eq!(release.version, "0.2.1");
        assert_eq!(release.url, "https://example.test/Lark-Setup.exe");
        let plan = plan_update("0.2.0", Some(&release));
        assert!(plan.download.is_some());
        assert!(!plan.already_current);
        let same = plan_update("0.2.1", Some(&release));
        assert!(same.download.is_none());
        assert!(same.already_current);
        assert!(plan_update("0.2.0", None).download.is_none());
    }

    #[test]
    fn helper_waits_for_the_process_then_installs_silently() {
        let dir = std::env::temp_dir().join(format!("lark-helper-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let script = write_windows_helper(
            &dir,
            4242,
            Path::new("C:/Temp/Lark-Setup.exe"),
            Path::new("C:/Users/me/AppData/Local/Lark/voice-changer.exe"),
            Some(Path::new("C:/src/voice-changer")),
        )
        .unwrap();
        let text = fs::read_to_string(script).unwrap();
        assert!(text.contains("PID eq 4242"));
        assert!(text.contains("/VERYSILENT"));
        assert!(text.contains("git -C \"C:/src/voice-changer\" lfs pull"));
        assert!(text.contains("voice-changer.exe"));
        let _ = fs::remove_dir_all(&dir);
    }
}
