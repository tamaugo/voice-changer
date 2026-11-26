//! First-run dependency install. The window drives this. No shell for the user.
#![cfg_attr(not(windows), allow(dead_code))]

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
#[cfg(windows)]
use std::process::{Command, Stdio};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

pub const PYTHON_URL: &str = "https://www.python.org/ftp/python/3.12.8/python-3.12.8-amd64.exe";
pub const GIT_URL: &str =
    "https://github.com/git-for-windows/git/releases/download/v2.47.1.windows.2/Git-2.47.1.2-64-bit.exe";
pub const GIT_LFS_URL: &str =
    "https://github.com/git-lfs/git-lfs/releases/download/v3.6.1/git-lfs-windows-v3.6.1.exe";
pub const ORT_URL: &str =
    "https://globalcdn.nuget.org/packages/microsoft.ml.onnxruntime.gpu.windows.1.26.0.nupkg";
pub const VBCABLE_URL: &str =
    "https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack46.zip";
pub const RVC_REPO: &str =
    "https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git";
pub const HF_BASE: &str = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main";

pub struct Step {
    pub id: &'static str,
    pub label: &'static str,
}

pub const STEPS: &[Step] = &[
    Step {
        id: "git",
        label: "Git",
    },
    Step {
        id: "git-lfs",
        label: "Git LFS",
    },
    Step {
        id: "python",
        label: "Python 3.12",
    },
    Step {
        id: "venv",
        label: "Python virtual environment",
    },
    Step {
        id: "rvc",
        label: "RVC WebUI",
    },
    Step {
        id: "models",
        label: "RVC model files",
    },
    Step {
        id: "onnx",
        label: "ONNX Runtime GPU 1.26",
    },
    Step {
        id: "cable",
        label: "VB-CABLE",
    },
];

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HostFacts {
    pub git: bool,
    pub git_lfs: bool,
    pub python312: bool,
    pub venv: bool,
    pub rvc: bool,
    pub models: bool,
    pub onnx: bool,
    pub cable: bool,
}

pub fn missing(facts: &HostFacts) -> Vec<&'static str> {
    let mut pending = Vec::new();
    for (present, id) in [
        (facts.git, "git"),
        (facts.git_lfs, "git-lfs"),
        (facts.python312, "python"),
        (facts.venv, "venv"),
        (facts.rvc, "rvc"),
        (facts.models, "models"),
        (facts.onnx, "onnx"),
        (facts.cable, "cable"),
    ] {
        if !present {
            pending.push(id);
        }
    }
    pending
}

pub struct SetupPaths {
    pub root: PathBuf,
}

impl SetupPaths {
    pub fn new() -> Self {
        Self { root: data_dir() }
    }

    pub fn venv_python(&self) -> PathBuf {
        if cfg!(windows) {
            self.root.join("venv").join("Scripts").join("python.exe")
        } else {
            self.root.join("venv").join("bin").join("python")
        }
    }

    pub fn rvc(&self) -> PathBuf {
        self.root.join("rvc")
    }

    pub fn onnx_dll(&self) -> PathBuf {
        self.root.join("onnx").join("onnxruntime.dll")
    }
}

pub fn data_dir() -> PathBuf {
    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        if !local.is_empty() {
            return PathBuf::from(local).join("Lark");
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home)
            .join(".local")
            .join("share")
            .join("Lark");
    }
    std::env::temp_dir().join("Lark")
}

#[derive(Debug)]
pub struct SetupResult {
    pub python: String,
    pub rvc_root: String,
    pub onnx_dll: String,
}

pub enum SetupEvent {
    Progress { fraction: f32, message: String },
    Log(String),
    Finished(Result<SetupResult, String>),
}

pub struct SetupHandle {
    rx: Receiver<SetupEvent>,
}

impl SetupHandle {
    pub fn poll(&self) -> Option<SetupEvent> {
        match self.rx.try_recv() {
            Ok(event) => Some(event),
            Err(_) => None,
        }
    }
}

pub fn start(paths: SetupPaths, cable_present: bool) -> SetupHandle {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let result = run(&paths, cable_present, &tx);
        let _ = tx.send(SetupEvent::Finished(result));
    });
    SetupHandle { rx }
}

fn run(
    paths: &SetupPaths,
    cable_present: bool,
    tx: &Sender<SetupEvent>,
) -> Result<SetupResult, String> {
    #[cfg(windows)]
    {
        return install_windows(paths, cable_present, tx);
    }
    #[cfg(not(windows))]
    {
        let _ = (paths, cable_present, tx);
        Err("Dependency setup runs on Windows.".into())
    }
}

#[cfg(windows)]
fn install_windows(
    paths: &SetupPaths,
    cable_present: bool,
    tx: &Sender<SetupEvent>,
) -> Result<SetupResult, String> {
    fs::create_dir_all(&paths.root).map_err(|err| err.to_string())?;
    let downloads = paths.root.join("downloads");
    fs::create_dir_all(&downloads).map_err(|err| err.to_string())?;

    step(tx, 0, "Git");
    if !command_ok("git", &["--version"]) {
        let exe = downloads.join("Git-Setup.exe");
        fetch(GIT_URL, &exe, tx)?;
        silent(&exe, &["/VERYSILENT", "/NORESTART", "/NOCANCEL"], tx)?;
    }
    let git = git_bin();

    step(tx, 1, "Git LFS");
    if !git_lfs_ok(&git) {
        let exe = downloads.join("git-lfs.exe");
        fetch(GIT_LFS_URL, &exe, tx)?;
        silent(&exe, &["/VERYSILENT", "/NORESTART"], tx)?;
        run_cmd(&git, &["lfs", "install"], tx)?;
    }

    step(tx, 2, "Python 3.12");
    if python312().is_none() {
        let exe = downloads.join("python-3.12.8-amd64.exe");
        fetch(PYTHON_URL, &exe, tx)?;
        silent(
            &exe,
            &[
                "/quiet",
                "InstallAllUsers=0",
                "PrependPath=1",
                "Include_pip=1",
                "Include_test=0",
                "Shortcuts=0",
            ],
            tx,
        )?;
    }
    let python = python312().ok_or("Python 3.12 installed but was not found on disk")?;

    step(tx, 3, "Python virtual environment");
    let venv_python = paths.venv_python();
    if !venv_python.is_file() {
        let venv = paths.root.join("venv");
        let venv_arg = venv.to_string_lossy().to_string();
        run_cmd(&python, &["-m", "venv", &venv_arg], tx)?;
        run_cmd(
            &venv_python,
            &["-m", "pip", "install", "--upgrade", "pip"],
            tx,
        )?;
    }

    step(tx, 4, "RVC WebUI");
    let rvc = paths.rvc();
    if !rvc.join("infer-web.py").is_file() {
        let rvc_arg = rvc.to_string_lossy().to_string();
        run_cmd(&git, &["clone", "--depth", "1", RVC_REPO, &rvc_arg], tx)?;
    }
    let requirements = rvc.join("requirments_cu128_py312.txt");
    if requirements.is_file() {
        run_cmd(
            &venv_python,
            &[
                "-m",
                "pip",
                "install",
                "torch",
                "torchvision",
                "torchaudio",
                "--index-url",
                "https://download.pytorch.org/whl/cu128",
            ],
            tx,
        )?;
        let req = requirements.to_string_lossy().to_string();
        run_cmd(&venv_python, &["-m", "pip", "install", "-r", &req], tx)?;
    }

    step(tx, 5, "RVC model files");
    download_models(&rvc, tx)?;

    step(tx, 6, "ONNX Runtime GPU 1.26");
    let dll = paths.onnx_dll();
    if !dll.is_file() {
        let nupkg = downloads.join("onnxruntime.nupkg");
        fetch(ORT_URL, &nupkg, tx)?;
        let unpacked = downloads.join("onnxruntime");
        extract_zip(&nupkg, &unpacked)?;
        let found = find_file(&unpacked, "onnxruntime.dll")
            .ok_or("onnxruntime.dll was not in the package")?;
        if let Some(dir) = found.parent() {
            copy_dir_files(dir, &paths.root.join("onnx"))?;
        }
    }

    step(tx, 7, "VB-CABLE");
    if !cable_present {
        let zip = downloads.join("VBCABLE.zip");
        fetch(VBCABLE_URL, &zip, tx)?;
        let unpacked = downloads.join("vbcable");
        extract_zip(&zip, &unpacked)?;
        let setup = find_file(&unpacked, "VBCABLE_Setup_x64.exe")
            .ok_or("VBCABLE_Setup_x64.exe was not in the zip")?;
        log(
            tx,
            "Opening the VB-CABLE installer. Approve the driver window, then reboot.",
        );
        let _ = Command::new("powershell")
            .args([
                "-NoProfile",
                "-Command",
                &format!(
                    "Start-Process -FilePath '{}' -Verb RunAs",
                    setup.display().to_string().replace('\'', "''")
                ),
            ])
            .spawn()
            .map_err(|err| err.to_string())?;
    }

    Ok(SetupResult {
        python: venv_python.display().to_string(),
        rvc_root: rvc.display().to_string(),
        onnx_dll: dll.display().to_string(),
    })
}

#[cfg(windows)]
fn step(tx: &Sender<SetupEvent>, index: usize, label: &str) {
    let fraction = index as f32 / STEPS.len() as f32;
    let _ = tx.send(SetupEvent::Progress {
        fraction,
        message: label.to_string(),
    });
    log(tx, label);
}

#[cfg(windows)]
fn log(tx: &Sender<SetupEvent>, line: &str) {
    let _ = tx.send(SetupEvent::Log(line.to_string()));
}

#[cfg(windows)]
fn fetch(url: &str, dest: &Path, tx: &Sender<SetupEvent>) -> Result<(), String> {
    if dest.is_file() && fs::metadata(dest).map(|m| m.len() > 0).unwrap_or(false) {
        log(tx, &format!("already downloaded {}", dest.display()));
        return Ok(());
    }
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(|err| err.to_string())?;
    }
    download_file(url, dest, |fraction, message| {
        let _ = tx.send(SetupEvent::Progress {
            fraction,
            message: message.to_string(),
        });
    })
}

pub fn download_file(
    url: &str,
    dest: &Path,
    mut progress: impl FnMut(f32, &str),
) -> Result<(), String> {
    let response = ureq::get(url)
        .set("User-Agent", "Lark")
        .call()
        .map_err(|err| err.to_string())?;
    let total = response
        .header("content-length")
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(0);
    let mut reader = response.into_reader();
    let mut file = fs::File::create(dest).map_err(|err| err.to_string())?;
    let mut buf = [0u8; 64 * 1024];
    let mut got = 0u64;
    loop {
        let read = reader.read(&mut buf).map_err(|err| err.to_string())?;
        if read == 0 {
            break;
        }
        file.write_all(&buf[..read])
            .map_err(|err| err.to_string())?;
        got += read as u64;
        let fraction = if total == 0 {
            0.0
        } else {
            got as f32 / total as f32
        };
        progress(fraction, &format!("{url}  {got}"));
    }
    Ok(())
}

pub fn extract_zip(zip_path: &Path, dest: &Path) -> Result<(), String> {
    let file = fs::File::open(zip_path).map_err(|err| err.to_string())?;
    let mut archive = zip::ZipArchive::new(file).map_err(|err| err.to_string())?;
    fs::create_dir_all(dest).map_err(|err| err.to_string())?;
    for index in 0..archive.len() {
        let mut entry = archive.by_index(index).map_err(|err| err.to_string())?;
        let name = entry.name().to_string();
        if name.contains("..") {
            return Err(format!("zip entry escaped the folder: {name}"));
        }
        let out = dest.join(&name);
        if entry.is_dir() {
            fs::create_dir_all(&out).map_err(|err| err.to_string())?;
            continue;
        }
        if let Some(parent) = out.parent() {
            fs::create_dir_all(parent).map_err(|err| err.to_string())?;
        }
        let mut file = fs::File::create(&out).map_err(|err| err.to_string())?;
        std::io::copy(&mut entry, &mut file).map_err(|err| err.to_string())?;
    }
    Ok(())
}

#[cfg(windows)]
fn silent(exe: &Path, args: &[&str], tx: &Sender<SetupEvent>) -> Result<(), String> {
    log(tx, &format!("running {}", exe.display()));
    let status = Command::new(exe)
        .args(args)
        .status()
        .map_err(|err| err.to_string())?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("{} exited {status}", exe.display()))
    }
}

#[cfg(windows)]
fn run_cmd(program: &Path, args: &[&str], tx: &Sender<SetupEvent>) -> Result<(), String> {
    log(tx, &format!("{} {}", program.display(), args.join(" ")));
    let status = Command::new(program)
        .args(args)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .map_err(|err| err.to_string())?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("{} exited {status}", program.display()))
    }
}

#[cfg(windows)]
fn command_ok(program: &str, args: &[&str]) -> bool {
    Command::new(program)
        .args(args)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

#[cfg(windows)]
fn git_bin() -> PathBuf {
    let program_files =
        std::env::var("ProgramFiles").unwrap_or_else(|_| "C:\\Program Files".into());
    let installed = PathBuf::from(program_files)
        .join("Git")
        .join("cmd")
        .join("git.exe");
    if installed.is_file() {
        installed
    } else {
        PathBuf::from("git")
    }
}

#[cfg(windows)]
fn git_lfs_ok(git: &Path) -> bool {
    Command::new(git)
        .args(["lfs", "version"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

#[cfg(windows)]
fn python312() -> Option<PathBuf> {
    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        let installed = PathBuf::from(local)
            .join("Programs")
            .join("Python")
            .join("Python312")
            .join("python.exe");
        if installed.is_file() {
            return Some(installed);
        }
    }
    let output = Command::new("py")
        .args(["-3.12", "-c", "import sys; print(sys.executable)"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let path = String::from_utf8(output.stdout).ok()?;
    let path = PathBuf::from(path.trim());
    path.is_file().then_some(path)
}

#[cfg(windows)]
fn download_models(rvc: &Path, tx: &Sender<SetupEvent>) -> Result<(), String> {
    let files = [
        ("hubert_base/config.json", "assets/hubert_base/config.json"),
        (
            "hubert_base/preprocessor_config.json",
            "assets/hubert_base/preprocessor_config.json",
        ),
        (
            "hubert_base/pytorch_model.bin",
            "assets/hubert_base/pytorch_model.bin",
        ),
        ("rmvpe.pt", "assets/rmvpe/rmvpe.pt"),
        (
            "pretrained_v2/f0G40k.pth",
            "assets/pretrained_v2/f0G40k.pth",
        ),
        (
            "pretrained_v2/f0D40k.pth",
            "assets/pretrained_v2/f0D40k.pth",
        ),
        ("pretrained_v2/G40k.pth", "assets/pretrained_v2/G40k.pth"),
        ("pretrained_v2/D40k.pth", "assets/pretrained_v2/D40k.pth"),
    ];
    for (remote, relative) in files {
        let dest = rvc.join(relative);
        if dest.is_file() {
            continue;
        }
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent).map_err(|err| err.to_string())?;
        }
        let url = format!("{HF_BASE}/{remote}");
        fetch(&url, &dest, tx)?;
    }
    Ok(())
}

#[cfg(windows)]
fn find_file(dir: &Path, name: &str) -> Option<PathBuf> {
    let entries = fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if let Some(found) = find_file(&path, name) {
                return Some(found);
            }
        } else if path.file_name().and_then(|n| n.to_str()) == Some(name) {
            return Some(path);
        }
    }
    None
}

#[cfg(windows)]
fn copy_dir_files(from: &Path, to: &Path) -> Result<(), String> {
    fs::create_dir_all(to).map_err(|err| err.to_string())?;
    for entry in fs::read_dir(from).map_err(|err| err.to_string())? {
        let entry = entry.map_err(|err| err.to_string())?;
        let path = entry.path();
        if path.is_file() {
            let dest = to.join(entry.file_name());
            fs::copy(&path, dest).map_err(|err| err.to_string())?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_lists_only_absent_steps() {
        let none = HostFacts::default();
        assert_eq!(missing(&none).len(), STEPS.len());
        let mut ready = HostFacts {
            git: true,
            git_lfs: true,
            python312: true,
            venv: true,
            rvc: true,
            models: true,
            onnx: true,
            cable: true,
        };
        assert!(missing(&ready).is_empty());
        ready.cable = false;
        ready.onnx = false;
        assert_eq!(missing(&ready), ["onnx", "cable"]);
    }

    #[test]
    fn urls_pin_the_windows_versions() {
        assert!(PYTHON_URL.contains("python-3.12.8"));
        assert!(GIT_URL.contains("Git-2.47.1.2-64-bit.exe"));
        assert!(ORT_URL.contains("1.26.0"));
        assert!(VBCABLE_URL.contains("VBCABLE_Driver_Pack46.zip"));
        assert!(RVC_REPO.contains("Retrieval-based-Voice-Conversion-WebUI"));
    }

    #[test]
    fn extract_zip_rejects_parent_paths_and_writes_files() {
        let root = std::env::temp_dir().join(format!("lark-zip-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let zip_path = root.join("sample.zip");
        {
            let file = fs::File::create(&zip_path).unwrap();
            let mut writer = zip::ZipWriter::new(file);
            let options = zip::write::SimpleFileOptions::default();
            writer.start_file("onnxruntime.dll", options).unwrap();
            writer.write_all(b"dll").unwrap();
            writer.finish().unwrap();
        }
        let dest = root.join("out");
        extract_zip(&zip_path, &dest).unwrap();
        assert_eq!(fs::read(dest.join("onnxruntime.dll")).unwrap(), b"dll");
        let _ = fs::remove_dir_all(&root);
    }
}
