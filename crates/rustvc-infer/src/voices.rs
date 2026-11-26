//! Voice manifests stored as `voices/<name>/voice.toml`.
//!
//! The built-in passthrough voice is not a file. Everything the trainer writes
//! is one of these manifests plus, once training really ran, a `.pth` or
//! `.onnx` checkpoint beside it.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceManifest {
    pub name: String,
    #[serde(default = "default_version")]
    pub version: String,
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,
    #[serde(default = "default_true")]
    pub f0: bool,
    #[serde(default)]
    pub pitch: f32,
    #[serde(default = "default_index_rate")]
    pub index_rate: f32,
    #[serde(default = "default_f0_method")]
    pub f0_method: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub index: String,
    #[serde(default)]
    pub content_model: String,
    #[serde(default)]
    pub f0_model: String,
    #[serde(default)]
    pub trained: bool,
    #[serde(default)]
    pub note: String,
    /// Brute-force retrieval bank written by the exporter (`features.npy`).
    #[serde(default)]
    pub features: String,
    /// 0 leaves the converted loudness alone. 1 matches the input envelope.
    #[serde(default = "default_rms_mix")]
    pub rms_mix: f32,
    /// Consonant protection, 0 to 0.5. Unvoiced frames keep more of the original features.
    #[serde(default = "default_protect")]
    pub protect: f32,
    #[serde(default = "default_crossfade_ms")]
    pub crossfade_ms: f32,
}

fn default_version() -> String {
    "v2".to_string()
}
fn default_sample_rate() -> u32 {
    40_000
}
fn default_true() -> bool {
    true
}
fn default_index_rate() -> f32 {
    0.75
}
fn default_f0_method() -> String {
    "rmvpe".to_string()
}
fn default_rms_mix() -> f32 {
    0.25
}
fn default_protect() -> f32 {
    0.33
}
fn default_crossfade_ms() -> f32 {
    10.0
}

impl Default for VoiceManifest {
    fn default() -> Self {
        Self {
            name: "Voice".to_string(),
            version: default_version(),
            sample_rate: default_sample_rate(),
            f0: true,
            pitch: 0.0,
            index_rate: default_index_rate(),
            f0_method: default_f0_method(),
            model: String::new(),
            index: String::new(),
            content_model: String::new(),
            f0_model: String::new(),
            trained: false,
            note: String::new(),
            features: String::new(),
            rms_mix: default_rms_mix(),
            protect: default_protect(),
            crossfade_ms: default_crossfade_ms(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VoiceRecord {
    pub directory: PathBuf,
    pub manifest: VoiceManifest,
}

impl VoiceRecord {
    pub fn resolve(&self, relative: &str) -> Option<PathBuf> {
        if relative.is_empty() {
            return None;
        }
        let path = PathBuf::from(relative);
        let full = if path.is_absolute() {
            path
        } else {
            self.directory.join(path)
        };
        full.is_file().then_some(full)
    }

    /// Synthesizer checkpoint. An exported ONNX file wins over a `.pth`.
    pub fn synthesizer(&self) -> Option<PathBuf> {
        existing(self.directory.join("synthesizer.onnx"))
            .or_else(|| onnx_only(self.resolve(&self.manifest.model)))
            .or_else(|| self.resolve(&self.manifest.model))
            .or_else(|| existing(self.directory.join("model.pth")))
    }

    pub fn content_model(&self) -> Option<PathBuf> {
        self.resolve(&self.manifest.content_model)
            .or_else(|| existing(self.directory.join("content.onnx")))
    }

    pub fn f0_model(&self) -> Option<PathBuf> {
        self.resolve(&self.manifest.f0_model)
            .or_else(|| existing(self.directory.join("f0.onnx")))
    }

    pub fn features(&self) -> Option<PathBuf> {
        self.resolve(&self.manifest.features)
            .or_else(|| existing(self.directory.join("features.npy")))
    }
}

fn onnx_only(path: Option<PathBuf>) -> Option<PathBuf> {
    path.filter(|candidate| candidate.extension().and_then(|ext| ext.to_str()) == Some("onnx"))
}

fn existing(path: PathBuf) -> Option<PathBuf> {
    path.is_file().then_some(path)
}

/// Read every `voice.toml` directly under `root`. Broken files are skipped.
pub fn load_voices(root: &Path) -> Vec<VoiceRecord> {
    let entries = match fs::read_dir(root) {
        Ok(entries) => entries,
        Err(_) => return Vec::new(),
    };
    let mut voices = Vec::new();
    for entry in entries.flatten() {
        let directory = entry.path();
        if !directory.is_dir() {
            continue;
        }
        let manifest_path = directory.join("voice.toml");
        let Ok(text) = fs::read_to_string(&manifest_path) else {
            continue;
        };
        let Ok(manifest) = toml::from_str::<VoiceManifest>(&text) else {
            continue;
        };
        voices.push(VoiceRecord {
            directory,
            manifest,
        });
    }
    voices.sort_by(|a, b| {
        a.manifest
            .name
            .to_ascii_lowercase()
            .cmp(&b.manifest.name.to_ascii_lowercase())
    });
    voices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_manifest_and_scan() {
        let root = std::env::temp_dir().join(format!("rustvc-voices-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let dir = root.join("ada");
        fs::create_dir_all(&dir).unwrap();
        let manifest = VoiceManifest {
            name: "Ada".to_string(),
            trained: false,
            note: "scaffold".to_string(),
            ..VoiceManifest::default()
        };
        fs::write(dir.join("voice.toml"), toml::to_string(&manifest).unwrap()).unwrap();
        let loaded = load_voices(&root);
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].manifest.name, "Ada");
        assert!(!loaded[0].manifest.trained);
        let _ = fs::remove_dir_all(&root);
    }
}
