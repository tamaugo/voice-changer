//! The live voice hook.
//!
//! `process` always returns device-rate audio of the same length the engine
//! expects. A missing checkpoint stays on passthrough and says so in `label`,
//! so a game session keeps working while a voice is only half set up.

#[cfg(feature = "onnx")]
use rustvc_audio::BlockResampler;

#[cfg(feature = "onnx")]
use crate::dsp::rms_mix;
use crate::dsp::{Crossfade, FeatureBank};
use crate::npy::load_feature_bank;
use crate::voices::VoiceManifest;

/// Where live inference should run. Training has its own device choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferDevice {
    Cpu,
    /// CUDA execution provider. The ONNX Runtime build must include
    /// Blackwell kernels (CUDA 12.8+, sm_120) to use an RTX 5070.
    Cuda {
        device_id: i32,
    },
}

impl InferDevice {
    pub fn parse(name: &str, device_id: i32) -> Self {
        if name.eq_ignore_ascii_case("cuda") {
            Self::Cuda { device_id }
        } else {
            Self::Cpu
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda { .. } => "cuda",
        }
    }
}

/// Parameters passed into one conversion block.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct InferParams {
    pub version: String,
    pub model_sample_rate: u32,
    pub f0: bool,
    pub pitch_semitones: f32,
    pub index_rate: f32,
    pub speaker_id: i64,
    pub f0_method: String,
    pub rms_mix: f32,
    pub protect: f32,
    pub crossfade_samples: usize,
}

impl InferParams {
    pub fn from_manifest(manifest: &VoiceManifest) -> Self {
        Self {
            version: manifest.version.clone(),
            model_sample_rate: manifest.sample_rate,
            f0: manifest.f0,
            pitch_semitones: manifest.pitch,
            index_rate: manifest.index_rate,
            speaker_id: 0,
            f0_method: manifest.f0_method.clone(),
            rms_mix: manifest.rms_mix,
            protect: manifest.protect,
            crossfade_samples: ((manifest.crossfade_ms.max(0.0) / 1000.0)
                * manifest.sample_rate as f32) as usize,
        }
    }

    pub fn feature_dim(&self) -> usize {
        if self.version == "v1" {
            256
        } else {
            768
        }
    }
}

enum Backend {
    Passthrough {
        reason: String,
    },
    #[cfg(feature = "onnx")]
    Onnx(crate::onnx::OnnxSessions),
}

/// A selected voice plus the device rate of the capture stream.
pub struct VoiceModel {
    label: String,
    #[allow(dead_code)]
    device_sample_rate: u32,
    #[allow(dead_code)]
    params: InferParams,
    backend: Backend,
    #[cfg_attr(not(feature = "onnx"), allow(dead_code))]
    bank: Option<FeatureBank>,
    #[cfg_attr(not(feature = "onnx"), allow(dead_code))]
    crossfade: Crossfade,
    #[cfg(feature = "onnx")]
    to_content_rate: Option<BlockResampler>,
    #[cfg(feature = "onnx")]
    to_device_rate: Option<BlockResampler>,
    /// Recent capture audio so the encoder sees more than one block.
    #[cfg(feature = "onnx")]
    history: Vec<f32>,
    last_error: Option<String>,
}

impl VoiceModel {
    pub fn passthrough(device_sample_rate: u32) -> Self {
        Self {
            label: "Passthrough".to_string(),
            device_sample_rate,
            params: InferParams {
                version: "v2".to_string(),
                model_sample_rate: device_sample_rate,
                f0: false,
                pitch_semitones: 0.0,
                index_rate: 0.0,
                speaker_id: 0,
                f0_method: "rmvpe".to_string(),
                rms_mix: 0.0,
                protect: 0.0,
                crossfade_samples: 0,
            },
            backend: Backend::Passthrough {
                reason: "built-in passthrough".to_string(),
            },
            bank: None,
            crossfade: Crossfade::new(0),
            #[cfg(feature = "onnx")]
            to_content_rate: None,
            #[cfg(feature = "onnx")]
            to_device_rate: None,
            #[cfg(feature = "onnx")]
            history: Vec::new(),
            last_error: None,
        }
    }

    /// Load `record` if its ONNX sessions exist. Otherwise keep the audio
    /// path open and explain the gap in [`Self::label`].
    pub fn load(
        manifest: VoiceManifest,
        directory: &std::path::Path,
        device_sample_rate: u32,
        device: InferDevice,
        onnx_dll: Option<&std::path::Path>,
    ) -> Self {
        let params = InferParams::from_manifest(&manifest);
        let record = crate::voices::VoiceRecord {
            directory: directory.to_path_buf(),
            manifest: manifest.clone(),
        };
        let (backend, label) = open_backend(&record, &params, device, onnx_dll);
        let bank = record
            .features()
            .and_then(|path| load_feature_bank(&path).ok());
        let overlap =
            ((manifest.crossfade_ms.max(0.0) / 1000.0) * device_sample_rate as f32) as usize;
        Self {
            label,
            device_sample_rate,
            params,
            backend,
            bank,
            crossfade: Crossfade::new(overlap),
            #[cfg(feature = "onnx")]
            to_content_rate: None,
            #[cfg(feature = "onnx")]
            to_device_rate: None,
            #[cfg(feature = "onnx")]
            history: Vec::new(),
            last_error: None,
        }
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn last_error(&self) -> Option<&str> {
        self.last_error.as_deref()
    }

    #[allow(dead_code)]
    pub fn detail(&self) -> &str {
        match &self.backend {
            Backend::Passthrough { reason } => reason,
            #[cfg(feature = "onnx")]
            Backend::Onnx(_) => "onnx session loaded",
        }
    }

    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        match self.convert(input) {
            Ok(output) => {
                self.last_error = None;
                output
            }
            Err(err) => {
                self.last_error = Some(err.clone());
                if !self.label.contains("passthrough") {
                    self.label = format!("{} — passthrough ({err})", self.params_name());
                }
                input.to_vec()
            }
        }
    }

    fn params_name(&self) -> &str {
        self.label
            .split(" — ")
            .next()
            .unwrap_or(self.label.as_str())
    }

    fn convert(&mut self, input: &[f32]) -> Result<Vec<f32>, String> {
        match &mut self.backend {
            Backend::Passthrough { .. } => Ok(input.to_vec()),
            #[cfg(feature = "onnx")]
            Backend::Onnx(_) => self.run_onnx(input),
        }
    }

    #[cfg(feature = "onnx")]
    fn run_onnx(&mut self, input: &[f32]) -> Result<Vec<f32>, String> {
        if self.to_content_rate.is_none() {
            self.to_content_rate = Some(
                BlockResampler::new(self.device_sample_rate, 16_000)
                    .map_err(|err| err.to_string())?,
            );
        }
        if self.to_device_rate.is_none() {
            self.to_device_rate = Some(
                BlockResampler::new(self.params.model_sample_rate, self.device_sample_rate)
                    .map_err(|err| err.to_string())?,
            );
        }
        self.history.extend_from_slice(input);
        let context = self.device_sample_rate as usize * 3 / 10;
        let keep = context + input.len();
        if self.history.len() > keep {
            let drop_count = self.history.len() - keep;
            self.history.drain(..drop_count);
        }
        let window = self.history.clone();
        let audio_16k = self
            .to_content_rate
            .as_mut()
            .expect("content resampler")
            .process(&window)
            .map_err(|err| err.to_string())?;
        let converted = {
            let bank = self.bank.as_ref();
            let params = &self.params;
            let sessions = match &mut self.backend {
                Backend::Onnx(sessions) => sessions,
                Backend::Passthrough { .. } => return Ok(input.to_vec()),
            };
            sessions.run(&audio_16k, params, bank)?
        };
        let wet = self
            .to_device_rate
            .as_mut()
            .expect("device resampler")
            .process(&converted)
            .map_err(|err| err.to_string())?;
        let hop = ((wet.len() as f64) * (input.len() as f64) / (window.len().max(1) as f64))
            .round()
            .clamp(1.0, wet.len() as f64) as usize;
        let mut played = wet[wet.len() - hop..].to_vec();
        let count = played.len().min(input.len());
        rms_mix(input, &mut played[..count], self.params.rms_mix, 480);
        Ok(self.crossfade.apply(&played))
    }
}

fn open_backend(
    record: &crate::voices::VoiceRecord,
    params: &InferParams,
    device: InferDevice,
    onnx_dll: Option<&std::path::Path>,
) -> (Backend, String) {
    let name = record.manifest.name.clone();
    #[cfg(not(feature = "onnx"))]
    {
        let _ = (params, device, onnx_dll);
        let reason = if record.synthesizer().is_some() {
            "rebuild with --features onnx to load this checkpoint".to_string()
        } else if record.manifest.trained {
            "checkpoint is not an ONNX session yet".to_string()
        } else if record.manifest.note.trim().is_empty() {
            "weights have not been trained".to_string()
        } else {
            record.manifest.note.clone()
        };
        (
            Backend::Passthrough {
                reason: reason.clone(),
            },
            format!("{name} — passthrough ({reason})"),
        )
    }
    #[cfg(feature = "onnx")]
    {
        match crate::onnx::OnnxSessions::load(record, params, device, onnx_dll) {
            Ok(sessions) => {
                let where_ = match device {
                    InferDevice::Cpu => "onnx cpu".to_string(),
                    InferDevice::Cuda { device_id } => format!("onnx cuda:{device_id}"),
                };
                (Backend::Onnx(sessions), format!("{name} — {where_}"))
            }
            Err(err) => (
                Backend::Passthrough {
                    reason: err.clone(),
                },
                format!("{name} — passthrough ({err})"),
            ),
        }
    }
}
