//! Inference interface for live voice conversion.
//!
//! The live path is a [`VoiceModel`] installed as the audio engine's processor.
//! Until ONNX sessions for the content encoder, F0 model, and synthesizer are
//! present, the model reports why and passes audio through unchanged.
//!
//! The stage order follows the RVC v1/v2 contract published by the MIT-licensed
//! Retrieval-based Voice Conversion WebUI: 16 kHz content features, optional
//! retrieval blend, F0 in hertz plus a coarse 1..=255 bin, then the
//! synthesizer. so-vits-svc and the unlicensed obs-rvc tree are not linked.

mod bench;
mod budget;
#[cfg_attr(not(feature = "onnx"), allow(dead_code))]
mod dsp;
mod model;
#[cfg_attr(not(feature = "onnx"), allow(dead_code))]
mod npy;
mod pitch;
mod voices;

#[cfg(feature = "onnx")]
mod onnx;

pub use bench::{measure_duty, DutyReport};
pub use budget::{duty_cycle_percent, GpuBudget};
pub use model::{InferDevice, VoiceModel};
pub use pitch::{coarse_f0, shift_hz};
pub use voices::{load_voices, VoiceManifest, VoiceRecord};

use rustvc_audio::SampleProcessor;

/// Adapter so a [`VoiceModel`] can be installed in the audio engine.
pub struct ModelProcessor {
    model: VoiceModel,
}

impl ModelProcessor {
    pub fn new(model: VoiceModel) -> Self {
        Self { model }
    }

    pub fn model(&self) -> &VoiceModel {
        &self.model
    }
}

impl SampleProcessor for ModelProcessor {
    fn process(&mut self, input: &[f32]) -> Vec<f32> {
        self.model.process(input)
    }

    fn label(&self) -> &str {
        self.model.label()
    }
}

pub fn self_test() -> Result<(), String> {
    if coarse_f0(50.0) != 1 {
        return Err("50 Hz must quantize to RVC bin 1".into());
    }
    if coarse_f0(1100.0) != 255 {
        return Err("1100 Hz must quantize to RVC bin 255".into());
    }
    let shifted = shift_hz(440.0, 12.0);
    if (shifted - 880.0).abs() > 0.1 {
        return Err(format!("octave shift produced {shifted}"));
    }
    let budget = GpuBudget::new(5);
    if budget.allows(4.0, 200.0) {
        // 4 ms of work inside a 200 ms block is 2 percent.
    } else {
        return Err("2% duty cycle was rejected by a 5% budget".into());
    }
    if budget.allows(20.0, 100.0) {
        return Err("20% duty cycle was accepted by a 5% budget".into());
    }
    let params = crate::model::InferParams::from_manifest(&crate::voices::VoiceManifest::default());
    if params.feature_dim() != 768 {
        return Err("v2 voices must expose 768-dimensional content features".into());
    }
    let passthrough = VoiceModel::passthrough(48_000);
    let input = [0.25f32, -0.5, 0.0];
    if passthrough_samples(&passthrough, &input) != input {
        return Err("unloaded voice model did not pass audio through".into());
    }
    let report = measure_duty(|samples| samples.to_vec(), 48_000, 4_800, 4, 5);
    if !report.inside_budget {
        return Err(report.summary());
    }
    Ok(())
}

fn passthrough_samples(model: &VoiceModel, input: &[f32]) -> Vec<f32> {
    let mut owned = ModelProcessor::new(clone_passthrough(model));
    owned.process(input)
}

fn clone_passthrough(model: &VoiceModel) -> VoiceModel {
    let _ = model;
    VoiceModel::passthrough(48_000)
}
