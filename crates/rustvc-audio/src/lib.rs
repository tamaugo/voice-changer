//! Real-time audio path: device enumeration, a lock-free capture/playback
//! loop, and sample-rate conversion.
//!
//! The processing hook is [`SampleProcessor`]. The live engine never runs a
//! model itself. It pulls a block off the capture ring, asks the processor
//! for samples, and pushes the result to the playback ring.

mod drift;
mod engine;
mod resample;

pub use drift::DriftResampler;
pub use engine::{
    list_devices, run_offline, self_test, virtual_output_hint, AudioEngine, DeviceList,
    EngineConfig, EngineStatus, Passthrough, SampleProcessor,
};
pub use resample::{resample_mono, BlockResampler};

use std::time::Duration;

/// How long one processing block lasts, in milliseconds.
pub fn block_millis(block_frames: usize, sample_rate: u32) -> f32 {
    if sample_rate == 0 {
        return 0.0;
    }
    block_frames as f32 * 1000.0 / sample_rate as f32
}

pub fn block_duration(block_frames: usize, sample_rate: u32) -> Duration {
    Duration::from_secs_f32(block_millis(block_frames, sample_rate) / 1000.0)
}
