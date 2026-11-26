//! Sample-rate conversion through rubato's FFT resampler.
//!
//! Live inference resamples capture audio to the 16 kHz rate the content
//! encoder expects, then resamples the synthesizer waveform back to the
//! playback device.

use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Resampler};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ResampleError {
    #[error("sample rate must be non-zero")]
    ZeroRate,
    #[error("resampler construction failed: {0}")]
    Construct(String),
    #[error("resample failed: {0}")]
    Process(String),
}

/// Reused FFT resampler for the model path (device rate to 16 kHz and back).
///
/// Building a plan on every block would spend the live GPU budget on the CPU.
pub struct BlockResampler {
    from_sr: u32,
    to_sr: u32,
    inner: Option<Fft<f32>>,
}

impl BlockResampler {
    pub fn new(from_sr: u32, to_sr: u32) -> Result<Self, ResampleError> {
        if from_sr == 0 || to_sr == 0 {
            return Err(ResampleError::ZeroRate);
        }
        let inner = if from_sr == to_sr {
            None
        } else {
            Some(
                Fft::<f32>::new(from_sr as usize, to_sr as usize, 1024, 1, FixedSync::Both)
                    .map_err(|err| ResampleError::Construct(err.to_string()))?,
            )
        };
        Ok(Self {
            from_sr,
            to_sr,
            inner,
        })
    }

    pub fn process(&mut self, input: &[f32]) -> Result<Vec<f32>, ResampleError> {
        if self.from_sr == self.to_sr || input.is_empty() {
            return Ok(input.to_vec());
        }
        let resampler = self
            .inner
            .as_mut()
            .ok_or_else(|| ResampleError::Process("resampler was not constructed".into()))?;
        let adapter = InterleavedSlice::new(input, 1, input.len())
            .map_err(|err| ResampleError::Process(err.to_string()))?;
        let output = resampler
            .process_all(&adapter, input.len(), None)
            .map_err(|err| ResampleError::Process(err.to_string()))?;
        Ok(output.take_data())
    }
}

/// Resample mono `f32` audio. Equal rates return a copy. An empty input
/// returns an empty output.
pub fn resample_mono(input: &[f32], from_sr: u32, to_sr: u32) -> Result<Vec<f32>, ResampleError> {
    if from_sr == 0 || to_sr == 0 {
        return Err(ResampleError::ZeroRate);
    }
    if from_sr == to_sr || input.is_empty() {
        return Ok(input.to_vec());
    }

    let mut resampler = Fft::<f32>::new(from_sr as usize, to_sr as usize, 1024, 1, FixedSync::Both)
        .map_err(|err| ResampleError::Construct(err.to_string()))?;
    let adapter = InterleavedSlice::new(input, 1, input.len())
        .map_err(|err| ResampleError::Process(err.to_string()))?;
    let output = resampler
        .process_all(&adapter, input.len(), None)
        .map_err(|err| ResampleError::Process(err.to_string()))?;
    Ok(output.take_data())
}
