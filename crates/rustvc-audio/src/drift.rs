//! Correct capture/playback clock drift with rubato. Equal rates use `Slip`. Different rates use `Async`.

use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Adjustable, Async, FixedAsync, PolynomialDegree, Resampler, Slip};

use crate::resample::ResampleError;

const HEALTHY_BLOCKS: f64 = 0.35;

enum Engine {
    Slip(Slip<f32>),
    Async(Async<f32>),
}

pub struct DriftResampler {
    engine: Engine,
    nominal: f64,
    chunk: usize,
}

impl DriftResampler {
    pub fn new(from_sr: u32, to_sr: u32, chunk: usize) -> Result<Self, ResampleError> {
        if from_sr == 0 || to_sr == 0 || chunk < 4 {
            return Err(ResampleError::Construct(
                "drift resampler needs a positive rate and a chunk of at least 4 frames".into(),
            ));
        }
        let nominal = f64::from(to_sr) / f64::from(from_sr);
        let engine = if from_sr == to_sr {
            Engine::Slip(
                Slip::<f32>::new(chunk, 1, FixedAsync::Input)
                    .map_err(|err| ResampleError::Construct(err.to_string()))?,
            )
        } else {
            Engine::Async(
                Async::<f32>::new_poly(
                    nominal,
                    1.05,
                    PolynomialDegree::Cubic,
                    chunk,
                    1,
                    FixedAsync::Input,
                )
                .map_err(|err| ResampleError::Construct(err.to_string()))?,
            )
        };
        Ok(Self {
            engine,
            nominal,
            chunk,
        })
    }

    /// Resample one capture block.
    ///
    /// `playback_fill` and `playback_capacity` are the output ring's occupancy.
    /// A starving ring raises the ratio a little. A full ring lowers it.
    pub fn process(
        &mut self,
        input: &[f32],
        playback_fill: usize,
        playback_capacity: usize,
    ) -> Result<Vec<f32>, ResampleError> {
        if input.len() != self.chunk {
            return Err(ResampleError::Process(format!(
                "drift resampler expected {} frames, got {}",
                self.chunk,
                input.len()
            )));
        }
        let ratio = steered_ratio(self.nominal, playback_fill, playback_capacity, self.chunk);
        // Slip reports an error when the request is outside the tiny correction
        // range, after clamping. The next chunk still runs at the clamped ratio.
        let _ = self.set_ratio(ratio);
        self.render(input)
    }

    fn set_ratio(&mut self, ratio: f64) {
        match &mut self.engine {
            Engine::Slip(resampler) => {
                let _ = resampler.set_resample_ratio(ratio, false);
            }
            Engine::Async(resampler) => {
                let _ = resampler.set_resample_ratio(ratio, false);
            }
        }
    }

    fn render(&mut self, input: &[f32]) -> Result<Vec<f32>, ResampleError> {
        let max_out = match &self.engine {
            Engine::Slip(resampler) => resampler.output_frames_max(),
            Engine::Async(resampler) => resampler.output_frames_max(),
        };
        let mut output = vec![0.0f32; max_out.max(input.len()) + 8];
        let frames = output.len();
        let input_view = InterleavedSlice::new(input, 1, input.len())
            .map_err(|err| ResampleError::Process(err.to_string()))?;
        let mut output_view = InterleavedSlice::new_mut(&mut output, 1, frames)
            .map_err(|err| ResampleError::Process(err.to_string()))?;
        let written = match &mut self.engine {
            Engine::Slip(resampler) => {
                resampler
                    .process_into_buffer(&input_view, &mut output_view, None)
                    .map_err(|err| ResampleError::Process(err.to_string()))?
                    .1
            }
            Engine::Async(resampler) => {
                resampler
                    .process_into_buffer(&input_view, &mut output_view, None)
                    .map_err(|err| ResampleError::Process(err.to_string()))?
                    .1
            }
        };
        output.truncate(written);
        Ok(output)
    }
}

fn steered_ratio(nominal: f64, fill: usize, capacity: usize, chunk: usize) -> f64 {
    let capacity = capacity.max(chunk).max(1) as f64;
    let target = (capacity * 0.25).clamp(chunk as f64, capacity * 0.5);
    let error = (fill as f64 - target) / target;
    if error.abs() < HEALTHY_BLOCKS && (nominal - 1.0).abs() < 1.0e-6 {
        return 1.0;
    }
    let nudge = (1.0 - error * 0.01).clamp(0.99, 1.01);
    (nominal * nudge).max(1.0e-3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_rates_keep_the_block_length_when_the_ring_is_healthy() {
        let mut resampler = DriftResampler::new(48_000, 48_000, 480).unwrap();
        let input: Vec<f32> = (0..480).map(|index| (index as f32 * 0.01).sin()).collect();
        let output = resampler.process(&input, 4_000, 16_000).unwrap();
        assert_eq!(output.len(), input.len());
        let error: f32 = input
            .iter()
            .zip(&output)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0, f32::max);
        assert!(error < 1.0e-5, "{error}");
    }

    #[test]
    fn unequal_rates_change_the_length() {
        let mut resampler = DriftResampler::new(48_000, 16_000, 480).unwrap();
        let input = vec![0.2f32; 480];
        let output = resampler.process(&input, 100, 8_000).unwrap();
        assert!(output.len() > 100 && output.len() < 250, "{}", output.len());
        assert!(output.iter().any(|sample| sample.abs() > 1.0e-4));
    }
}
