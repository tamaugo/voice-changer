//! Live GPU budget.
//!
//! A 5% reading in nvidia-smi is a one-second average. Voice conversion does
//! not need the GPU for the whole second: one short inference burst per audio
//! block is enough. The duty cycle is
//!
//! ```text
//! duty% = infer_milliseconds / block_milliseconds * 100
//! ```
//!
//! On an RTX 5070 the burst itself rarely fills the chip, so the nvidia-smi
//! figure is usually lower than this duty cycle. The settings screen shows
//! the duty cycle because the process can measure it without NVML. The ONNX
//! session also asks the CUDA execution provider for a small arena and the
//! efficient workload type, which keeps the gaming process's VRAM and clocks
//! in better shape than a PyTorch runtime would.
//!
//! Training ignores this budget on purpose.

/// Target live-inference GPU duty cycle and the last measurement.
#[derive(Debug, Clone, Copy)]
pub struct GpuBudget {
    pub max_percent: u8,
}

impl GpuBudget {
    pub fn new(max_percent: u8) -> Self {
        Self {
            max_percent: max_percent.clamp(1, 100),
        }
    }

    pub fn duty_percent(infer_ms: f32, block_ms: f32) -> f32 {
        duty_cycle_percent(infer_ms, block_ms)
    }

    pub fn allows(&self, infer_ms: f32, block_ms: f32) -> bool {
        Self::duty_percent(infer_ms, block_ms) <= f32::from(self.max_percent)
    }

    /// Suggest a longer block when a measurement is over budget.
    ///
    /// Latency rises with the block. The cap is 400 ms, which is already a
    /// lot for conversation; past that the right fix is a smaller model or
    /// the CPU execution provider, not a longer buffer.
    pub fn suggest_block_ms(&self, infer_ms: f32, current_block_ms: f32) -> f32 {
        if infer_ms <= 0.0 {
            return current_block_ms;
        }
        let needed = infer_ms * 100.0 / f32::from(self.max_percent);
        needed.clamp(current_block_ms, 400.0)
    }
}

impl Default for GpuBudget {
    fn default() -> Self {
        Self { max_percent: 5 }
    }
}

pub fn duty_cycle_percent(infer_ms: f32, block_ms: f32) -> f32 {
    if block_ms <= 0.0 {
        return 0.0;
    }
    (infer_ms / block_ms) * 100.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn five_percent_budget_matches_the_live_target() {
        let budget = GpuBudget::default();
        assert!(budget.allows(5.0, 100.0));
        assert!(!budget.allows(6.0, 100.0));
        assert!((duty_cycle_percent(5.0, 100.0) - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn suggestion_grows_the_block_and_then_stops() {
        let budget = GpuBudget::new(5);
        let suggested = budget.suggest_block_ms(20.0, 100.0);
        assert!((suggested - 400.0).abs() < 0.1);
    }
}
