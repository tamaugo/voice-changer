//! Time `process` and compare it with the live GPU budget.

use std::time::Instant;

use crate::budget::duty_cycle_percent;

#[derive(Debug, Clone)]
pub struct DutyReport {
    pub blocks: usize,
    pub block_ms: f32,
    pub mean_process_ms: f32,
    pub max_process_ms: f32,
    pub duty_percent: f32,
    pub budget_percent: u8,
    pub inside_budget: bool,
}

impl DutyReport {
    pub fn summary(&self) -> String {
        let verdict = if self.inside_budget {
            "inside the live budget"
        } else {
            "over the live budget"
        };
        format!(
            "{:.2}% duty over {} blocks (mean {:.2} ms, max {:.2} ms, block {:.1} ms, budget {}%) — {verdict}",
            self.duty_percent,
            self.blocks,
            self.mean_process_ms,
            self.max_process_ms,
            self.block_ms,
            self.budget_percent
        )
    }
}

/// Time `process` over `blocks` of a sine and compare it with the budget.
pub fn measure_duty<F>(
    mut process: F,
    sample_rate: u32,
    block_frames: usize,
    blocks: usize,
    budget_percent: u8,
) -> DutyReport
where
    F: FnMut(&[f32]) -> Vec<f32>,
{
    let block_frames = block_frames.max(1);
    let blocks = blocks.max(1);
    let sample_rate = sample_rate.max(1);
    let mut block = vec![0.0f32; block_frames];
    for (index, sample) in block.iter_mut().enumerate() {
        *sample = (index as f32 * 0.02).sin() * 0.2;
    }
    // One warm-up block so session setup is not the whole average.
    let _ = process(&block);
    let mut total = 0.0f32;
    let mut max_ms = 0.0f32;
    for _ in 0..blocks {
        let started = Instant::now();
        let _ = process(&block);
        let elapsed = started.elapsed().as_secs_f32() * 1000.0;
        total += elapsed;
        max_ms = max_ms.max(elapsed);
    }
    let mean = total / blocks as f32;
    let block_ms = block_frames as f32 * 1000.0 / sample_rate as f32;
    let duty = duty_cycle_percent(mean, block_ms);
    let budget = budget_percent.clamp(1, 100);
    DutyReport {
        blocks,
        block_ms,
        mean_process_ms: mean,
        max_process_ms: max_ms,
        duty_percent: duty,
        budget_percent: budget,
        inside_budget: duty <= f32::from(budget),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passthrough_stays_inside_a_five_percent_budget() {
        let report = measure_duty(|input| input.to_vec(), 48_000, 4_800, 8, 5);
        assert!(report.inside_budget, "{}", report.summary());
        assert!(report.duty_percent < 5.0, "{}", report.duty_percent);
    }
}
