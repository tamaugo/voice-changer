//! Crossfade, loudness match, consonant protection, and neighbour retrieval.

/// Length-preserving crossfade. The start of each new block is blended with
/// the end of the previous block. A constant signal is unchanged.
#[derive(Clone, Debug)]
pub struct Crossfade {
    overlap: usize,
    tail: Vec<f32>,
}

impl Crossfade {
    pub fn new(overlap: usize) -> Self {
        Self {
            overlap,
            tail: Vec::new(),
        }
    }

    pub fn apply(&mut self, block: &[f32]) -> Vec<f32> {
        let mut out = block.to_vec();
        let n = self.overlap.min(out.len()).min(self.tail.len());
        for (index, sample) in out.iter_mut().take(n).enumerate() {
            let fade_in = (index + 1) as f32 / n as f32;
            *sample = self.tail[index] * (1.0 - fade_in) + *sample * fade_in;
        }
        if self.overlap > 0 && out.len() >= self.overlap {
            self.tail = out[out.len() - self.overlap..].to_vec();
        }
        out
    }
}

/// Pull `wet` toward the loudness of `dry`.
///
/// `rate` is 0 to leave `wet` alone and 1 to match `dry` frame by frame.
/// Frames are short so a loud syllable does not turn the whole block up.
pub fn rms_mix(dry: &[f32], wet: &mut [f32], rate: f32, frame: usize) {
    let rate = rate.clamp(0.0, 1.0);
    if rate == 0.0 {
        return;
    }
    let frame = frame.max(1);
    let count = dry.len().min(wet.len());
    let mut start = 0;
    while start < count {
        let end = (start + frame).min(count);
        let dry_rms = rms(&dry[start..end]);
        let wet_rms = rms(&wet[start..end]).max(1.0e-6);
        let gain = (dry_rms / wet_rms) * rate + (1.0 - rate);
        for sample in &mut wet[start..end] {
            *sample *= gain;
        }
        start = end;
    }
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let energy = samples.iter().map(|sample| sample * sample).sum::<f32>() / samples.len() as f32;
    energy.sqrt()
}

/// Blend converted content features back toward the original on unvoiced frames.
///
/// `protect` is the RVC consonant-protection amount, from 0 to 0.5. Voiced
/// frames (positive hertz) keep the converted features. Unvoiced frames use
/// `protect` as the converted weight, so 0.33 keeps a third of the conversion
/// and two thirds of the original breath or consonant.
pub fn protect_features(
    converted: &mut [f32],
    original: &[f32],
    pitch_hz: &[f32],
    dim: usize,
    protect: f32,
) {
    if dim == 0 {
        return;
    }
    let protect = protect.clamp(0.0, 0.5);
    let frames = converted.len().min(original.len()) / dim;
    for frame in 0..frames.min(pitch_hz.len()) {
        let weight = if pitch_hz[frame] > 0.0 { 1.0 } else { protect };
        let offset = frame * dim;
        for bin in 0..dim {
            let index = offset + bin;
            converted[index] = converted[index] * weight + original[index] * (1.0 - weight);
        }
    }
}

/// Rows of content features saved from training (`features.npy`).
///
/// Search is brute-force L2. A ten-minute voice is a few thousand rows, which
/// is small enough to scan on the audio thread. It is the same blend RVC
/// applies with faiss (inverse-distance weights, then `index_rate`), without
/// linking faiss into the live process.
#[derive(Clone, Debug)]
pub struct FeatureBank {
    dim: usize,
    rows: Vec<f32>,
}

impl FeatureBank {
    pub fn from_rows(dim: usize, rows: Vec<f32>) -> Option<Self> {
        if dim == 0 || rows.len() % dim != 0 || rows.is_empty() {
            return None;
        }
        Some(Self { dim, rows })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn row_count(&self) -> usize {
        self.rows.len() / self.dim
    }

    pub fn row(&self, index: usize) -> &[f32] {
        let start = index * self.dim;
        &self.rows[start..start + self.dim]
    }
}

/// Replace each frame of `features` with a mix of itself and its nearest rows.
///
/// `features` is row-major `[frames, dim]`. `k` neighbours, usually 8.
pub fn retrieve(features: &mut [f32], bank: &FeatureBank, index_rate: f32, k: usize) {
    let rate = index_rate.clamp(0.0, 1.0);
    if rate == 0.0 || k == 0 || bank.dim() == 0 {
        return;
    }
    let dim = bank.dim();
    if features.len() % dim != 0 {
        return;
    }
    let frames = features.len() / dim;
    let neighbours = k.min(bank.row_count()).max(1);
    // Each frame is independent, so large banks are split across cores.
    // Small banks stay on the calling thread to skip the spawn cost.
    let workers = if bank.row_count() * frames < 200_000 {
        1
    } else {
        std::thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(1)
            .min(frames)
            .max(1)
    };
    if workers == 1 {
        retrieve_frames(features, bank, rate, neighbours);
        return;
    }
    let frames_per_worker = frames.div_ceil(workers);
    std::thread::scope(|scope| {
        for chunk in features.chunks_mut(frames_per_worker * dim) {
            scope.spawn(move || retrieve_frames(chunk, bank, rate, neighbours));
        }
    });
}

fn retrieve_frames(features: &mut [f32], bank: &FeatureBank, rate: f32, neighbours: usize) {
    let dim = bank.dim();
    let frames = features.len() / dim;
    let mut mixed = vec![0.0f32; dim];
    let mut scored: Vec<(f32, usize)> = Vec::with_capacity(bank.row_count());
    for frame in 0..frames {
        let offset = frame * dim;
        let query = &features[offset..offset + dim];
        scored.clear();
        for row in 0..bank.row_count() {
            scored.push((squared_distance(query, bank.row(row)), row));
        }
        // Only the k nearest are used, so select them instead of sorting the
        // whole bank (tens of thousands of rows for a long dataset).
        let by_distance = |left: &(f32, usize), right: &(f32, usize)| {
            left.0
                .partial_cmp(&right.0)
                .unwrap_or(std::cmp::Ordering::Equal)
        };
        if neighbours < scored.len() {
            scored.select_nth_unstable_by(neighbours - 1, by_distance);
        }
        mixed.fill(0.0);
        let mut weight_sum = 0.0f32;
        for (distance, row) in scored.iter().take(neighbours) {
            let weight = 1.0 / (distance + 1.0e-8);
            weight_sum += weight;
            let neighbour = bank.row(*row);
            for (bin, value) in mixed.iter_mut().enumerate() {
                *value += neighbour[bin] * weight;
            }
        }
        if weight_sum == 0.0 {
            continue;
        }
        for (bin, value) in mixed.iter().enumerate() {
            let retrieved = value / weight_sum;
            features[offset + bin] = retrieved * rate + features[offset + bin] * (1.0 - rate);
        }
    }
}

/// Eight independent accumulators so the compiler can vectorise the loop. A
/// single running `sum()` forces strictly ordered float adds, which keeps it
/// scalar, and retrieval runs this once per bank row per frame.
fn squared_distance(left: &[f32], right: &[f32]) -> f32 {
    const LANES: usize = 8;
    let mut acc = [0.0f32; LANES];
    let chunks = left.len() / LANES * LANES;
    for (a, b) in left[..chunks]
        .chunks_exact(LANES)
        .zip(right[..chunks].chunks_exact(LANES))
    {
        for lane in 0..LANES {
            let delta = a[lane] - b[lane];
            acc[lane] += delta * delta;
        }
    }
    let mut total: f32 = acc.iter().sum();
    for (a, b) in left[chunks..].iter().zip(&right[chunks..]) {
        let delta = a - b;
        total += delta * delta;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crossfade_keeps_a_constant_signal() {
        let mut crossfade = Crossfade::new(4);
        let first = crossfade.apply(&[0.25; 8]);
        let second = crossfade.apply(&[0.25; 8]);
        assert_eq!(first, vec![0.25; 8]);
        assert_eq!(second, vec![0.25; 8]);
    }

    #[test]
    fn crossfade_blends_the_join() {
        let mut crossfade = Crossfade::new(2);
        let _ = crossfade.apply(&[1.0, 1.0, 1.0, 1.0]);
        let next = crossfade.apply(&[0.0, 0.0, 0.0, 0.0]);
        assert!(next[0] > 0.0 && next[0] < 1.0);
        assert_eq!(next[2], 0.0);
    }

    #[test]
    fn rms_mix_matches_the_dry_loudness() {
        let dry = [1.0f32; 8];
        let mut wet = [0.5f32; 8];
        rms_mix(&dry, &mut wet, 1.0, 8);
        let level = rms(&wet);
        assert!((level - 1.0).abs() < 1.0e-5, "{level}");
    }

    #[test]
    fn protect_restores_unvoiced_frames() {
        let mut converted = vec![1.0f32, 1.0, 0.0, 0.0];
        let original = vec![0.0f32, 0.0, 0.5, 0.5];
        protect_features(&mut converted, &original, &[100.0, 0.0], 2, 0.0);
        assert_eq!(&converted[..2], &[1.0, 1.0]);
        assert_eq!(&converted[2..], &[0.5, 0.5]);
    }

    #[test]
    fn threaded_retrieval_matches_single_thread() {
        let dim = 16;
        let mut seed = 7u32;
        let mut next = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1u32 << 24) as f32
        };
        let rows: Vec<f32> = (0..20_000 * dim).map(|_| next()).collect();
        let bank = FeatureBank::from_rows(dim, rows).unwrap();
        let input: Vec<f32> = (0..40 * dim).map(|_| next()).collect();
        let mut threaded = input.clone();
        retrieve(&mut threaded, &bank, 0.75, 8);
        let mut single = input;
        retrieve_frames(&mut single, &bank, 0.75, 8);
        assert_eq!(threaded, single);
    }

    #[test]
    fn retrieval_copies_the_matching_row() {
        let bank = FeatureBank::from_rows(2, vec![0.0, 0.0, 10.0, 10.0]).unwrap();
        let mut features = vec![10.0f32, 10.0];
        retrieve(&mut features, &bank, 1.0, 1);
        assert!((features[0] - 10.0).abs() < 1.0e-4);
        assert!((features[1] - 10.0).abs() < 1.0e-4);
    }
}
