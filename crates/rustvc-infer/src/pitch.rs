//! RVC's published F0 input encoding.
//!
//! The synthesizer consumes a continuous pitch in hertz and a coarse bin in
//! `1..=255`. The bin edges used by RVC are 50 Hz and 1100 Hz on the mel
//! scale `1127 * ln(1 + hz / 700)`. This is the model input contract from the
//! MIT-licensed WebUI, not a copy of its pitch tracker.

const F0_MIN_HZ: f32 = 50.0;
const F0_MAX_HZ: f32 = 1100.0;

fn mel(hz: f32) -> f32 {
    1127.0 * (1.0 + hz / 700.0).ln()
}

/// Shift a pitch by `semitones`. Twelve semitones doubles the frequency.
pub fn shift_hz(hz: f32, semitones: f32) -> f32 {
    if hz <= 0.0 {
        return 0.0;
    }
    hz * 2.0f32.powf(semitones / 12.0)
}

/// Quantize hertz to the RVC coarse bin. Unvoiced frames (non-positive hz)
/// become bin 1, which is what the synthesizer treats as the floor.
pub fn coarse_f0(hz: f32) -> i64 {
    if hz <= 0.0 {
        return 1;
    }
    let mel_min = mel(F0_MIN_HZ);
    let mel_max = mel(F0_MAX_HZ);
    let bin = (mel(hz) - mel_min) * 254.0 / (mel_max - mel_min) + 1.0;
    bin.round().clamp(1.0, 255.0) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoints_land_on_the_rvc_bins() {
        assert_eq!(coarse_f0(0.0), 1);
        assert_eq!(coarse_f0(50.0), 1);
        assert_eq!(coarse_f0(1100.0), 255);
        assert_eq!(coarse_f0(10_000.0), 255);
    }

    #[test]
    fn octave_doubles_frequency() {
        let shifted = shift_hz(440.0, 12.0);
        assert!((shifted - 880.0).abs() < 0.01);
    }
}
