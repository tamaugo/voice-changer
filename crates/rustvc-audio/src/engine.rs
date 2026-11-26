//! Capture, process, and playback. On Windows, cpal uses WASAPI.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{
    BufferSize, Device, SampleFormat, Stream, StreamConfig, SupportedBufferSize,
    SupportedStreamConfig,
};
use ringbuf::traits::{Consumer, Observer, Producer, Split};
use ringbuf::HeapRb;

use crate::drift::DriftResampler;
use crate::resample::resample_mono;

/// Installed into the live loop. Return one output sample per input sample
/// when you can. The engine queues extras and pads a shortfall with silence
/// so playback stays aligned with capture.
pub trait SampleProcessor: Send {
    fn process(&mut self, input: &[f32]) -> Vec<f32>;
    fn label(&self) -> &str;
}

/// Copies capture to playback. This is the live path until a voice is loaded.
pub struct Passthrough;

impl SampleProcessor for Passthrough {
    fn process(&mut self, input: &[f32]) -> Vec<f32> {
        input.to_vec()
    }

    fn label(&self) -> &str {
        "Passthrough"
    }
}

#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Empty selects the host default.
    pub input_device: String,
    /// Empty selects the host default. For a virtual microphone, select the
    /// cable's playback endpoint (VB-CABLE names it "CABLE Input").
    pub output_device: String,
    /// Preferred stream rate. A device that cannot open it keeps its default,
    /// and the loop resamples between the two rates.
    pub sample_rate: u32,
    /// Samples per processing block, at the capture rate.
    pub block_frames: usize,
    /// WASAPI/shared-mode callback quantum, in frames. Used when the device
    /// advertises a buffer-size range that contains it. 480 frames is 10 ms
    /// at 48 kHz.
    pub callback_frames: u32,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            input_device: String::new(),
            output_device: String::new(),
            sample_rate: 48_000,
            block_frames: 4_800,
            callback_frames: 480,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EngineStatus {
    pub running: bool,
    pub input_device: String,
    pub output_device: String,
    pub input_sample_rate: u32,
    pub output_sample_rate: u32,
    pub label: String,
    pub input_overflows: u64,
    pub output_underruns: u64,
    /// Wall time of the last `process` call divided by the block period.
    /// This is a duty cycle, not an nvidia-smi utilization percentage.
    pub process_duty_percent: f32,
    pub last_process_ms: f32,
    pub error: Option<String>,
    pub virtual_output: Option<String>,
}

impl Default for EngineStatus {
    fn default() -> Self {
        Self {
            running: false,
            input_device: String::new(),
            output_device: String::new(),
            input_sample_rate: 0,
            output_sample_rate: 0,
            label: "Passthrough".to_string(),
            input_overflows: 0,
            output_underruns: 0,
            process_duty_percent: 0.0,
            last_process_ms: 0.0,
            error: None,
            virtual_output: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DeviceList {
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub default_input: Option<String>,
    pub default_output: Option<String>,
}

pub fn list_devices() -> DeviceList {
    let host = cpal::default_host();
    let inputs = host
        .input_devices()
        .map(|iter| iter.map(|device| device.to_string()).collect())
        .unwrap_or_default();
    let outputs = host
        .output_devices()
        .map(|iter| iter.map(|device| device.to_string()).collect())
        .unwrap_or_default();
    DeviceList {
        inputs,
        outputs,
        default_input: host.default_input_device().map(|device| device.to_string()),
        default_output: host
            .default_output_device()
            .map(|device| device.to_string()),
    }
}

/// Hint shown when playback is a known virtual endpoint.
pub fn virtual_output_hint(name: &str) -> Option<&'static str> {
    let lower = name.to_ascii_lowercase();
    if lower.contains("cable input") || lower.contains("vb-audio") {
        Some("VB-CABLE playback endpoint. In Discord, choose \"CABLE Output\" as the microphone.")
    } else if lower.contains("voicemeeter") {
        Some("VoiceMeeter endpoint. Route this output to a virtual bus, then select that bus in Discord.")
    } else if lower.contains("synchronous audio router") {
        Some("Synchronous Audio Router endpoint. Select the paired input in Discord.")
    } else {
        None
    }
}

enum Command {
    Swap(Box<dyn SampleProcessor>),
}

/// Owns the audio thread. Dropping the handle stops capture and playback.
pub struct AudioEngine {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
    status: Arc<Mutex<EngineStatus>>,
    commands: Sender<Command>,
}

impl AudioEngine {
    pub fn start(config: EngineConfig, processor: Box<dyn SampleProcessor>) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let status = Arc::new(Mutex::new(EngineStatus {
            label: processor.label().to_string(),
            ..EngineStatus::default()
        }));
        let (tx, rx) = mpsc::channel();
        let stop_flag = Arc::clone(&stop);
        let status_flag = Arc::clone(&status);
        let thread = thread::spawn(move || session(stop_flag, status_flag, rx, processor, config));
        Self {
            stop,
            thread: Some(thread),
            status,
            commands: tx,
        }
    }

    pub fn status(&self) -> EngineStatus {
        self.status.lock().expect("audio status").clone()
    }

    pub fn swap_processor(&self, processor: Box<dyn SampleProcessor>) {
        let _ = self.commands.send(Command::Swap(processor));
    }
}

impl Drop for AudioEngine {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn session(
    stop: Arc<AtomicBool>,
    status: Arc<Mutex<EngineStatus>>,
    commands: Receiver<Command>,
    mut processor: Box<dyn SampleProcessor>,
    config: EngineConfig,
) {
    let block = config.block_frames.max(1);
    match open_streams(&config, &status) {
        Ok(opened) => run_loop(stop, status, commands, &mut processor, opened, block),
        Err(err) => {
            let mut state = status.lock().expect("audio status");
            state.running = false;
            state.error = Some(err);
        }
    }
}

struct Opened {
    _input: Stream,
    _output: Stream,
    capture: ringbuf::HeapCons<f32>,
    playback: ringbuf::HeapProd<f32>,
    overflows: Arc<AtomicU64>,
    underruns: Arc<AtomicU64>,
    input_rate: u32,
    output_rate: u32,
    playback_capacity: usize,
}

fn open_streams(
    config: &EngineConfig,
    status: &Arc<Mutex<EngineStatus>>,
) -> Result<Opened, String> {
    let host = cpal::default_host();
    let input_device = pick_device(
        host.input_devices().map_err(|err| err.to_string())?,
        &config.input_device,
    )
    .or_else(|| host.default_input_device())
    .ok_or_else(|| "no input device is available".to_string())?;
    let output_device = pick_device(
        host.output_devices().map_err(|err| err.to_string())?,
        &config.output_device,
    )
    .or_else(|| host.default_output_device())
    .ok_or_else(|| "no output device is available".to_string())?;

    let input_name = input_device.to_string();
    let output_name = output_device.to_string();
    let input_cfg = input_device
        .default_input_config()
        .map_err(|err| format!("input config: {err}"))?;
    let output_cfg = choose_output_config(&output_device, config.sample_rate)?;

    let input_rate = input_cfg.sample_rate();
    let output_rate = output_cfg.sample_rate();
    let input_channels = input_cfg.channels() as usize;
    let output_channels = output_cfg.channels() as usize;

    let capacity = (input_rate as usize * 4)
        .max(config.block_frames * 8)
        .max(1024);
    let playback_capacity = capacity * 2;
    let capture_rb = HeapRb::<f32>::new(capacity);
    let playback_rb = HeapRb::<f32>::new(playback_capacity);
    let (capture_prod, capture_cons) = capture_rb.split();
    let (playback_prod, playback_cons) = playback_rb.split();

    let overflows = Arc::new(AtomicU64::new(0));
    let underruns = Arc::new(AtomicU64::new(0));
    let input_stream = build_input(
        &input_device,
        &input_cfg,
        input_channels,
        capture_prod,
        Arc::clone(&overflows),
        config.callback_frames,
    )?;
    let output_stream = build_output(
        &output_device,
        &output_cfg,
        output_channels,
        playback_cons,
        Arc::clone(&underruns),
        config.callback_frames,
    )?;
    input_stream
        .play()
        .map_err(|err| format!("start input: {err}"))?;
    output_stream
        .play()
        .map_err(|err| format!("start output: {err}"))?;

    {
        let mut state = status.lock().expect("audio status");
        state.running = true;
        state.input_device = input_name;
        state.output_device = output_name.clone();
        state.input_sample_rate = input_rate;
        state.output_sample_rate = output_rate;
        state.error = None;
        state.virtual_output = virtual_output_hint(&output_name).map(str::to_string);
    }

    Ok(Opened {
        _input: input_stream,
        _output: output_stream,
        capture: capture_cons,
        playback: playback_prod,
        overflows,
        underruns,
        input_rate,
        output_rate,
        playback_capacity,
    })
}

fn run_loop(
    stop: Arc<AtomicBool>,
    status: Arc<Mutex<EngineStatus>>,
    commands: Receiver<Command>,
    processor: &mut Box<dyn SampleProcessor>,
    mut opened: Opened,
    block: usize,
) {
    let mut scratch = vec![0.0f32; block];
    let mut pending: Vec<f32> = Vec::new();
    let mut drift = DriftResampler::new(opened.input_rate, opened.output_rate, block).ok();
    while !stop.load(Ordering::Relaxed) {
        while let Ok(command) = commands.try_recv() {
            match command {
                Command::Swap(next) => {
                    let label = next.label().to_string();
                    *processor = next;
                    pending.clear();
                    status.lock().expect("audio status").label = label;
                }
            }
        }

        if opened.capture.occupied_len() < block {
            thread::sleep(Duration::from_millis(2));
            continue;
        }

        let read = opened.capture.pop_slice(&mut scratch[..block]);
        let started = Instant::now();
        let produced = processor.process(&scratch[..read]);
        let elapsed = started.elapsed();
        let aligned = take_aligned(&mut pending, &produced, read);
        let playback = match drift.as_mut() {
            Some(resampler) => {
                let fill = opened.playback.occupied_len();
                match resampler.process(&aligned, fill, opened.playback_capacity) {
                    Ok(samples) => samples,
                    Err(err) => {
                        status.lock().expect("audio status").error = Some(err.to_string());
                        aligned
                    }
                }
            }
            None => aligned,
        };
        let _ = opened.playback.push_slice(&playback);

        let block_ms = read as f32 * 1000.0 / opened.input_rate.max(1) as f32;
        let process_ms = elapsed.as_secs_f32() * 1000.0;
        let duty = if block_ms <= 0.0 {
            0.0
        } else {
            process_ms / block_ms * 100.0
        };
        let mut state = status.lock().expect("audio status");
        state.label = processor.label().to_string();
        state.last_process_ms = process_ms;
        state.process_duty_percent = duty;
        state.input_overflows = opened.overflows.load(Ordering::Relaxed);
        state.output_underruns = opened.underruns.load(Ordering::Relaxed);
        state.running = true;
    }
}

/// Queue `produced` and return exactly `frames` samples at the capture rate.
fn take_aligned(pending: &mut Vec<f32>, produced: &[f32], frames: usize) -> Vec<f32> {
    pending.extend_from_slice(produced);
    let mut out = vec![0.0f32; frames];
    let take = frames.min(pending.len());
    out[..take].copy_from_slice(&pending[..take]);
    pending.drain(..take);
    out
}

fn pick_device(devices: impl Iterator<Item = Device>, wanted: &str) -> Option<Device> {
    if wanted.is_empty() {
        return None;
    }
    let wanted_lower = wanted.to_ascii_lowercase();
    devices.into_iter().find(|device| {
        let name = device.to_string();
        name == wanted || name.to_ascii_lowercase().contains(&wanted_lower)
    })
}

fn choose_output_config(device: &Device, preferred: u32) -> Result<SupportedStreamConfig, String> {
    let default = device
        .default_output_config()
        .map_err(|err| format!("output config: {err}"))?;
    if preferred == 0 || default.sample_rate() == preferred {
        return Ok(default);
    }
    if let Ok(ranges) = device.supported_output_configs() {
        for range in ranges {
            if range.min_sample_rate() <= preferred && preferred <= range.max_sample_rate() {
                return Ok(range.with_sample_rate(preferred));
            }
        }
    }
    Ok(default)
}

fn build_input(
    device: &Device,
    config: &SupportedStreamConfig,
    channels: usize,
    prod: ringbuf::HeapProd<f32>,
    overflows: Arc<AtomicU64>,
    callback_frames: u32,
) -> Result<Stream, String> {
    let stream_config = stream_config_with_buffer(config, callback_frames);
    match config.sample_format() {
        SampleFormat::F32 => {
            open_input::<f32, _>(device, stream_config, channels, prod, overflows, |sample| {
                sample
            })
        }
        SampleFormat::I16 => {
            open_input::<i16, _>(device, stream_config, channels, prod, overflows, |sample| {
                sample as f32 / i16::MAX as f32
            })
        }
        SampleFormat::U16 => {
            open_input::<u16, _>(device, stream_config, channels, prod, overflows, |sample| {
                (sample as f32 - 32768.0) / 32768.0
            })
        }
        other => Err(format!("unsupported input sample format {other:?}")),
    }
}

fn open_input<T, F>(
    device: &Device,
    config: StreamConfig,
    channels: usize,
    mut prod: ringbuf::HeapProd<f32>,
    overflows: Arc<AtomicU64>,
    convert: F,
) -> Result<Stream, String>
where
    T: cpal::SizedSample,
    F: Fn(T) -> f32 + Send + 'static,
{
    let channels = channels.max(1);
    let mut mono = Vec::<f32>::with_capacity(4096);
    device
        .build_input_stream(
            config,
            move |data: &[T], _| {
                mono.clear();
                for frame in data.chunks(channels) {
                    let sum: f32 = frame.iter().copied().map(&convert).sum();
                    mono.push(sum / channels as f32);
                }
                let pushed = prod.push_slice(&mono);
                if pushed < mono.len() {
                    overflows.fetch_add(1, Ordering::Relaxed);
                }
            },
            |err| eprintln!("input stream error: {err}"),
            None,
        )
        .map_err(|err| format!("open input: {err}"))
}

fn build_output(
    device: &Device,
    config: &SupportedStreamConfig,
    channels: usize,
    cons: ringbuf::HeapCons<f32>,
    underruns: Arc<AtomicU64>,
    callback_frames: u32,
) -> Result<Stream, String> {
    let stream_config = stream_config_with_buffer(config, callback_frames);
    match config.sample_format() {
        SampleFormat::F32 => {
            open_output::<f32, _>(device, stream_config, channels, cons, underruns, |sample| {
                sample
            })
        }
        SampleFormat::I16 => {
            open_output::<i16, _>(device, stream_config, channels, cons, underruns, |sample| {
                (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16
            })
        }
        SampleFormat::U16 => {
            open_output::<u16, _>(device, stream_config, channels, cons, underruns, |sample| {
                ((sample.clamp(-1.0, 1.0) * 32768.0) + 32768.0) as u16
            })
        }
        other => Err(format!("unsupported output sample format {other:?}")),
    }
}

fn open_output<T, F>(
    device: &Device,
    config: StreamConfig,
    channels: usize,
    mut cons: ringbuf::HeapCons<f32>,
    underruns: Arc<AtomicU64>,
    convert: F,
) -> Result<Stream, String>
where
    T: cpal::SizedSample + Copy,
    F: Fn(f32) -> T + Send + 'static,
{
    let channels = channels.max(1);
    device
        .build_output_stream(
            config,
            move |data: &mut [T], _| {
                let mut missed = false;
                for frame in data.chunks_mut(channels) {
                    let sample = cons.try_pop().unwrap_or_else(|| {
                        missed = true;
                        0.0
                    });
                    let converted = convert(sample);
                    for slot in frame.iter_mut() {
                        *slot = converted;
                    }
                }
                if missed {
                    underruns.fetch_add(1, Ordering::Relaxed);
                }
            },
            |err| eprintln!("output stream error: {err}"),
            None,
        )
        .map_err(|err| format!("open output: {err}"))
}

/// Request a fixed shared-mode quantum when the device publishes a range.
/// WASAPI uses this as the callback buffer. A device that does not publish a
/// range keeps `BufferSize::Default`, which is the mix-format period.
fn stream_config_with_buffer(config: &SupportedStreamConfig, requested: u32) -> StreamConfig {
    let mut stream = config.config();
    stream.buffer_size = match config.buffer_size() {
        SupportedBufferSize::Range { min, max } => {
            BufferSize::Fixed(requested.max(32).clamp(*min, *max))
        }
        SupportedBufferSize::Unknown => BufferSize::Default,
    };
    stream
}

/// Run `input` through `processor` with the same block alignment as the live
/// loop. No sound device is opened.
pub fn run_offline(
    processor: &mut dyn SampleProcessor,
    input: &[f32],
    block_frames: usize,
) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }
    let block = block_frames.max(1);
    let capture = HeapRb::<f32>::new(input.len() + 8);
    let playback = HeapRb::<f32>::new(input.len() * 4 + 8);
    let (mut prod, mut cons) = capture.split();
    let (mut out_prod, mut out_cons) = playback.split();
    let _ = prod.push_slice(input);
    let mut pending = Vec::new();
    let mut scratch = vec![0.0f32; block];
    while cons.occupied_len() > 0 {
        let want = block.min(cons.occupied_len());
        let read = cons.pop_slice(&mut scratch[..want]);
        let produced = processor.process(&scratch[..read]);
        let aligned = take_aligned(&mut pending, &produced, read);
        let _ = out_prod.push_slice(&aligned);
    }
    let mut out = vec![0.0f32; out_cons.occupied_len()];
    let _ = out_cons.pop_slice(&mut out);
    out
}

/// Offline check used by `voice-changer --self-test` and the unit tests.
pub fn self_test() -> Result<(), String> {
    let mut processor = Passthrough;
    let input: Vec<f32> = (0..4096).map(|i| (i as f32 * 0.02).sin()).collect();
    let output = run_offline(&mut processor, &input, 480);
    if output != input {
        return Err("passthrough loop did not reproduce the input".into());
    }
    let resampled = resample_mono(&input, 48_000, 16_000).map_err(|err| err.to_string())?;
    let expected = (input.len() as f32 * 16_000.0 / 48_000.0).round();
    if (resampled.len() as f32 - expected).abs() > 8.0 {
        return Err(format!(
            "resample length {} was far from {expected}",
            resampled.len()
        ));
    }
    if resampled.iter().all(|sample| *sample == 0.0) {
        return Err("resampler produced silence".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Gain(f32);

    impl SampleProcessor for Gain {
        fn process(&mut self, input: &[f32]) -> Vec<f32> {
            input.iter().map(|sample| sample * self.0).collect()
        }

        fn label(&self) -> &str {
            "gain"
        }
    }

    #[test]
    fn passthrough_is_bit_exact() {
        let mut processor = Passthrough;
        let input: Vec<f32> = (0..5000).map(|i| (i as f32 * 0.01).sin()).collect();
        let output = run_offline(&mut processor, &input, 480);
        assert_eq!(output, input);
    }

    #[test]
    fn processor_hook_changes_samples() {
        let mut processor = Gain(0.5);
        let output = run_offline(&mut processor, &[0.5, -0.25, 1.0], 2);
        assert_eq!(output, vec![0.25, -0.125, 0.5]);
    }

    #[test]
    fn virtual_cable_hint_matches_vb_cable() {
        assert!(virtual_output_hint("CABLE Input (VB-Audio Virtual Cable)").is_some());
        assert!(virtual_output_hint("Microphone (Realtek)").is_none());
    }

    #[test]
    fn offline_self_test_passes() {
        self_test().unwrap();
    }
}
