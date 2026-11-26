//! ONNX Runtime sessions. Compiled with `--features onnx`. Load the DLL from `ORT_DYLIB_PATH`.
//! CUDA 12.8+ is required for RTX 5070 (`sm_120`).
//!
//! `content.onnx`: 16 kHz waveform to features. `f0.onnx`: 16 kHz waveform to hertz.
//! `synthesizer.onnx`: `phone`, `phone_lengths`, `pitch`, `pitchf`, `ds` to audio.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use ndarray::{Array2, Array3, ArrayD};
use ort::session::Session;
use ort::value::TensorRef;

use crate::dsp::{protect_features, retrieve, FeatureBank};
use crate::model::InferParams;
use crate::pitch::{coarse_f0, shift_hz};
use crate::voices::VoiceRecord;
use crate::InferDevice;

pub struct OnnxSessions {
    _environment: ort::environment::Environment,
    content: Session,
    f0: Option<Session>,
    synthesizer: Session,
    content_path: PathBuf,
}

impl OnnxSessions {
    pub fn load(
        record: &VoiceRecord,
        params: &InferParams,
        device: InferDevice,
        onnx_dll: Option<&Path>,
    ) -> Result<Self, String> {
        let synthesizer_path = record.synthesizer().ok_or_else(|| {
            "no synthesizer checkpoint. Export the trained RVC model to synthesizer.onnx"
                .to_string()
        })?;
        if synthesizer_path.extension().and_then(|ext| ext.to_str()) != Some("onnx") {
            return Err(format!(
                "{} is not ONNX. The live hook loads synthesizer.onnx; .pth checkpoints stay on the training side",
                synthesizer_path.display()
            ));
        }
        let content_path = record
            .content_model()
            .ok_or_else(|| "content.onnx is missing (ContentVec or HuBERT export)".to_string())?;
        let f0_path = record.f0_model();
        if params.f0 && f0_path.is_none() {
            return Err("f0.onnx is missing (RMVPE export). Uncheck F0 only if the voice was trained without it".into());
        }

        let environment = shared_environment(onnx_dll)?;
        let content = open_session(&environment, &content_path, device)?;
        let f0 = match f0_path {
            Some(path) => Some(open_session(&environment, &path, device)?),
            None => None,
        };
        let synthesizer = open_session(&environment, &synthesizer_path, device)?;
        Ok(Self {
            _environment: environment,
            content,
            f0,
            synthesizer,
            content_path,
        })
    }

    pub fn run(
        &mut self,
        audio_16k: &[f32],
        params: &InferParams,
        bank: Option<&FeatureBank>,
    ) -> Result<Vec<f32>, String> {
        let mut features = run_content(&mut self.content, audio_16k, params.feature_dim())?;
        let original = features.clone();
        if let Some(bank) = bank {
            if let Some(flat) = features.as_slice_mut() {
                retrieve(flat, bank, params.index_rate, 8);
            }
        }
        let (coarse, continuous) = if params.f0 {
            let f0 = self.f0.as_mut().ok_or("f0 session missing")?;
            let hz = run_f0(f0, audio_16k)?;
            let continuous: Vec<f32> = hz
                .into_iter()
                .map(|hz| shift_hz(hz, params.pitch_semitones))
                .collect();
            if let (Some(converted), Some(source)) = (features.as_slice_mut(), original.as_slice())
            {
                protect_features(
                    converted,
                    source,
                    &continuous,
                    params.feature_dim(),
                    params.protect,
                );
            }
            let coarse: Vec<i64> = continuous.iter().copied().map(coarse_f0).collect();
            (coarse, continuous)
        } else {
            (Vec::new(), Vec::new())
        };
        run_synthesizer(
            &mut self.synthesizer,
            &features,
            &coarse,
            &continuous,
            params,
        )
    }
}

/// ort allows one environment per process and remembers the first one only
/// weakly: once it is dropped, every later `build()` fails with "only one
/// environment is allowed per process". Keep one alive for the whole process
/// and share it, so loading a second voice or restarting audio works.
///
/// A failed init is not cached, so fixing the DLL path in Settings can be
/// retried. A successful one is: switching to a different onnxruntime.dll
/// needs an app restart.
fn shared_environment(onnx_dll: Option<&Path>) -> Result<ort::environment::Environment, String> {
    static SHARED: OnceLock<ort::environment::Environment> = OnceLock::new();
    static INIT: Mutex<()> = Mutex::new(());
    if let Some(environment) = SHARED.get() {
        return Ok(environment.clone());
    }
    let _guard = INIT.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(environment) = SHARED.get() {
        return Ok(environment.clone());
    }
    let environment = init_environment(onnx_dll)?;
    let _ = SHARED.set(environment.clone());
    Ok(environment)
}

fn init_environment(onnx_dll: Option<&Path>) -> Result<ort::environment::Environment, String> {
    let env_dll = std::env::var_os("ORT_DYLIB_PATH").map(std::path::PathBuf::from);
    if let Some(dir) = onnx_dll.or(env_dll.as_deref()).and_then(Path::parent) {
        prepend_to_path(dir);
    }
    let builder = if let Some(path) = onnx_dll {
        if !path.is_file() {
            return Err(format!(
                "ONNX Runtime library not found at {}",
                path.display()
            ));
        }
        ort::init_from(path).map_err(|err| err.to_string())?
    } else {
        ort::init()
    };
    // CUDA is registered per session in open_session. Registering it on the
    // environment as well makes ONNX Runtime reject the session with
    // "Provider CUDAExecutionProvider has already been registered".
    builder.build().map_err(|err| {
        format!("ONNX Runtime init failed: {err}. Set ORT_DYLIB_PATH to onnxruntime.dll")
    })
}

/// The CUDA provider loads cuDNN with a plain LoadLibrary call, which searches
/// PATH but not the provider's own folder. Setup puts the CUDA 13 and cuDNN
/// DLLs next to onnxruntime.dll, so that folder has to be on PATH.
fn prepend_to_path(dir: &Path) {
    if dir.as_os_str().is_empty() {
        return;
    }
    let current = std::env::var_os("PATH").unwrap_or_default();
    if std::env::split_paths(&current).any(|entry| entry == dir) {
        return;
    }
    let joined = std::env::join_paths(
        std::iter::once(dir.to_path_buf()).chain(std::env::split_paths(&current)),
    );
    if let Ok(joined) = joined {
        std::env::set_var("PATH", joined);
    }
}

fn cuda_provider(device_id: i32) -> ort::ep::ExecutionProviderDispatch {
    ort::ep::CUDA::default()
        .with_device_id(device_id)
        .with_memory_limit(512 * 1024 * 1024)
        .with_arena_extend_strategy(ort::ep::ArenaExtendStrategy::SameAsRequested)
        .with_conv_algorithm_search(ort::ep::cuda::ConvAlgorithmSearch::Heuristic)
        .with_tf32(false)
        .build()
}

fn open_session(
    env: &ort::environment::Environment,
    path: &Path,
    device: InferDevice,
) -> Result<Session, String> {
    let mut builder = Session::builder(env).map_err(|err| err.to_string())?;
    builder = builder
        .with_intra_threads(1)
        .map_err(|err| err.to_string())?;
    if let InferDevice::Cuda { device_id } = device {
        builder = builder
            .with_execution_providers([cuda_provider(device_id)])
            .map_err(|err| format!("session CUDA provider: {err}"))?;
    }
    let mut session = builder
        .commit_from_file(path)
        .map_err(|err| format!("load {}: {err}", path.display()))?;
    if let Err(err) = session.set_workload_type(ort::session::WorkloadType::Efficient) {
        eprintln!(
            "workload type left at default for {}: {err}",
            path.display()
        );
    }
    Ok(session)
}

fn run_content(
    session: &mut Session,
    audio_16k: &[f32],
    feature_dim: usize,
) -> Result<Array3<f32>, String> {
    let input = Array2::from_shape_vec((1, audio_16k.len()), audio_16k.to_vec())
        .map_err(|err| err.to_string())?;
    let outputs = run_first_input(session, &input.into_dyn())?;
    let view = outputs
        .into_dimensionality::<ndarray::Ix3>()
        .map_err(|err| format!("content features must be [batch, time, dim]: {err}"))?;
    let dim = view.shape()[2];
    if dim != feature_dim {
        return Err(format!(
            "content features are {dim}-d, but this voice version expects {feature_dim}-d"
        ));
    }
    Ok(view)
}

fn run_f0(session: &mut Session, audio_16k: &[f32]) -> Result<Vec<f32>, String> {
    let input = Array2::from_shape_vec((1, audio_16k.len()), audio_16k.to_vec())
        .map_err(|err| err.to_string())?;
    let outputs = run_first_input(session, &input.into_dyn())?;
    Ok(outputs.iter().copied().collect())
}

fn run_synthesizer(
    session: &mut Session,
    features: &Array3<f32>,
    coarse: &[i64],
    continuous: &[f32],
    params: &InferParams,
) -> Result<Vec<f32>, String> {
    let time = features.shape()[1];
    let phone = features.clone().into_dyn();
    let lengths = ArrayD::from_shape_vec(ndarray::IxDyn(&[1]), vec![time as i64])
        .map_err(|err| err.to_string())?;
    let speaker = ArrayD::from_shape_vec(ndarray::IxDyn(&[1]), vec![params.speaker_id])
        .map_err(|err| err.to_string())?;

    let phone_tensor = TensorRef::from_array_view(&phone).map_err(|err| err.to_string())?;
    let lengths_tensor = TensorRef::from_array_view(&lengths).map_err(|err| err.to_string())?;
    let speaker_tensor = TensorRef::from_array_view(&speaker).map_err(|err| err.to_string())?;

    let outputs = if params.f0 {
        let frames = time.min(coarse.len()).min(continuous.len());
        let pitch = ArrayD::from_shape_vec(ndarray::IxDyn(&[1, frames]), coarse[..frames].to_vec())
            .map_err(|err| err.to_string())?;
        let pitchf =
            ArrayD::from_shape_vec(ndarray::IxDyn(&[1, frames]), continuous[..frames].to_vec())
                .map_err(|err| err.to_string())?;
        let pitch_tensor = TensorRef::from_array_view(&pitch).map_err(|err| err.to_string())?;
        let pitchf_tensor = TensorRef::from_array_view(&pitchf).map_err(|err| err.to_string())?;
        session
            .run(ort::inputs![
                "phone" => phone_tensor,
                "phone_lengths" => lengths_tensor,
                "pitch" => pitch_tensor,
                "pitchf" => pitchf_tensor,
                "ds" => speaker_tensor,
            ])
            .map_err(|err| format!("synthesizer: {err}"))?
    } else {
        session
            .run(ort::inputs![
                "phone" => phone_tensor,
                "phone_lengths" => lengths_tensor,
                "ds" => speaker_tensor,
            ])
            .map_err(|err| format!("synthesizer: {err}"))?
    };
    first_f32(&outputs)
}

fn run_first_input(session: &mut Session, input: &ArrayD<f32>) -> Result<ArrayD<f32>, String> {
    let tensor = TensorRef::from_array_view(input).map_err(|err| err.to_string())?;
    let outputs = session
        .run(ort::inputs![tensor])
        .map_err(|err| format!("session run: {err}"))?;
    first_array(&outputs)
}

fn first_array(outputs: &ort::session::SessionOutputs<'_>) -> Result<ArrayD<f32>, String> {
    if outputs.len() == 0 {
        return Err("model returned no outputs".into());
    }
    let view = outputs[0]
        .try_extract_array::<f32>()
        .map_err(|err| format!("output is not an f32 tensor: {err}"))?;
    Ok(view.to_owned())
}

fn first_f32(outputs: &ort::session::SessionOutputs<'_>) -> Result<Vec<f32>, String> {
    Ok(first_array(outputs)?.iter().copied().collect())
}

impl std::fmt::Debug for OnnxSessions {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OnnxSessions")
            .field("content", &self.content_path)
            .finish_non_exhaustive()
    }
}
