//! ONNX Runtime backend. Compiled only with `--features onnx`.
//!
//! The runtime library is loaded at startup (`load-dynamic`), so this crate
//! does not link ONNX Runtime or CUDA at build time. Point `ORT_DYLIB_PATH`
//! or the settings field at an ONNX Runtime build made with CUDA 12.8 or
//! newer. That is the first CUDA release with native sm_120 kernels for the
//! RTX 5070. A 12.2 build will fall back or fail on Blackwell.
//!
//! Expected files next to a voice (names follow the RVC `net_g.infer` inputs
//! from the MIT WebUI: phone, lengths, pitch, pitchf, speaker):
//!
//! * `content.onnx` — 16 kHz waveform in, content features out (256-d v1 or 768-d v2)
//! * `f0.onnx` — 16 kHz waveform in, fundamental frequency in hertz out
//! * `synthesizer.onnx` — the RVC generator
//!
//! Retrieval (faiss) is not applied here. `index_rate` is recorded on the
//! manifest and honored by the Python trainer's index file, but blending the
//! index into live features still has to be wired up.

use std::path::{Path, PathBuf};

use ndarray::{Array2, Array3, ArrayD};
use ort::session::Session;
use ort::value::TensorRef;

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

        let environment = init_environment(device, onnx_dll)?;
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

    pub fn run(&mut self, audio_16k: &[f32], params: &InferParams) -> Result<Vec<f32>, String> {
        let features = run_content(&mut self.content, audio_16k, params.feature_dim())?;
        let (coarse, continuous) = if params.f0 {
            let f0 = self.f0.as_mut().ok_or("f0 session missing")?;
            let hz = run_f0(f0, audio_16k)?;
            let continuous: Vec<f32> = hz
                .into_iter()
                .map(|hz| shift_hz(hz, params.pitch_semitones))
                .collect();
            let coarse: Vec<i64> = continuous.iter().copied().map(coarse_f0).collect();
            (coarse, continuous)
        } else {
            (Vec::new(), Vec::new())
        };
        // index_rate is intentionally unused until a faiss binding is added.
        let _ = params.index_rate;
        run_synthesizer(
            &mut self.synthesizer,
            &features,
            &coarse,
            &continuous,
            params,
        )
    }
}

fn init_environment(
    device: InferDevice,
    onnx_dll: Option<&Path>,
) -> Result<ort::environment::Environment, String> {
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
    let builder = match device {
        InferDevice::Cpu => builder,
        InferDevice::Cuda { device_id } => {
            builder.with_execution_providers([cuda_provider(device_id)])
        }
    };
    builder.build().map_err(|err| {
        format!("ONNX Runtime init failed: {err}. Set ORT_DYLIB_PATH to onnxruntime.dll")
    })
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
