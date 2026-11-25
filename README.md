# voice-changer

A free, open-source real-time voice changer. The app is Rust, the license is MIT, and the Windows build talks to the sound device through WASAPI. Live conversion is designed to leave the GPU almost idle so a game can keep the RTX 5070. Training is allowed to use the whole card.

Capture, conversion, and playback run in one process. A voice is a folder of ONNX files exported from the MIT-licensed RVC WebUI. Until those files exist, the hook passes audio through and says why. Training and export shell out to that checkout instead of reimplementing it.

## Layout

| Crate | Role |
| --- | --- |
| `crates/rustvc-audio` | Device list, lock-free capture/playback rings, rubato resampling, the `SampleProcessor` hook |
| `crates/rustvc-infer` | Voice manifests, the RVC pitch-bin contract, the GPU duty-cycle budget, optional ONNX Runtime sessions |
| `crates/rustvc-train` | Spawns `python/rvc_train.py` and parses its progress protocol |
| `crates/rustvc-app` | The dark window: pick a voice, train a voice, settings, close |
| `python/rvc_train.py` | Original wrapper. It shells out to the RVC CLI, then to the exporter |
| `python/export_onnx.py` | Imports the RVC checkout and writes `content.onnx`, `f0.onnx`, `synthesizer.onnx`, `features.npy` |

```text
microphone --WASAPI--> ring buffer --processor hook--> ring buffer --WASAPI--> virtual cable input
                                                                              |
                                                           Discord selects the cable output
```

On Windows, cpal's default host is WASAPI. The same binary also builds on Linux (ALSA) so the loop and the window can be checked without a Windows machine.

## GPU budget

The live target is about 5% GPU on an RTX 5070, averaged over a second, while a game is running. Training ignores that cap.

A voice block does not need the GPU for the whole block. One short inference burst per block is enough, then the card goes back to the game. The duty cycle is

```text
duty% = inference milliseconds / block milliseconds * 100
```

A 100 ms block (4800 frames at 48 kHz) stays inside a 5% budget when the burst finishes in 5 ms or less. The status line shows this duty cycle. It is the time the process was blocked in `process`, not an nvidia-smi utilization reading. A small kernel does not fill a 5070, so the Task Manager figure is usually lower than the duty cycle.

The settings that keep the live path small:

- One audio block in flight, on a thread that is not the WASAPI callback.
- ONNX Runtime rather than a PyTorch process. The runtime loads at startup (`ort` with `load-dynamic`) and is not linked into the default build.
- CUDA execution provider with a 512 MB arena, heuristic convolution search (no exhaustive warmup), TF32 off, and one intra-op thread.
- `WorkloadType::Efficient` on each session, so the provider is asked to prefer efficiency over peak clocks.
- FP16 checkpoints (`is_half` on the training side). The heavy part of RVC is the content encoder. A 768-d v2 ContentVec export, not a full-precision PyTorch runtime, is what has to fit in the burst.
- If a measurement is over budget, lengthen the block up to 400 ms. Past that, drop to the CPU provider or a smaller export. A longer buffer is already a lot of conversational latency.

Blackwell (compute capability 12.0, `sm_120`) needs CUDA 12.8 or newer. Older toolkits have no native kernel image for the RTX 5070. Use an ONNX Runtime GPU build made with CUDA 12.8+ (ORT 1.21 through 1.26 publish CUDA 12.8 packages; 1.27+ also publishes a separate CUDA 12.x package built with 12.8). Point settings, or `ORT_DYLIB_PATH`, at that `onnxruntime.dll`.

## Why ONNX Runtime

| Option | Why it was not the live backend |
| --- | --- |
| `ort` + CUDA EP | Selected. A prebuilt runtime, IO-bound sessions, a hard memory cap, and no PyTorch process sitting on the GPU during a game. |
| `tch` | Pulls libtorch into the live process. That is the same heavy runtime training already uses, and it fights the 5% budget. |
| `candle` | Would mean reimplementing the RVC generator. The checkpoint format would drift from the trainer. |
| `cudarc` | Kernel-level CUDA. The model math would be new code instead of the existing ONNX graphs. |

Build the CUDA hook with:

```text
cargo build -p voice-changer --release --features onnx
```

The default `cargo build` stays a passthrough and does not download ONNX Runtime. The `cuda` feature of `ort` only enables the execution-provider API. The CUDA toolkit is not required to compile it. The matching runtime DLL is required to run it.

## Why training stays in Python

A faithful RVC trainer is the WebUI's pipeline: slice the dataset, extract RMVPE pitch, extract ContentVec or HuBERT features, train the NSF-HiFiGAN generator against a discriminator, then build a faiss index. That stack is tens of thousands of lines of PyTorch and it is what writes the checkpoint the community already shares. Rewriting it in candle or tch would fork the format for no gain on the live path.

`python/rvc_train.py` runs the CLI documented in the WebUI's `docs/en/cli.md`:

```text
train/preprocess.py
train/dataset/extract_f0.py
train/dataset/extract_hubert_feature.py
train/train.py
train/train_index.py
```

The script checks that a Blackwell GPU sees a PyTorch wheel built with CUDA 12.8 or newer (`sm_120` in `torch.cuda.get_arch_list()`). PyTorch 2.7 and later ship those wheels:

```text
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Then install the checkout's `requirments_cu128_py312.txt` (the upstream filename spells it that way). The driver must be recent enough to report CUDA 12.8 or higher in `nvidia-smi` (R570 or newer).

Dry run skips the checkout and the GPU. It still writes `voices/<name>/voice.toml` so the picker can show the voice. The manifest's `trained` flag stays false.

## Windows build

1. Install the stable Rust toolchain and the MSVC C++ build tools (the `x86_64-pc-windows-msvc` target).
2. From this directory:

```text
cargo run -p voice-changer
cargo run -p voice-changer -- --self-test
cargo run -p voice-changer -- --list-devices
```

`--self-test` checks the passthrough loop, resampling, the RVC pitch bins, and a training dry run. It does not open a window.

`cargo check --target x86_64-pc-windows-msvc` is how the Windows build is type-checked from another OS. cpal selects WASAPI for that target. The check does not link a `.exe`; producing the binary still wants the MSVC linker on Windows.

Python is only required for training and ONNX export. The live window runs without it once the `.onnx` files exist.

## Try it on Windows

1. Install Rust (stable, MSVC) and the Visual Studio C++ build tools.
2. Install a driver that reports CUDA 12.8 or newer (`nvidia-smi`). An RTX 5070 needs this for `sm_120`.
3. Install VB-CABLE if you want Discord to see the converted voice. In the app, set Output to `CABLE Input` and set Discord's microphone to `CABLE Output`.
4. Clone the MIT RVC WebUI and install its CUDA 12.8 requirements, including PyTorch from the cu128 index:

```text
git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r Retrieval-based-Voice-Conversion-WebUI\requirments_cu128_py312.txt
```

Download the HuBERT/ContentVec folder into `assets\hubert_base`, the RMVPE file into `assets\rmvpe\rmvpe.pt`, and the v2 pretrained generator/discriminator into `assets\pretrained_v2`, the same files the WebUI's own docs ask for.

5. Build the converter with the ONNX feature and an ONNX Runtime DLL built with CUDA 12.8 or newer (ORT 1.21–1.26, or the CUDA 12.x package of 1.27+):

```text
cargo build -p voice-changer --release --features onnx
set ORT_DYLIB_PATH=C:\path\to\onnxruntime.dll
target\release\voice-changer.exe
```

6. In Settings, set Python, the RVC checkout path, and the ONNX Runtime DLL. Save.
7. Train a voice: put wavs in one folder, open Train a voice, turn dry run off, and start. The trainer preprocesses, extracts F0 and HuBERT features, trains, builds the index, then `python/export_onnx.py` writes `content.onnx`, `f0.onnx`, `synthesizer.onnx`, and `features.npy` into `voices\<name>\`. The voice shows up in Pick a voice.
8. To export a checkpoint you already have, without training again:

```text
python python\export_onnx.py --rvc-root C:\rvc --checkpoint C:\rvc\assets\weights\demo.pth --out-dir voices\demo --version v2 --name demo
```

9. Confirm the live GPU budget on the 5070:

```text
voice-changer.exe --benchmark --voice YourVoice --seconds 5
```

Settings also has Benchmark GPU duty. The number is process time divided by the audio block. Under 5% means the burst fits the live target. Task Manager's GPU column is usually lower, because a small graph does not fill the chip for that whole burst.

`voice-changer.exe --self-test` checks the audio loop, the pitch bins, retrieval/crossfade math, and a training dry run. It does not need a GPU.

## Virtual microphone

The app does not install a kernel driver. Playback is a normal WASAPI render stream. You choose the device in Settings. If that device is the recording side of a virtual cable, other programs see the far end as a microphone.

Recommended setup, VB-CABLE (donationware from VB-Audio, not an open-source license, not bundled):

1. Install VB-CABLE and reboot if Windows asks.
2. In Settings, set Output to `CABLE Input`.
3. In Discord, set the microphone to `CABLE Output`.
4. Leave the game on the real speakers or headset. The cable only carries the converted voice.

An open-source alternative is Synchronous Audio Router (GPL-3.0). It also creates endpoints you can select. It is not bundled. Shipping it inside this MIT app, or linking its driver, would pull GPL obligations onto that combined distribution. Selecting an endpoint the user installed separately does not.

Trade-off. A private kernel driver could appear as a single microphone with no extra install, and it could avoid the extra buffer a cable adds. It would also need a signed driver, an installer, and a maintenance burden that does not help the voice model. A cable the user already trusts is the smaller risk, at the cost of one install and a few tens of milliseconds. The block size in Settings is the rest of the latency (100 ms at the default 4800 frames / 48 kHz).

## Window

The home screen has four actions: Pick a voice, Train a voice, Settings, and Close. Pick installs a voice into the live hook. Train asks for a dataset folder and either dry-runs or calls the RVC CLI. Settings chooses devices, the block size, the GPU budget, and the paths to Python, the RVC checkout, and `onnxruntime.dll`.

Voices are folders under `voices/` with a `voice.toml`. Passthrough is always available and is not a file.

## Live model contract

When `synthesizer.onnx`, `content.onnx`, and (for F0 voices) `f0.onnx` are present and the binary was built with `--features onnx`, each block does the following. The synthesizer tensor names match the RVC generator inputs published by the MIT WebUI (`phone`, `phone_lengths`, `pitch`, `pitchf`, `ds`):

1. Keep about 300 ms of context, resample that window to 16 kHz with a reused FFT plan.
2. Run the content encoder. v1 is 256-d, v2 is 768-d.
3. If `features.npy` is present, blend each frame with its nearest training rows (brute-force L2, inverse-distance weights, `index_rate`). This is the faiss blend without linking faiss into the live process. The exporter fills the file from `total_fea.npy` or, if faiss is installed, from the added index.
4. Run the F0 model, shift by semitones (`2^(semitones/12)`), and quantize to bins 1..=255 between 50 Hz and 1100 Hz.
5. On unvoiced frames, mix the converted features back toward the original by the `protect` amount (default 0.33).
6. Run the synthesizer, resample to the playback rate, match loudness with `rms_mix` (default 0.25), and crossfade the block join (`crossfade_ms`, default 10).

A missing file, or a build without the `onnx` feature, keeps the audio path open and labels the voice as passthrough.

The audio thread uses a rubato slip resampler when capture and playback share a nominal rate, and a polynomial async resampler when they do not. Both nudge the ratio from the playback-ring fill so the two clocks can drift without a growing underrun. When the device publishes a buffer-size range, the callback quantum is the Settings value `callback frames` (default 480, which is 10 ms at 48 kHz). That is the WASAPI shared-mode buffer. Devices that do not publish a range keep the default period.

## What was borrowed

Nothing below is vendored. Training and the base models stay in their own checkouts. The Rust code in this repo is original.

| Piece | Project | How it is used | License |
| --- | --- | --- | --- |
| WASAPI / ALSA I/O | [RustAudio/cpal](https://github.com/RustAudio/cpal) 0.18 | Capture and playback | Apache-2.0 |
| Resampling | [HEnquist/rubato](https://github.com/HEnquist/rubato) 5.1 | 16 kHz feature rate and device-rate mismatch | MIT OR Apache-2.0 |
| Lock-free rings | [agerasev/ringbuf](https://github.com/agerasev/ringbuf) 0.6 | Callback queues | MIT OR Apache-2.0 |
| Window | [emilk/egui](https://github.com/emilk/egui) and eframe 0.37 | The four-action dashboard | MIT OR Apache-2.0 |
| ONNX bindings | [pykeio/ort](https://github.com/pykeio/ort) 2.0.0-rc.15 | Optional. CUDA execution provider, loaded at runtime | MIT OR Apache-2.0 |
| ONNX Runtime | [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) | Not vendored. User supplies a CUDA 12.8+ build | MIT |
| RVC trainer and model contract | [RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) | Subprocess CLI. Pitch-bin and generator inputs | MIT |
| ContentVec | [auspicious3000/contentvec](https://github.com/auspicious3000/contentvec) | Feature encoder used inside the RVC checkout, not linked here | MIT |
| VITS | [jaywalnut310/vits](https://github.com/jaywalnut310/vits) | Generator family used by RVC training, not linked here | MIT |
| HiFi-GAN | [jik876/hifi-gan](https://github.com/jik876/hifi-gan) | Vocoder pieces inside RVC, not linked here | MIT |
| RMVPE | [Dream-High/RMVPE](https://github.com/Dream-High/RMVPE) | F0 model used by the RVC CLI, not linked here | Apache-2.0 |
| faiss | [facebookresearch/faiss](https://github.com/facebookresearch/faiss) | Index built by `train_index.py`, not linked from Rust | MIT |
| audio slicer | [openvpi/audio-slicer](https://github.com/openvpi/audio-slicer) | Used by RVC preprocess, not linked here | MIT |
| PyTorch | [pytorch/pytorch](https://github.com/pytorch/pytorch) | Training only, cu128 wheels for `sm_120` | BSD-3-Clause |
| serde, toml, thiserror | their upstream crates | Config, manifests, errors | MIT OR Apache-2.0 |
| VB-CABLE | [vb-audio.com/Cable](https://vb-audio.com/Cable/) | Optional virtual microphone the user installs | Proprietary donationware. Not bundled |
| Synchronous Audio Router | [eiz/SynchronousAudioRouter](https://github.com/eiz/SynchronousAudioRouter) | Optional open-source endpoints the user installs | GPL-3.0. Not bundled |

RVC's own tree also documents LGPL FFmpeg builds and other tools used by its WebUI. This app does not launch that WebUI and does not ship FFmpeg. Preprocess inside a real training run uses whatever the checkout uses.

### Looked at, not used

| Project | License | Why it is not in this tree |
| --- | --- | --- |
| [RVC-Project/obs-rvc](https://github.com/RVC-Project/obs-rvc) | No license file. All rights reserved by default | It is the Rust RVC inference port and it describes an ONNX pipeline. No license means the code cannot be copied or linked. The stage order here follows the MIT WebUI's published `infer` inputs instead. |
| [svc-develop-team/so-vits-svc](https://github.com/svc-develop-team/so-vits-svc) | AGPL-3.0 | Copying or linking it would put this app under the AGPL. Streaming overlap and F0-conditioned conversion are general ideas. None of that code is included. |

## What is left

The training path, the ONNX export, the live DSP, the drift resampler, and the duty-cycle benchmark are wired. What this repository cannot do by itself is run them on an RTX 5070: that needs the RVC weights, a CUDA 12.8 runtime, and a listen test.

Still open:

- Confirm `export_onnx.py` traces the current WebUI's Transformers HuBERT and the generator on your machine. A trace error is printed and the `.pth` is still saved.
- Run `--benchmark --voice YourVoice` during a game and, if the duty cycle is over 5%, lengthen the block or switch inference to CPU.
- Formant shift, a noise gate, and an installer.
- Very large retrieval banks. The live search is exact and fine for a few thousand rows; it is not an IVF index.
