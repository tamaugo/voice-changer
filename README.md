# voice-changer

Real-time voice conversion for Windows. Rust app, MIT license. Training and ONNX export call the MIT-licensed [RVC WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI). `obs-rvc` has no license and so-vits-svc is AGPL-3.0; neither is included. VB-CABLE is donationware and is not bundled.

## Install (PowerShell)

Run the Visual Studio and VB-CABLE steps from an elevated PowerShell. Open a new PowerShell after the `winget` block so `cargo` and `py` are on `PATH`.

```powershell
winget install -e --id Rustlang.Rustup --accept-package-agreements --accept-source-agreements
winget install -e --id Git.Git --accept-package-agreements --accept-source-agreements
winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
winget install -e --id Microsoft.VisualStudio.2022.BuildTools --accept-package-agreements --accept-source-agreements --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

`Python.Python.3.10` and `Python.Python.3.11` are the 3.10 and 3.11 packages. This RVC branch ships `requirments_cu128_py312.txt`, so the steps below use 3.12.

```powershell
nvidia-smi
```

The CUDA version in that output must be 12.8 or higher (required for an RTX 5070, `sm_120`). If it is lower, install a current NVIDIA driver, then the [CUDA 12.8 toolkit](https://developer.nvidia.com/cuda-12-8-0-download-archive) and [cuDNN 9](https://developer.nvidia.com/cudnn). Put `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin` and the cuDNN `bin` directory on `PATH`.

```powershell
Invoke-WebRequest -Uri "https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack46.zip" -OutFile "$env:TEMP\VBCABLE_Driver_Pack46.zip"
Expand-Archive -Path "$env:TEMP\VBCABLE_Driver_Pack46.zip" -DestinationPath "$env:TEMP\VBCABLE" -Force
Start-Process -FilePath "$env:TEMP\VBCABLE\VBCABLE_Setup_x64.exe" -Verb RunAs -Wait
```

Reboot after VB-CABLE. Page: https://vb-audio.com/Cable/

```powershell
git clone https://github.com/tamaugo/voice-changer.git
git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git
cd Retrieval-based-Voice-Conversion-WebUI
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirments_cu128_py312.txt
New-Item -ItemType Directory -Force -Path assets\hubert_base, assets\rmvpe, assets\pretrained_v2 | Out-Null
$hf = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main"
$ua = @{ "User-Agent" = "Mozilla/5.0" }
Invoke-WebRequest "$hf/hubert_base/config.json" -OutFile assets\hubert_base\config.json -Headers $ua
Invoke-WebRequest "$hf/hubert_base/preprocessor_config.json" -OutFile assets\hubert_base\preprocessor_config.json -Headers $ua
Invoke-WebRequest "$hf/hubert_base/pytorch_model.bin" -OutFile assets\hubert_base\pytorch_model.bin -Headers $ua
Invoke-WebRequest "$hf/rmvpe.pt" -OutFile assets\rmvpe\rmvpe.pt -Headers $ua
Invoke-WebRequest "$hf/pretrained_v2/f0G40k.pth" -OutFile assets\pretrained_v2\f0G40k.pth -Headers $ua
Invoke-WebRequest "$hf/pretrained_v2/f0D40k.pth" -OutFile assets\pretrained_v2\f0D40k.pth -Headers $ua
Invoke-WebRequest "$hf/pretrained_v2/G40k.pth" -OutFile assets\pretrained_v2\G40k.pth -Headers $ua
Invoke-WebRequest "$hf/pretrained_v2/D40k.pth" -OutFile assets\pretrained_v2\D40k.pth -Headers $ua
cd ..\voice-changer
$ort = "$PWD\onnxruntime"
New-Item -ItemType Directory -Force -Path $ort | Out-Null
Invoke-WebRequest "https://globalcdn.nuget.org/packages/microsoft.ml.onnxruntime.gpu.windows.1.26.0.nupkg" -OutFile "$ort\ort.zip"
Expand-Archive "$ort\ort.zip" -DestinationPath "$ort\pkg" -Force
$env:ORT_DYLIB_PATH = "$ort\pkg\runtimes\win-x64\native\onnxruntime.dll"
$env:PATH = "$ort\pkg\runtimes\win-x64\native;" + $env:PATH
cargo build -p voice-changer --release --features onnx
.\target\release\voice-changer.exe
```

ONNX Runtime 1.26 GPU is the CUDA 12.8 / cuDNN 9 build. In Settings, set Python to `Retrieval-based-Voice-Conversion-WebUI\.venv\Scripts\python.exe`, RVC checkout to that clone, and the ONNX Runtime library to `ORT_DYLIB_PATH`.

## Usage

The window has four actions: Pick a voice, Train a voice, Settings, Close.

`git pull`, then Train a voice. Clips committed in `voices/<Name>/samples/` are listed there; Train uses the checked files and writes the model into that folder. See `voices/README.md`.

Set the app output device to `CABLE Input`. In Discord, set the microphone to `CABLE Output`.

Train with dry run off, or export a checkpoint you already have:

```powershell
.\.venv\Scripts\python.exe ..\voice-changer\python\export_onnx.py --rvc-root . --checkpoint assets\weights\NAME.pth --out-dir ..\voice-changer\voices\NAME --version v2 --name NAME
```

Run that from the RVC checkout, with its venv python. The voice then shows up in Pick a voice.

```powershell
.\target\release\voice-changer.exe --benchmark --voice NAME --seconds 5
.\target\release\voice-changer.exe --self-test
```

`--benchmark` prints process time divided by block length. Under 5% is the live target on an RTX 5070. Settings has the same check.
