#!/usr/bin/env python3
"""Run the RVC WebUI training CLI, then export_onnx.py.

Prints PROGRESS, LOG, VOICE, and ERROR lines for the Rust UI.
--dry-run writes a manifest and does not train.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"}
SR_TAG = {32000: "32k", 40000: "40k", 48000: "48k"}


def emit(kind: str, message: str = "") -> None:
    text = message.strip()
    if text:
        print(f"{kind} {text}", flush=True)
    else:
        print(kind, flush=True)


def progress(fraction: float, message: str) -> None:
    emit("PROGRESS", f"{fraction:.3f} {message}")


def fail(message: str) -> None:
    emit("ERROR", message)
    sys.exit(1)


def count_audio(dataset: Path) -> int:
    return sum(1 for path in dataset.iterdir() if path.suffix.lower() in AUDIO_SUFFIXES)


def write_manifest(
    voices_dir: Path,
    name: str,
    args: argparse.Namespace,
    trained: bool,
    note: str,
    model: str = "",
    index: str = "",
    content_model: str = "",
    f0_model: str = "",
    features: str = "",
) -> Path:
    destination = voices_dir / name.replace(" ", "_")
    destination.mkdir(parents=True, exist_ok=True)

    def local_name(source: str) -> str:
        if not source:
            return ""
        source_path = Path(source)
        if not source_path.is_file():
            return source_path.name
        target = destination / source_path.name
        if source_path.resolve() != target.resolve():
            shutil.copy2(source_path, target)
        return target.name

    model_name = local_name(model)
    index_name = local_name(index)
    content_name = local_name(content_model)
    f0_name = local_name(f0_model)
    feature_name = local_name(features)
    note = note.replace('"', "'")
    lines = [
        f'name = "{args.name}"',
        f'version = "{args.version}"',
        f"sample_rate = {args.sample_rate}",
        f"f0 = {'true' if args.f0 else 'false'}",
        "pitch = 0.0",
        "index_rate = 0.75",
        'f0_method = "rmvpe"',
        f'model = "{model_name}"',
        f'index = "{index_name}"',
        f'content_model = "{content_name}"',
        f'f0_model = "{f0_name}"',
        f'features = "{feature_name}"',
        "rms_mix = 0.25",
        "protect = 0.33",
        "crossfade_ms = 10.0",
        f"trained = {'true' if trained else 'false'}",
        f'note = "{note}"',
        "",
    ]
    manifest = destination / "voice.toml"
    manifest.write_text("\n".join(lines), encoding="utf-8")
    return manifest


def cuda_version_at_least(version: str, major: int, minor: int) -> bool:
    parts = version.split(".")
    try:
        found = (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
    except ValueError:
        return False
    return found >= (major, minor)


def check_blackwell(gpu: int) -> None:
    try:
        import torch
    except ImportError:
        fail(
            "PyTorch is not installed in this interpreter. "
            "Install a CUDA 12.8+ wheel: pip install torch torchvision torchaudio "
            "--index-url https://download.pytorch.org/whl/cu128"
        )
    if not torch.cuda.is_available():
        fail("torch.cuda.is_available() is false. Training on the RTX 5070 needs a working CUDA driver (R570 or newer).")
    if gpu >= torch.cuda.device_count():
        fail(f"GPU index {gpu} is out of range; torch sees {torch.cuda.device_count()} device(s).")
    capability = torch.cuda.get_device_capability(gpu)
    archs = [str(arch) for arch in torch.cuda.get_arch_list()]
    cuda = torch.version.cuda or ""
    name = torch.cuda.get_device_name(gpu)
    emit("LOG", f"torch {torch.__version__} cuda {cuda} device {name} capability {capability} archs {archs}")
    if capability >= (12, 0):
        if not cuda_version_at_least(cuda, 12, 8):
            fail(
                f"{name} is sm_{capability[0]}{capability[1]}, but this PyTorch was built with CUDA {cuda}. "
                "Blackwell needs the cu128 wheel (CUDA 12.8) or newer."
            )
        if not any("120" in arch or arch.endswith("12") or "sm_12" in arch for arch in archs):
            fail(f"PyTorch arch list {archs} has no sm_120 kernel. Reinstall torch from the cu128 index.")


def pretrained_paths(root: Path, version: str, sr_tag: str, f0: bool) -> tuple[Path, Path]:
    folder = root / ("assets/pretrained_v2" if version == "v2" else "assets/pretrained")
    prefix = "f0" if f0 else ""
    generator = folder / f"{prefix}G{sr_tag}.pth"
    discriminator = folder / f"{prefix}D{sr_tag}.pth"
    return generator, discriminator


def run_step(root: Path, args: list[str]) -> None:
    emit("LOG", " ".join(args))
    process = subprocess.Popen(
        args,
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert process.stdout is not None
    for line in process.stdout:
        stripped = line.rstrip()
        if stripped:
            emit("LOG", stripped)
    code = process.wait()
    if code != 0:
        fail(f"{Path(args[1]).name} exited with status {code}")


def newest(paths: list[Path]) -> Path | None:
    files = [path for path in paths if path.is_file()]
    if not files:
        return None
    return max(files, key=lambda path: path.stat().st_mtime)


def find_checkpoint(root: Path, name: str) -> Path | None:
    preferred = root / "assets" / "weights" / f"{name}.pth"
    if preferred.is_file():
        return preferred
    logs = root / "logs" / name
    if not logs.is_dir():
        return None
    return newest(list(logs.rglob("*.pth")))


def find_index(root: Path, name: str) -> Path | None:
    candidates = []
    for folder in (root / "assets" / "indices", root / "logs" / name):
        if folder.is_dir():
            candidates.extend(folder.rglob("*.index"))
    added = [path for path in candidates if "added" in path.name]
    return newest(added or candidates)


def train(args: argparse.Namespace) -> None:
    dataset = Path(args.dataset).expanduser().resolve()
    if not dataset.is_dir():
        fail(f"dataset directory does not exist: {dataset}")
    audio_files = count_audio(dataset)
    emit("LOG", f"dataset {dataset} has {audio_files} audio file(s)")
    if audio_files == 0 and not args.dry_run:
        fail("the dataset directory has no wav, flac, mp3, ogg, or m4a files in its top level")

    sr_tag = SR_TAG.get(args.sample_rate)
    if sr_tag is None:
        fail("sample rate must be 32000, 40000, or 48000")

    voices_dir = Path(args.voices_dir).expanduser().resolve()
    voices_dir.mkdir(parents=True, exist_ok=True)
    safe_name = args.name.strip().replace(" ", "_")

    if args.dry_run:
        stages = [
            (0.15, "preprocess dataset"),
            (0.35, "extract f0"),
            (0.55, "extract content features"),
            (0.80, "train generator"),
            (0.95, "train retrieval index"),
        ]
        for fraction, message in stages:
            progress(fraction, message)
            if args.sleep > 0:
                time.sleep(args.sleep)
        manifest = write_manifest(
            voices_dir,
            safe_name,
            args,
            trained=False,
            note="Dry run only. No weights were trained. Set the RVC checkout path and turn dry run off.",
        )
        progress(1.0, "dry run finished")
        emit("VOICE", str(manifest))
        emit("LOG", "dry run did not call the RVC trainer")
        return

    root = Path(args.rvc_root).expanduser().resolve() if args.rvc_root else None
    if root is None or not (root / "train" / "train.py").is_file():
        fail("set the RVC checkout path to a clone of Retrieval-based-Voice-Conversion-WebUI (train/train.py was not found)")

    check_blackwell(args.gpu)
    generator, discriminator = pretrained_paths(root, args.version, sr_tag, args.f0)
    if not generator.is_file() or not discriminator.is_file():
        fail(
            "pretrained checkpoints are missing. Expected "
            f"{generator.name} and {discriminator.name} under {generator.parent}. "
            "Download the RVC v2 pretrained weights into the checkout before training."
        )

    python = sys.executable
    experiment = str(Path("logs") / safe_name)
    f0_flag = "1" if args.f0 else "0"
    half = "true"

    progress(0.05, "preprocess dataset")
    run_step(
        root,
        [
            python,
            "train/preprocess.py",
            str(dataset),
            str(args.sample_rate),
            str(args.workers),
            experiment,
            "False",
            "3.7",
        ],
    )

    progress(0.25, "extract f0")
    if args.f0:
        run_step(
            root,
            [
                python,
                "train/dataset/extract_f0.py",
                "cuda",
                "1",
                "0",
                str(args.gpu),
                experiment,
                half,
            ],
        )

    progress(0.45, "extract content features")
    run_step(
        root,
        [
            python,
            "train/dataset/extract_hubert_feature.py",
            f"cuda:{args.gpu}",
            "1",
            "0",
            str(args.gpu),
            experiment,
            args.version,
            half,
        ],
    )

    progress(0.60, "train generator")
    save_every = 5 if args.epochs >= 5 else 1
    run_step(
        root,
        [
            python,
            "train/train.py",
            "-e",
            safe_name,
            "-sr",
            sr_tag,
            "-f0",
            f0_flag,
            "-bs",
            str(args.batch_size),
            "-g",
            str(args.gpu),
            "-te",
            str(args.epochs),
            "-se",
            str(save_every),
            "-pg",
            str(generator),
            "-pd",
            str(discriminator),
            "-l",
            "0",
            "-c",
            "0",
            "-sw",
            "1",
            "-v",
            args.version,
        ],
    )

    progress(0.90, "train retrieval index")
    run_step(
        root,
        [
            python,
            "train/train_index.py",
            safe_name,
            args.version,
            "assets/indices",
            str(args.workers),
            "single",
        ],
    )

    checkpoint = find_checkpoint(root, safe_name)
    if checkpoint is None:
        fail(f"training finished but no .pth checkpoint was found for {safe_name}")
    index = find_index(root, safe_name)
    voice_dir = voices_dir / safe_name
    voice_dir.mkdir(parents=True, exist_ok=True)
    export_note = export_checkpoint(root, checkpoint, voice_dir, args.version, args.f0, safe_name)
    content = voice_dir / "content.onnx"
    synthesizer = voice_dir / "synthesizer.onnx"
    f0_file = voice_dir / "f0.onnx"
    features = voice_dir / "features.npy"
    exported = synthesizer.is_file() and content.is_file()
    manifest = write_manifest(
        voices_dir,
        safe_name,
        args,
        trained=True,
        note=export_note if exported else f"Trained. {export_note}",
        model=str(synthesizer if exported else checkpoint),
        index=str(index) if index else "",
        content_model=str(content) if content.is_file() else "",
        f0_model=str(f0_file) if f0_file.is_file() else "",
        features=str(features) if features.is_file() else "",
    )
    progress(1.0, "training finished")
    emit("VOICE", str(manifest))


def export_checkpoint(root: Path, checkpoint: Path, voice_dir: Path, version: str, f0: bool, name: str) -> str:
    script = Path(__file__).resolve().parent / "export_onnx.py"
    if not script.is_file():
        return "ONNX export script is missing."
    progress(0.95, "export ONNX")
    command = [
        sys.executable,
        str(script),
        "--rvc-root",
        str(root),
        "--checkpoint",
        str(checkpoint),
        "--out-dir",
        str(voice_dir),
        "--version",
        version,
        "--name",
        name,
        "--f0" if f0 else "--no-f0",
    ]
    emit("LOG", " ".join(command))
    process = subprocess.Popen(
        command,
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert process.stdout is not None
    for line in process.stdout:
        stripped = line.rstrip()
        if stripped:
            emit("LOG", stripped)
    code = process.wait()
    if code != 0:
        return f"ONNX export failed with status {code}. The .pth is still in the voice folder."
    return "Trained and exported to ONNX. Select the voice with a binary built --features onnx."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or dry-run RVC training for voice-changer.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sample-rate", type=int, default=40000)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--version", choices=("v1", "v2"), default="v2")
    parser.add_argument("--voices-dir", default="voices")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rvc-root", default="")
    parser.add_argument("--sleep", type=float, default=0.12)
    parser.add_argument("--f0", dest="f0", action="store_true")
    parser.add_argument("--no-f0", dest="f0", action="store_false")
    parser.set_defaults(f0=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.name.strip():
        fail("voice name is empty")
    train(args)


if __name__ == "__main__":
    main()
