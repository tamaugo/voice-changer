#!/usr/bin/env python3
"""Export an RVC .pth to content.onnx, f0.onnx, synthesizer.onnx, and features.npy.

Imports the MIT RVC WebUI at --rvc-root. Does not vendor that repo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def emit(kind: str, message: str) -> None:
    print(f"{kind} {message}", flush=True)


def fail(message: str) -> None:
    emit("ERROR", message)
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an RVC voice to ONNX for voice-changer.")
    parser.add_argument("--rvc-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--version", choices=("v1", "v2"), default="v2")
    parser.add_argument("--name", default="")
    parser.add_argument("--f0", dest="f0", action="store_true")
    parser.add_argument("--no-f0", dest="f0", action="store_false")
    parser.set_defaults(f0=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.rvc_root).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if not (root / "infer").is_dir():
        fail(f"{root} does not look like an RVC WebUI checkout (missing infer/)")
    if not checkpoint.is_file():
        fail(f"checkpoint not found: {checkpoint}")
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(root))
    try:
        import torch
    except ImportError:
        fail(
            "PyTorch is not installed. Install the CUDA 12.8 wheel before exporting: "
            "pip install torch --index-url https://download.pytorch.org/whl/cu128"
        )

    import torch.nn as nn

    from infer.hubert import load_hubert_model
    from infer.rmvpe import RMVPE
    from infer.rtrvc import get_synthesizer

    device = torch.device("cpu")
    version = args.version
    feature_dim = 256 if version == "v1" else 768
    emit("LOG", f"exporting {checkpoint.name} ({version}, f0={args.f0}) on CPU")

    export_content(load_hubert_model(device, is_half=False), version, out_dir / "content.onnx", torch, nn)
    if args.f0:
        weights = root / "assets" / "rmvpe" / "rmvpe.pt"
        if not weights.is_file():
            fail(f"RMVPE checkpoint not found at {weights}")
        rmvpe = RMVPE(str(weights), is_half=False, device="cpu")
        export_f0(rmvpe, out_dir / "f0.onnx", torch, nn)
    net, _config = get_synthesizer(str(checkpoint), device)
    net.eval()
    export_synthesizer(net, args.f0, feature_dim, out_dir / "synthesizer.onnx", torch, nn)
    copied = export_feature_bank(root, args.name or checkpoint.stem, out_dir / "features.npy")
    if copied:
        emit("LOG", f"wrote retrieval bank {copied}")
    else:
        emit("LOG", "no total_fea.npy or faiss index found; live retrieval will stay off")
    emit("LOG", f"ONNX files are in {out_dir}")


def export_content(model, version: str, path: Path, torch, nn) -> None:
    class Content(nn.Module):
        def __init__(self, model, version: str):
            super().__init__()
            self.model = model
            self.version = version

        def forward(self, source):
            if self.version == "v1":
                outputs = self.model(
                    input_values=source,
                    output_hidden_states=True,
                    return_dict=True,
                )
                return self.model.final_proj(outputs.hidden_states[9])
            return self.model(input_values=source, return_dict=True).last_hidden_state

    module = Content(model, version).eval()
    example = torch.randn(1, 16000)
    emit("LOG", "tracing content encoder")
    write_onnx(
        torch,
        module,
        (example,),
        path,
        ["source"],
        ["features"],
        {"source": {1: "samples"}, "features": {1: "frames"}},
    )


def export_f0(rmvpe, path: Path, torch, nn) -> None:
    """16 kHz waveform to hertz. Uses the RMVPE mel basis and 360 cent bins."""

    class F0(nn.Module):
        def __init__(self, rmvpe):
            super().__init__()
            self.net = rmvpe.model
            basis = rmvpe.mel_extractor.mel_basis.detach().float().cpu()
            self.register_buffer("mel_basis", basis)
            cents = torch.arange(360, dtype=torch.float32) * 20.0 + 1997.3794084376191
            self.register_buffer("cents", cents)
            self.n_fft = int(rmvpe.mel_extractor.n_fft)
            self.hop = int(rmvpe.mel_extractor.hop_length)
            self.win = int(rmvpe.mel_extractor.win_length)
            self.clamp = float(rmvpe.mel_extractor.clamp)

        def forward(self, audio):
            window = torch.hann_window(self.win, device=audio.device, dtype=audio.dtype)
            spectrum = torch.stft(
                audio,
                n_fft=self.n_fft,
                hop_length=self.hop,
                win_length=self.win,
                window=window,
                center=True,
                return_complex=False,
            )
            magnitude = torch.sqrt(spectrum[..., 0] ** 2 + spectrum[..., 1] ** 2)
            mel = torch.matmul(self.mel_basis, magnitude)
            mel = torch.log(torch.clamp(mel, min=self.clamp))
            frames = mel.shape[-1]
            pad = (32 - frames % 32) % 32
            if pad:
                mel = torch.nn.functional.pad(mel, (0, pad))
            hidden = self.net(mel)
            hidden = hidden[0, :frames]
            return decode_hz(hidden, self.cents)

    module = F0(rmvpe).eval()
    example = torch.randn(1, 16000)
    emit("LOG", "tracing RMVPE")
    write_onnx(
        torch,
        module,
        (example,),
        path,
        ["audio"],
        ["f0"],
        {"audio": {1: "samples"}, "f0": {0: "frames"}},
    )


def decode_hz(hidden, cents):
    import torch
    import torch.nn.functional as functional

    center = hidden.argmax(dim=-1)
    padded = functional.pad(hidden, (4, 4))
    padded_center = center + 4
    offsets = torch.arange(-4, 5, device=hidden.device)
    index = padded_center.unsqueeze(1) + offsets.unsqueeze(0)
    gathered = torch.gather(padded, 1, index)
    cents_map = functional.pad(cents, (4, 4))
    gathered_cents = torch.gather(cents_map.expand(index.shape[0], -1), 1, index)
    weight = gathered.sum(dim=1).clamp_min(1.0e-6)
    decoded = (gathered * gathered_cents).sum(dim=1) / weight
    peak = hidden.max(dim=-1).values
    decoded = torch.where(peak <= 0.03, torch.zeros_like(decoded), decoded)
    f0 = 10.0 * torch.pow(2.0, decoded / 1200.0)
    return torch.where(decoded <= 0, torch.zeros_like(f0), f0)


def _patch_relative_attention(net, torch) -> int:
    """Swap RVC's relative-position helpers for gather-based versions.

    The stock helpers call int()/max() on the sequence length, which the ONNX
    tracer freezes to the example length. These versions compute the same
    values from torch.arange on traced sizes, so the graph stays dynamic.
    """
    import types

    def get_relative_embeddings(self, relative_embeddings, length):
        window = self.window_size
        positions = torch.arange(2 * length - 1, device=relative_embeddings.device) - (length - 1)
        index = torch.clamp(positions + window, 0, 2 * window)
        mask = (positions.abs() <= window).to(relative_embeddings.dtype)
        return relative_embeddings[:, index] * mask.view(1, -1, 1)

    def relative_to_absolute(self, x):
        batch, heads, length = x.size(0), x.size(1), x.size(2)
        rows = torch.arange(length, device=x.device)
        index = rows.view(1, -1) - rows.view(-1, 1) + (length - 1)
        index = index.view(1, 1, length, length).expand(batch, heads, length, length)
        return torch.gather(x, 3, index)

    def absolute_to_relative(self, x):
        batch, heads, length = x.size(0), x.size(1), x.size(2)
        rows = torch.arange(length, device=x.device)
        offsets = torch.arange(2 * length - 1, device=x.device) - (length - 1)
        columns = rows.view(-1, 1) + offsets.view(1, -1)
        valid = ((columns >= 0) & (columns < length)).to(x.dtype)
        index = torch.clamp(columns, 0, length - 1)
        index = index.view(1, 1, length, 2 * length - 1).expand(batch, heads, length, 2 * length - 1)
        return torch.gather(x, 3, index) * valid.view(1, 1, length, 2 * length - 1)

    patched = 0
    for module in net.modules():
        if getattr(module, "window_size", None) is not None and hasattr(module, "_relative_position_to_absolute_position"):
            module._get_relative_embeddings = types.MethodType(get_relative_embeddings, module)
            module._relative_position_to_absolute_position = types.MethodType(relative_to_absolute, module)
            module._absolute_position_to_relative_position = types.MethodType(absolute_to_relative, module)
            patched += 1
    return patched


def _check_patch(stock, net, use_f0: bool, feature_dim: int, torch, style: str) -> None:
    """Compare patched and stock attention on a few lengths before export."""
    for frames in (7, 33, 100):
        torch.manual_seed(frames)
        phone = torch.randn(1, frames, feature_dim)
        lengths = torch.tensor([frames], dtype=torch.long)
        pitch = torch.randint(1, 255, (1, frames), dtype=torch.long)
        pitchf = torch.full((1, frames), 150.0)
        speaker = torch.zeros(1, dtype=torch.long)
        outputs = []
        for model in (stock, net):
            torch.manual_seed(0)
            with torch.no_grad():
                if use_f0 and style == "realtime":
                    out = model.infer(phone, lengths, pitch, pitchf, speaker, 0, frames, frames)[0]
                elif use_f0:
                    out = model.infer(phone, lengths, pitch, pitchf, speaker)[0]
                elif style == "realtime":
                    out = model.infer(phone, lengths, speaker, 0, frames, frames)[0]
                else:
                    out = model.infer(phone, lengths, speaker)[0]
            outputs.append(out)
        diff = (outputs[0] - outputs[1]).abs().max().item()
        emit("LOG", f"patched attention check at {frames} frames: max diff {diff:.2e}")
        if diff > 1e-3:
            raise RuntimeError(f"patched attention differs from stock RVC by {diff} at {frames} frames")


def export_synthesizer(net, use_f0: bool, feature_dim: int, path: Path, torch, nn) -> None:
    style = "realtime" if _accepts_realtime_infer(net, use_f0, feature_dim, torch) else "simple"
    import copy

    stock = copy.deepcopy(net).eval()
    patched = _patch_relative_attention(net, torch)
    emit("LOG", f"patched {patched} relative attention layer(s) for dynamic length")
    if patched:
        _check_patch(stock, net, use_f0, feature_dim, torch, style)
    del stock
    emit("LOG", f"synthesizer call style: {style}")

    class Synthesizer(nn.Module):
        def __init__(self, net, use_f0: bool, style: str):
            super().__init__()
            self.net = net
            self.use_f0 = use_f0
            self.style = style

        def forward(self, phone, phone_lengths, pitch, pitchf, ds):
            frames = phone.size(1)
            if self.use_f0 and self.style == "realtime":
                audio = self.net.infer(phone, phone_lengths, pitch, pitchf, ds, 0, frames, frames)[0]
            elif self.use_f0:
                audio = self.net.infer(phone, phone_lengths, pitch, pitchf, ds)[0]
            elif self.style == "realtime":
                audio = self.net.infer(phone, phone_lengths, ds, 0, frames, frames)[0]
            else:
                audio = self.net.infer(phone, phone_lengths, ds)[0]
            if audio.dim() == 3:
                audio = audio.squeeze(1)
            return audio.float()

    module = Synthesizer(net, use_f0, style).eval()
    frames = 48
    phone = torch.randn(1, frames, feature_dim)
    lengths = torch.tensor([frames], dtype=torch.long)
    pitch = torch.ones(1, frames, dtype=torch.long)
    pitchf = torch.full((1, frames), 150.0)
    speaker = torch.zeros(1, dtype=torch.long)
    emit("LOG", "tracing RVC generator")
    write_onnx(
        torch,
        module,
        (phone, lengths, pitch, pitchf, speaker),
        path,
        ["phone", "phone_lengths", "pitch", "pitchf", "ds"],
        ["audio"],
        {
            "phone": {1: "frames"},
            "pitch": {1: "frames"},
            "pitchf": {1: "frames"},
            "audio": {1: "samples"},
        },
    )


def _accepts_realtime_infer(net, use_f0: bool, feature_dim: int, torch) -> bool:
    frames = 8
    phone = torch.randn(1, frames, feature_dim)
    lengths = torch.tensor([frames], dtype=torch.long)
    pitch = torch.ones(1, frames, dtype=torch.long)
    pitchf = torch.full((1, frames), 150.0)
    speaker = torch.zeros(1, dtype=torch.long)
    try:
        with torch.no_grad():
            if use_f0:
                net.infer(phone, lengths, pitch, pitchf, speaker, 0, frames, frames)
            else:
                net.infer(phone, lengths, speaker, 0, frames, frames)
        return True
    except TypeError:
        return False


def export_feature_bank(root: Path, name: str, destination: Path) -> str:
    import numpy as np

    candidates = [
        root / "logs" / name / "total_fea.npy",
        root / "logs" / name / "total_fea_0.npy",
    ]
    for candidate in candidates:
        if candidate.is_file():
            bank = np.load(candidate).astype("float32")
            if bank.ndim == 1:
                bank = bank.reshape(1, -1)
            np.save(destination, bank)
            return str(destination)
    index_dirs = [root / "assets" / "indices", root / "logs" / name]
    for folder in index_dirs:
        if not folder.is_dir():
            continue
        indexes = sorted(folder.glob("*.index"))
        added = [path for path in indexes if "added" in path.name]
        chosen = (added or indexes)
        if not chosen:
            continue
        try:
            import faiss
        except ImportError:
            return ""
        index = faiss.read_index(str(chosen[-1]))
        if index.ntotal <= 0:
            continue
        bank = index.reconstruct_n(0, index.ntotal).astype("float32")
        np.save(destination, bank)
        return str(destination)
    return ""


def write_onnx(torch, module, args, path: Path, input_names, output_names, dynamic_axes) -> None:
    try:
        torch.onnx.export(
            module,
            args,
            str(path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    except TypeError:
        torch.onnx.export(
            module,
            args,
            str(path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=True,
        )
    emit("LOG", f"wrote {path}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        fail(f"export failed: {exc}")
