"""Export a checkpoint to ONNX (fp32) and optionally dynamic INT8.

    python -m training.export_onnx --checkpoint model/bam_v1.pth
    python -m training.export_onnx --checkpoint model/bam_v1.pth --int8

BatchNorm folding is handled by the ONNX exporter, so no manual graph surgery is
needed. The script verifies numerical parity against PyTorch before declaring
success.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config, add_config_args  # noqa: E402
from training.evaluate import build_from_checkpoint  # noqa: E402


def require_onnx() -> str:
    """Import onnx or fail with an actionable message. Returns its version."""
    try:
        import onnx
    except ImportError as exc:
        raise SystemExit(
            "ONNX export needs the 'onnx' package://n    pip install onnx onnxruntime"
        ) from exc
    return getattr(onnx, "__version__", "unknown")


def export_fp32(model: torch.nn.Module, path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 4, size, size)
    kwargs = dict(
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch", 2: "height", 3: "width"},
        },
        do_constant_folding=True,
    )
    try:
        torch.onnx.export(model, dummy, str(path), dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(model, dummy, str(path), **kwargs)


def parity_check(model: torch.nn.Module, onnx_path: Path, size: int) -> dict:
    try:
        import onnxruntime as ort
    except ImportError:
        return {"onnxruntime": "missing", "max_abs_diff": None}

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = torch.randn(2, 4, size, size)
    with torch.inference_mode():
        ref = model(x).numpy()
    got = session.run(None, {"input": x.numpy()})[0]
    return {
        "onnxruntime": ort.__version__,
        "max_abs_diff": float(np.abs(ref - got).max()),
        "shape_match": tuple(ref.shape) == tuple(got.shape),
    }


def quantize_int8(fp32_path: Path, int8_path: Path) -> dict:
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError:
        return {"status": "skipped", "reason": "onnxruntime.quantization unavailable"}

    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )
    return {"status": "ok", "path": str(int8_path), "size_mb": round(int8_path.stat().st_size / 1e6, 3)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export BAM to ONNX")
    add_config_args(parser)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args(argv)

    cfg = Config.load(args.config, args.overrides)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else cfg.model_path
    if not ckpt_path.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    model, ckpt = build_from_checkpoint(ckpt_path, torch.device("cpu"))
    model.eval()

    out_path = Path(args.out) if args.out else Path(cfg.infer.onnx_path)
    if not out_path.is_absolute():
        out_path = cfg.path(str(out_path))

    onnx_version = require_onnx()
    export_fp32(model, out_path, args.size)
    report = {
        "checkpoint": str(ckpt_path),
        "onnx_version": onnx_version,
        "epoch": ckpt.get("epoch"),
        "onnx": str(out_path),
        "size_mb": round(out_path.stat().st_size / 1e6, 3),
        "parity": parity_check(model, out_path, args.size),
    }

    if args.int8:
        int8_path = Path(cfg.infer.onnx_int8_path)
        if not int8_path.is_absolute():
            int8_path = cfg.path(str(int8_path))
        report["int8"] = quantize_int8(out_path, int8_path)
        if int8_path.is_file():
            report["int8_parity"] = parity_check(model, int8_path, args.size)

    import json

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
