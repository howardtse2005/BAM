"""Evaluate a checkpoint on a dataset split.

    python -m training.evaluate --split test --checkpoint model/bam_v1.pth
    python -m training.evaluate --split val  --checkpoint runs/.../best.pth --limit 200

Writes a JSON report and prints a Markdown table.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config, add_config_args  # noqa: E402
from training.dataset import build_dataset  # noqa: E402
from training.metrics import MetricAccumulator, evaluate_masks  # noqa: E402
from training.train import load_checkpoint, resolve_device  # noqa: E402
from training.unet import BAMUNet  # noqa: E402


def build_from_checkpoint(path: str | Path, device: torch.device) -> tuple[BAMUNet, dict]:
    ckpt = load_checkpoint(path, map_location=device)
    model_cfg = ckpt.get("model_config") or ckpt.get("config", {}).get("model", {})
    model = BAMUNet(
        in_channels=int(model_cfg.get("in_channels", 4)),
        out_channels=int(model_cfg.get("out_channels", 1)),
        width=int(model_cfg.get("width", 20)),
        depth=int(model_cfg.get("depth", 4)),
        block=str(model_cfg.get("block", "ds")),
        bilinear=bool(model_cfg.get("bilinear", True)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate BAM on a dataset split")
    add_config_args(parser)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--limit", type=int, default=None, help="max images to evaluate")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    cfg = Config.load(args.overrides)
    cfg.validate()
    device = resolve_device(cfg.infer_device)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else cfg.model_file
    if not ckpt_path.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")
    model, ckpt = build_from_checkpoint(ckpt_path, device)

    dataset = build_dataset(cfg, args.split, train=False, length=args.limit)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    acc = MetricAccumulator()
    per_area: dict[str, MetricAccumulator] = {}

    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).float().cpu().numpy()
            gts = batch["gt"].cpu().numpy()
            for i in range(probs.shape[0]):
                pred = (probs[i, 0] > 0.5).astype("uint8") * 255
                gt = (gts[i, 0] > 0.5).astype("uint8") * 255
                metrics = evaluate_masks(pred, gt)
                acc.add(metrics)

                area = float((gt > 0).mean())
                bucket = "<1%" if area < 0.01 else "1-10%" if area < 0.10 else ">10%"
                per_area.setdefault(bucket, MetricAccumulator()).add(metrics)

    summary = {
        "dataset": cfg.dataset,
        "split": args.split,
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": ckpt.get("epoch"),
        "data_tier": ckpt.get("data_tier"),
        "dataset_hash": ckpt.get("dataset_hash"),
        "images": len(acc),
        "metrics": acc.mean(),
        "by_foreground_area": {k: v.mean() for k, v in per_area.items()},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    out_path = Path(args.out) if args.out else cfg.runs_root / f"eval_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{cfg.dataset} / {args.split}: {len(acc)} images")
    print("| metric | value |")
    print("| --- | --- |")
    for key, value in summary["metrics"].items():
        print(f"| {key} | {value:.4f} |")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
