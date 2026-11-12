"""Training entry point.

    python -m training.train                      # defaults from config.py
    python -m training.train --set epochs=10 --set batch_size=16
    python -m training.train --dry-run            # synthetic batch, no data needed

Plain loop: AdamW, cosine schedule with warmup, AMP on CUDA, EMA weights, early
stopping on validation boundary F@2px. The dataset is read straight from
``datasets/<name>/{train,val,test}``; there is no manifest to maintain.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if __package__ in (None, ""):  # allow `python training/train.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config, add_config_args  # noqa: E402
from training.dataset import build_dataset, dataset_fingerprint, scan_split  # noqa: E402
from training.losses import BAMLoss  # noqa: E402
from training.metrics import MetricAccumulator, evaluate_masks  # noqa: E402
from training.unet import build_model  # noqa: E402


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("train_device=cuda but CUDA is not available")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


class ModelEMA:
    """Exponential moving average of model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for param in self.module.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        decay = self.decay
        ema_state = self.module.state_dict()
        for key, value in model.state_dict().items():
            target = ema_state[key]
            if target.dtype.is_floating_point:
                target.mul_(decay).add_(value.detach().to(target.dtype), alpha=1.0 - decay)
            else:
                target.copy_(value)


def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        return torch.load(path, map_location=map_location, weights_only=False)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    cfg: Config,
    dataset_hash: str,
    epoch: int,
    metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "config": cfg.to_dict(),
        "model_config": model.config_dict() if hasattr(model, "config_dict") else {},
        "dataset": cfg.dataset,
        "dataset_hash": dataset_hash,
        "data_tier": cfg.tier,
        "epoch": int(epoch),
        "metrics": {k: float(v) for k, v in metrics.items()},
        "git_commit": git_commit(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "format": "bam-2",
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    acc = MetricAccumulator()
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        logits = model(images)
        probs = torch.sigmoid(logits).float().cpu().numpy()
        gts = batch["gt"].cpu().numpy()
        for i in range(probs.shape[0]):
            pred = (probs[i, 0] > 0.5).astype(np.uint8) * 255
            gt = (gts[i, 0] > 0.5).astype(np.uint8) * 255
            acc.add(evaluate_masks(pred, gt))
    return acc.mean()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the BAM UNet")
    add_config_args(parser)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run one synthetic batch through model + loss and exit",
    )
    return parser.parse_args(argv)


def dry_run(cfg: Config, device: torch.device) -> int:
    model = build_model(cfg).to(device)
    criterion = BAMLoss(pos_weight=cfg.pos_weight)
    images = torch.randn(2, 4, cfg.crop_size, cfg.crop_size, device=device)
    target = torch.zeros(2, 1, cfg.crop_size, cfg.crop_size, device=device)
    target[:, :, 100:104, 60:200] = 1.0
    weight = torch.ones_like(target)
    logits = model(images)
    loss, parts = criterion(logits, target, weight)
    loss.backward()
    print(json.dumps({
        "status": "ok",
        "device": str(device),
        "params": model.count_parameters(),
        "logits": list(logits.shape),
        "loss": round(parts["loss"], 4),
        "grad_norm": round(
            float(
                torch.sqrt(
                    sum((p.grad**2).sum() for p in model.parameters() if p.grad is not None)
                )
            ),
            4,
        ),
    }, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = Config.load(args.overrides)
    if args.epochs is not None:
        cfg.epochs = int(args.epochs)
    if args.batch_size is not None:
        cfg.batch_size = int(args.batch_size)
    cfg.validate()

    seed_everything(cfg.seed)
    device = resolve_device(cfg.train_device)

    if device.type == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 4))

    if args.dry_run:
        return dry_run(cfg, device)

    train_ds = build_dataset(cfg, "train", train=True, length=cfg.crops_per_epoch)
    val_ds = build_dataset(cfg, "val", train=False, length=cfg.val_crops)

    files_by_split = {"train": train_ds.files, "val": val_ds.files}
    try:
        files_by_split["test"] = scan_split(cfg.dataset_path, "test")
    except (FileNotFoundError, ValueError):
        pass
    dataset_hash = dataset_fingerprint(files_by_split)

    workers = int(cfg.num_workers)
    loader_kwargs = dict(
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=True, **loader_kwargs
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(1, cfg.batch_size // 2), shuffle=False, **loader_kwargs
    )

    model = build_model(cfg).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    ema = ModelEMA(model, decay=cfg.ema_decay)

    criterion = BAMLoss(pos_weight=cfg.pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = steps_per_epoch * cfg.warmup_epochs

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_amp = bool(cfg.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 0
    best_bf = -1.0
    patience = 0
    if args.resume:
        ckpt = load_checkpoint(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        ema = ModelEMA(model, decay=cfg.ema_decay)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_bf = float(ckpt.get("metrics", {}).get("val_bf", -1.0))
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    run_dir = cfg.runs_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(run_dir / "config.yaml")
    (run_dir / "git_commit.txt").write_text(git_commit(), encoding="utf-8")
    (run_dir / "dataset_hash.txt").write_text(dataset_hash, encoding="utf-8")

    log_file = (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8")
    writer = csv.writer(log_file)
    writer.writerow(
        ["epoch", "lr", "loss", "bce", "dice", "l1", "val_bf", "val_iou", "val_biou", "seconds"]
    )

    print(
        f"dataset={cfg.dataset} device={device} params={model.count_parameters()} "
        f"train={len(train_ds)} val={len(val_ds)} steps/epoch={steps_per_epoch}"
    )

    epoch = start_epoch
    for epoch in range(start_epoch, cfg.epochs):
        train_ds.set_epoch(epoch)
        model.train()
        running = {"loss": 0.0, "bce": 0.0, "dice": 0.0, "l1": 0.0}
        epoch_start = time.time()

        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            weight = batch["weight"].to(device, non_blocking=True)
            if device.type == "cuda":
                images = images.to(memory_format=torch.channels_last)

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                logits = model(images)
                loss, parts = criterion(logits, target, weight)

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                if cfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()

            scheduler.step()
            ema.update(model)

            for key in running:
                running[key] += parts[key]
            if step % 50 == 0:
                print(
                    f"epoch {epoch} step {step}/{steps_per_epoch} "
                    f"loss {parts['loss']:.4f} bce {parts['bce']:.4f} "
                    f"dice {parts['dice']:.4f} lr {scheduler.get_last_lr()[0]:.2e}",
                    flush=True,
                )

        n_steps = max(1, len(train_loader))
        train_metrics = {k: v / n_steps for k, v in running.items()}
        val_metrics = validate(ema.module, val_loader, device)

        elapsed = time.time() - epoch_start
        writer.writerow(
            [
                epoch,
                scheduler.get_last_lr()[0],
                train_metrics["loss"],
                train_metrics["bce"],
                train_metrics["dice"],
                train_metrics["l1"],
                val_metrics.get("bf", 0.0),
                val_metrics.get("iou", 0.0),
                val_metrics.get("boundary_iou", 0.0),
                round(elapsed, 1),
            ]
        )
        log_file.flush()

        print(
            f"[epoch {epoch}] loss {train_metrics['loss']:.4f} "
            f"val_bf {val_metrics.get('bf', 0):.4f} "
            f"val_iou {val_metrics.get('iou', 0):.4f} "
            f"val_biou {val_metrics.get('boundary_iou', 0):.4f} "
            f"({elapsed:.0f}s)",
            flush=True,
        )

        score = float(val_metrics.get("bf", 0.0))
        save_checkpoint(run_dir / "last.pth", ema.module, cfg, dataset_hash, epoch, val_metrics)
        if score > best_bf:
            best_bf = score
            patience = 0
            save_checkpoint(run_dir / "best.pth", ema.module, cfg, dataset_hash, epoch, val_metrics)
            save_checkpoint(cfg.model_file, ema.module, cfg, dataset_hash, epoch, val_metrics)
            print(f"  new best val_bf={best_bf:.4f} -> {cfg.model_file}")
        else:
            patience += 1
            if patience >= cfg.early_stop:
                print(f"early stop after {patience} epochs without improvement")
                break

    log_file.close()
    summary = {
        "best_val_bf": best_bf,
        "run_dir": str(run_dir),
        "epochs_run": epoch + 1 if cfg.epochs else 0,
        "dataset": cfg.dataset,
        "data_tier": cfg.tier,
        "dataset_hash": dataset_hash,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
