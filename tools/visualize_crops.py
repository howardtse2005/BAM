"""Visualise training crops.

Produces a contact sheet per sample:
    [ rgb crop ] [ stroke channel ] [ target band over rgb ]

    python tools/visualize_crops.py --count 8
    python tools/visualize_crops.py --index 3 --out datasets/_qa/crops
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config  # noqa: E402
from training.dataset import build_dataset  # noqa: E402


def _to_bgr(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _panel(image: np.ndarray, title: str, size: int = 256) -> np.ndarray:
    canvas = np.full((size + 22, size, 3), 28, np.uint8)
    canvas[22:, :, :] = image
    cv2.putText(
        canvas,
        title,
        (6, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return canvas


def denormalise(image: np.ndarray, mean, std) -> np.ndarray:
    rgb = image[:3].transpose(1, 2, 0).astype(np.float32)
    rgb = rgb * np.asarray(std[:3], np.float32) + np.asarray(mean[:3], np.float32)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Visualise dataset crops")
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--index", type=int, default=None, help="single sample index")
    parser.add_argument("--out", default="datasets/_qa/crops")
    args = parser.parse_args(argv)

    cfg = Config.load(args.config)
    cfg.validate()
    dataset = build_dataset(cfg, args.split, train=True, length=max(args.count, 1))

    indices = [args.index] if args.index is not None else list(range(args.count))
    panels = []

    for index in indices:
        item = dataset[index]
        rgb = denormalise(item["image"].numpy(), cfg.data.rgb_mean, cfg.data.rgb_std)
        stroke = item["image"][3].numpy()
        target = item["target"][0].numpy()

        stroke_vis = np.zeros((*stroke.shape, 3), np.uint8)
        stroke_vis[..., 2] = np.clip(stroke * 255, 0, 255).astype(np.uint8)
        stroke_vis[..., 1] = (np.clip(stroke, 0, 1) * 140).astype(np.uint8)

        overlay = rgb.astype(np.float32).copy()
        heat = np.clip(target, 0, 1)[..., None]
        overlay = overlay * (1 - 0.6 * heat) + np.array([80, 220, 120]) * (0.6 * heat)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        row = np.hstack(
            [
                _panel(_to_bgr(rgb), f"rgb #{index}"),
                _panel(stroke_vis, "stroke channel"),
                _panel(_to_bgr(overlay), "target band"),
            ]
        )
        panels.append(row)

    sheet = np.vstack(panels)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"crops_{args.split}.jpg"
    cv2.imwrite(str(path), sheet)
    print(f"wrote {path} ({sheet.shape[1]}x{sheet.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
