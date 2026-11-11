"""Generate a synthetic demo dataset.

This is not training data. It exists so the whole pipeline -- normalise, manifest,
split, dataset, model, server, UI -- can be exercised end to end before a real
dataset is downloaded.

    python tools/make_demo_subset.py --count 24
    python -m training.prepare.normalize --name demo --source datasets/_raw/demo --tier A
    python -m training.prepare.split
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imgio import imwrite  # noqa: E402


def _background(rng: np.random.Generator, height: int, width: int) -> np.ndarray:
    kind = int(rng.integers(0, 3))
    base = rng.integers(40, 210, size=3).astype(np.float32)

    if kind == 0:
        y = np.linspace(0, 1, height, dtype=np.float32)[:, None, None]
        delta = rng.uniform(-45, 45, size=3).astype(np.float32)
        img = base[None, None, :] + y * delta[None, None, :]
        img = np.repeat(img, width, axis=1)
    elif kind == 1:
        img = np.ones((height, width, 3), np.float32) * base
        period = int(rng.integers(24, 64))
        stripes = ((np.arange(width) // period) % 2).astype(np.float32)
        img *= (0.85 + 0.3 * stripes)[None, :, None]
    else:
        img = np.ones((height, width, 3), np.float32) * base
        img += rng.normal(0, 12, img.shape).astype(np.float32)

    img += rng.normal(0, 4, img.shape).astype(np.float32)
    return np.clip(img, 0, 255)


def _shape_mask(
    rng: np.random.Generator, height: int, width: int
) -> tuple[np.ndarray, tuple[int, int, int]]:
    mask = np.zeros((height, width), np.uint8)
    cx = int(rng.integers(width // 4, 3 * width // 4))
    cy = int(rng.integers(height // 4, 3 * height // 4))
    rx = int(rng.integers(width // 8, width // 3))
    ry = int(rng.integers(height // 8, height // 3))
    angle = float(rng.uniform(0, 180))
    color = tuple(int(c) for c in rng.integers(20, 235, size=3))

    kind = int(rng.integers(0, 4))
    if kind == 0:
        cv2.ellipse(mask, (cx, cy), (rx, ry), angle, 0, 360, 255, -1)
    elif kind == 1:
        rect = ((cx, cy), (rx * 2, ry * 2), angle)
        box = cv2.boxPoints(rect).astype(np.int32)
        cv2.fillPoly(mask, [box], 255)
    elif kind == 2:
        n = int(rng.integers(5, 9))
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False) + rng.uniform(0, 1)
        radius = np.linspace(0.6, 1.0, n) * rng.uniform(0.7, 1.0)
        pts = np.stack(
            [cx + rx * radius * np.cos(theta), cy + ry * radius * np.sin(theta)], axis=1
        ).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)
    else:
        pts = np.array(
            [
                [cx - rx, cy + ry],
                [cx - rx // 2, cy - ry],
                [cx, cy + ry // 3],
                [cx + rx // 2, cy - ry],
                [cx + rx, cy + ry],
            ],
            np.int32,
        )
        cv2.fillPoly(mask, [pts], 255)

    return mask, color


def make_scene(
    rng: np.random.Generator, height: int, width: int
) -> tuple[np.ndarray, np.ndarray]:
    rgb = _background(rng, height, width)
    mask, color = _shape_mask(rng, height, width)

    texture = rng.normal(0, 8, (height, width, 3)).astype(np.float32)
    obj = np.array(color, np.float32)[None, None, :] + texture
    alpha = cv2.GaussianBlur((mask > 0).astype(np.float32), (0, 0), 0.8)[..., None]
    rgb = rgb * (1 - alpha) + obj * alpha

    # A soft shadow makes the boundary a genuine appearance discontinuity.
    shadow = cv2.GaussianBlur((mask > 0).astype(np.float32), (0, 0), 6) * 0.25
    rgb = rgb * (1 - shadow[..., None])

    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb, mask


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a synthetic demo dataset")
    parser.add_argument("--out", default="datasets/_raw/demo")
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    root = Path(args.out)
    images = root / "images"
    masks = root / "masks"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    for i in range(args.count):
        rgb, mask = make_scene(rng, args.height, args.width)
        name = f"{i:04d}"
        imwrite(images / f"{name}.jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        imwrite(masks / f"{name}.png", mask)

    print(f"wrote {args.count} synthetic pairs to {root}")
    print(
        "next:\n"
        f"  python -m training.prepare.normalize --name demo --source {root} --tier A\n"
        "  python -m training.prepare.split"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
