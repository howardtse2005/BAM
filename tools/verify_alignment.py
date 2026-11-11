"""Verify that image, mask and synthetic stroke are aligned, per split.

Run this after preprocessing and before every training run. It asserts the
invariants the pipeline depends on and writes a contact sheet for visual review.

    python tools/verify_alignment.py
    python tools/verify_alignment.py --dataset datasets/coco --limit 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config  # noqa: E402
from imgio import imread_gray, imread_rgb  # noqa: E402
from training.paths import SPLITS, pair_split  # noqa: E402
from training.stroke_synth import boundary_from_mask, ensure_binary  # noqa: E402


def _contact_sheet(cells: list[np.ndarray], cols: int = 4, pad: int = 6) -> np.ndarray:
    if not cells:
        return np.zeros((1, 1, 3), np.uint8)
    h, w = cells[0].shape[:2]
    rows = (len(cells) + cols - 1) // cols
    sheet = np.full((rows * (h + pad) + pad, cols * (w + pad) + pad, 3), 24, np.uint8)
    for i, cell in enumerate(cells):
        r, c = divmod(i, cols)
        y = pad + r * (h + pad)
        x = pad + c * (w + pad)
        sheet[y : y + h, x : x + w] = cell
    return sheet


def _cell(rgb: np.ndarray, gt: np.ndarray, stroke: np.ndarray | None, size: int = 220) -> np.ndarray:
    canvas = rgb.copy()
    contour = boundary_from_mask(gt)
    canvas[contour > 0] = (
        0.35 * canvas[contour > 0] + 0.65 * np.array([80, 220, 120])
    ).astype(np.uint8)
    if stroke is not None:
        canvas[stroke > 0] = (
            0.25 * canvas[stroke > 0] + 0.75 * np.array([255, 176, 32])
        ).astype(np.uint8)

    h, w = canvas.shape[:2]
    scale = size / max(h, w)
    resized = cv2.resize(canvas, (max(1, int(w * scale)), max(1, int(h * scale))))
    tile = np.full((size, size, 3), 32, np.uint8)
    y = (size - resized.shape[0]) // 2
    x = (size - resized.shape[1]) // 2
    tile[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return tile


def verify_split(
    root: Path, split: str, short_side: int, limit: int | None, out_dir: Path
) -> dict:
    try:
        files = pair_split(root, split)
    except (FileNotFoundError, ValueError):
        return {"split": split, "skipped": "missing"}

    if limit:
        files = files[:limit]
    if not files:
        return {"split": split, "skipped": "empty"}

    shape_mismatch: list[str] = []
    not_binary: list[str] = []
    oversized: list[str] = []
    missing_edge = 0
    near_ratios: list[float] = []
    cells: list[np.ndarray] = []

    for item in files:
        rgb = imread_rgb(item["im"])
        gt = imread_gray(item["gt"])
        if rgb.shape[:2] != gt.shape[:2]:
            shape_mismatch.append(item["stem"])
            continue

        binary = ensure_binary(gt)
        if not np.array_equal(binary, gt):
            not_binary.append(item["stem"])

        if min(rgb.shape[:2]) > short_side:
            oversized.append(item["stem"])

        stroke = None
        if item["edge"] is not None:
            stroke = imread_gray(item["edge"])
            if stroke.shape[:2] != rgb.shape[:2]:
                shape_mismatch.append(item["stem"])
                continue
        else:
            missing_edge += 1

        if stroke is not None and (stroke > 127).any():
            boundary = boundary_from_mask(binary)
            dist = cv2.distanceTransform(255 - boundary, cv2.DIST_L2, 5)
            near_ratios.append(float((dist[stroke > 127] <= 20.0).mean()))

        if len(cells) < 16:
            cells.append(_cell(rgb, binary, stroke))

    ratios = np.array(near_ratios) if near_ratios else np.array([0.0])
    report = {
        "split": split,
        "images": len(files),
        "shape_mismatch": len(shape_mismatch),
        "non_binary_gt": len(not_binary),
        "not_downscaled": len(oversized),
        "missing_gt_edge": missing_edge,
        "stroke_near_boundary_mean": round(float(ratios.mean()), 4),
        "stroke_near_boundary_p05": round(float(np.percentile(ratios, 5)), 4),
        "examples": {"shape": shape_mismatch[:5], "size": oversized[:5]},
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    sheet = _contact_sheet(cells)
    if sheet.size > 1:
        cv2.imwrite(str(out_dir / f"alignment_{split}.jpg"), sheet[..., ::-1])
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify dataset alignment")
    parser.add_argument("--dataset", default=None, help="dataset path (default: config)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="datasets/_qa/alignment")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    cfg = Config.load(args.config)
    root = Path(args.dataset) if args.dataset else cfg.dataset_path
    if not root.is_dir():
        raise SystemExit(f"dataset folder not found: {root}")

    out_dir = Path(args.out)
    reports = [
        verify_split(root, split, cfg.short_side, args.limit, out_dir) for split in SPLITS
    ]

    hard_failures = 0
    for report in reports:
        if report.get("skipped"):
            print(f"{report['split']:5s} skipped ({report['skipped']})")
            continue
        hard_failures += report["shape_mismatch"] + report["non_binary_gt"]
        print(
            f"{report['split']:5s} {report['images']:6d} images | "
            f"shape {report['shape_mismatch']} | non-binary {report['non_binary_gt']} | "
            f"oversized {report['not_downscaled']} | no-edge {report['missing_gt_edge']} | "
            f"stroke-near-boundary {report['stroke_near_boundary_mean']:.3f}"
        )

    summary = {
        "dataset": str(root),
        "short_side": cfg.short_side,
        "splits": reports,
        "ok": hard_failures == 0,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"ok": summary["ok"], "report": str(out_dir / "report.json")}, indent=2))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
