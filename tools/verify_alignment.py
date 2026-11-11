"""Verify that RGB, GT and synthetic stroke are aligned.

Run this before every training run. It asserts the invariants that the whole
data pipeline depends on, and writes a contact sheet for visual inspection.

    python tools/verify_alignment.py
    python tools/verify_alignment.py --limit 200 --out datasets/_qa/alignment
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imgio import imread_gray, imread_rgb  # noqa: E402
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


def _cell(rgb: np.ndarray, gt: np.ndarray, stroke: np.ndarray, size: int = 220) -> np.ndarray:
    """One tile: image + GT contour, and the stroke in amber."""
    canvas = rgb.copy()
    contour = boundary_from_mask(gt)
    canvas[contour > 0] = (0.35 * canvas[contour > 0] + 0.65 * np.array([80, 220, 120])).astype(np.uint8)
    canvas[stroke > 0] = (0.25 * canvas[stroke > 0] + 0.75 * np.array([255, 176, 32])).astype(np.uint8)

    h, w = canvas.shape[:2]
    scale = size / max(h, w)
    resized = cv2.resize(canvas, (max(1, int(w * scale)), max(1, int(h * scale))))
    tile = np.full((size, size, 3), 32, np.uint8)
    y = (size - resized.shape[0]) // 2
    x = (size - resized.shape[1]) // 2
    tile[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return tile


def verify(root: Path, manifest_path: Path, limit: int | None, out_dir: Path) -> dict:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = data.get("samples", [])
    if limit:
        samples = samples[:limit]
    if not samples:
        raise ValueError(f"no samples in {manifest_path}")

    shape_mismatch: list[str] = []
    not_binary: list[str] = []
    size_mismatch: list[str] = []
    missing: list[str] = []
    near_ratios: list[float] = []
    cells: list[np.ndarray] = []

    for sample in samples:
        rgb_path = root / sample["rgb"]
        gt_path = root / sample["gt"]
        edge_path = root / sample.get("gt_edge", "")

        for path in (rgb_path, gt_path, edge_path):
            if not path.is_file():
                missing.append(str(path))
                break
        else:
            rgb = imread_rgb(rgb_path)
            gt = imread_gray(gt_path)
            stroke = imread_gray(edge_path)

            if not (rgb.shape[:2] == gt.shape[:2] == stroke.shape[:2]):
                shape_mismatch.append(sample["id"])
                continue

            binary = ensure_binary(gt)
            if not np.array_equal(binary, gt):
                not_binary.append(sample["id"])

            expected = sample.get("canonical_size")
            if expected and [rgb.shape[1], rgb.shape[0]] != list(expected):
                size_mismatch.append(sample["id"])

            boundary = boundary_from_mask(binary)
            dist = cv2.distanceTransform(255 - boundary, cv2.DIST_L2, 5)
            stroke_px = stroke > 127
            if stroke_px.any():
                near = float((dist[stroke_px] <= 20.0).mean())
                near_ratios.append(near)

            if len(cells) < 16:
                cells.append(_cell(rgb, binary, stroke))

    ratios = np.array(near_ratios) if near_ratios else np.array([0.0])
    report = {
        "samples": len(samples),
        "missing_files": len(missing),
        "shape_mismatch": len(shape_mismatch),
        "non_binary_gt": len(not_binary),
        "canonical_size_mismatch": len(size_mismatch),
        "stroke_near_boundary_mean": round(float(ratios.mean()), 4),
        "stroke_near_boundary_p05": round(float(np.percentile(ratios, 5)), 4),
        "examples": {
            "missing": missing[:5],
            "shape": shape_mismatch[:5],
            "size": size_mismatch[:5],
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    sheet = _contact_sheet(cells)
    if sheet.size > 1:
        cv2.imwrite(str(out_dir / "alignment.jpg"), sheet[..., ::-1])
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    hard_failures = (
        report["missing_files"]
        + report["shape_mismatch"]
        + report["non_binary_gt"]
        + report["canonical_size_mismatch"]
    )
    report["ok"] = hard_failures == 0
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify dataset alignment")
    parser.add_argument("--root", default="datasets")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="datasets/_qa/alignment")
    args = parser.parse_args(argv)

    root = Path(args.root)
    manifest = Path(args.manifest) if args.manifest else root / "manifest.json"
    report = verify(root, manifest, args.limit, Path(args.out))
    print(json.dumps(report, indent=2))
    print(f"contact sheet: {Path(args.out) / 'alignment.jpg'}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
