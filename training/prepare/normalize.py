"""Normalise a flat image/mask folder into the BAM dataset layout.

Input layout (any names, matching stems):

    <source>/images/0001.jpg
    <source>/images/0002.png
    <source>/masks/0001.png
    <source>/masks/0002.png

Output layout:

    datasets/<name>/rgb/000000001.jpg
    datasets/<name>/gt/000000001.png
    datasets/<name>/gt_edge/000000001.png
    datasets/manifest.json

The RGB, GT and synthetic stroke are produced from the same resize call, so they
cannot drift out of alignment. ``tools/verify_alignment.py`` checks this.

    python -m training.prepare.normalize --name duts --source datasets/_raw/duts --tier B
    python -m training.prepare.normalize --name coco --source datasets/_raw/coco --limit 5000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from imgio import (  # noqa: E402
    canonical_size,
    imread_gray,
    imread_rgb,
    imwrite,
    resize,
    to_binary_mask,
)
from training.prepare.dedupe import phash  # noqa: E402
from training.stroke_synth import (  # noqa: E402
    StrokeParams,
    boundary_from_mask,
    ensure_binary,
    synthesize_stroke,
)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
DEFAULT_QUALITY = {
    "openimages": 0.85,
    "coco": 0.90,
    "dis5k": 1.00,
    "duts": 0.60,
    "ecssd": 0.70,
    "hkuis": 0.70,
    "dutomron": 0.65,
}


# --------------------------------------------------------------------------- #
# Pair discovery
# --------------------------------------------------------------------------- #


def discover_pairs(source: Path) -> list[tuple[Path, Path]]:
    """Match images and masks by stem, case-insensitively.

    If ``pairs.json`` exists (written by the COCO converter) it takes precedence,
    so multi-gigabyte image trees never have to be copied or symlinked.
    """
    pairs_json = source / "pairs.json"
    if pairs_json.is_file():
        raw = json.loads(pairs_json.read_text(encoding="utf-8"))
        pairs = []
        for item in raw:
            image, mask = Path(item[0]), Path(item[1])
            if not image.is_absolute():
                image = source / image
            if not mask.is_absolute():
                mask = source / mask
            if image.is_file() and mask.is_file():
                pairs.append((image, mask))
        if not pairs:
            raise ValueError(f"pairs.json in {source} listed no existing files")
        return pairs

    images_dir = source / "images"
    masks_dir = source / "masks"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"missing images directory: {images_dir}")
    if not masks_dir.is_dir():
        raise FileNotFoundError(f"missing masks directory: {masks_dir}")

    mask_by_stem: dict[str, Path] = {}
    for path in sorted(masks_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            mask_by_stem.setdefault(path.stem.lower(), path)

    pairs: list[tuple[Path, Path]] = []
    missing = 0
    for path in sorted(images_dir.iterdir()):
        if not (path.is_file() and path.suffix.lower() in IMAGE_EXTS):
            continue
        mask = mask_by_stem.get(path.stem.lower())
        if mask is None:
            missing += 1
            continue
        pairs.append((path, mask))

    if missing:
        print(f"  {missing} image(s) had no matching mask and were skipped")
    if not pairs:
        raise ValueError(f"no image/mask pairs found in {source}")
    return pairs


# --------------------------------------------------------------------------- #
# Quality checks
# --------------------------------------------------------------------------- #


def boundary_sharpness(rgb: np.ndarray, gt: np.ndarray) -> float:
    """Mean Sobel magnitude along the GT boundary, normalised to [0, 1]."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    edge = boundary_from_mask(gt) > 0
    if not edge.any():
        return 0.0
    return float(np.clip(mag[edge].mean() / 1020.0, 0.0, 1.0))


def validate_pair(rgb: np.ndarray, gt: np.ndarray) -> tuple[bool, str]:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return False, "rgb is not 3-channel"
    if rgb.shape[:2] != gt.shape[:2]:
        return False, "shape mismatch"
    area = float((gt > 0).mean())
    if area < 0.002:
        return False, f"foreground too small ({area:.4f})"
    if area > 0.98:
        return False, f"foreground too large ({area:.4f})"
    if int((boundary_from_mask(gt) > 0).sum()) < 32:
        return False, "boundary too short"
    return True, "ok"


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #


def normalize_dataset(
    name: str,
    source: Path,
    root: Path,
    short_side: int = 640,
    tier: str = "A",
    quality_weight: float | None = None,
    limit: int | None = None,
    stroke_seed: int = 0,
    min_sharpness: float = 0.0,
    overwrite: bool = True,
    dry_run: bool = False,
) -> dict:
    pairs = discover_pairs(source)
    if limit:
        pairs = pairs[:limit]

    weight = quality_weight if quality_weight is not None else DEFAULT_QUALITY.get(name, 0.80)
    params = StrokeParams()
    out_dir = root / name
    if not dry_run and overwrite:
        for sub in ("rgb", "gt", "gt_edge"):
            for old in (out_dir / sub).glob("*"):
                if old.is_file():
                    old.unlink()

    samples: list[dict] = []
    rejected: list[dict] = []
    accepted = 0

    for index, (image_path, mask_path) in enumerate(pairs):
        try:
            rgb = imread_rgb(image_path)
            gt = to_binary_mask(imread_gray(mask_path))
        except Exception as exc:  # unreadable file
            rejected.append({"image": str(image_path), "reason": f"read error: {exc}"})
            continue

        if gt.shape[:2] != rgb.shape[:2]:
            gt = cv2.resize(gt, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

        height, width = rgb.shape[:2]
        target_w, target_h, scale = canonical_size(height, width, short_side)
        if (target_w, target_h) != (width, height):
            rgb = resize(rgb, target_w, target_h, is_mask=False)
            gt = resize(gt, target_w, target_h, is_mask=True)

        gt = ensure_binary(gt)
        ok, reason = validate_pair(rgb, gt)
        if not ok:
            rejected.append({"image": str(image_path), "reason": reason})
            continue

        sharpness = boundary_sharpness(rgb, gt)
        if sharpness < min_sharpness:
            rejected.append(
                {"image": str(image_path), "reason": f"blurry boundary ({sharpness:.3f})"}
            )
            continue

        stem = f"{index:09d}"
        if not dry_run:
            imwrite(out_dir / "rgb" / f"{stem}.jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            imwrite(out_dir / "gt" / f"{stem}.png", gt)
            stroke = synthesize_stroke(gt, rgb=rgb, params=params, rng=stroke_seed)
            imwrite(out_dir / "gt_edge" / f"{stem}.png", stroke)

        samples.append(
            {
                "id": f"{name}/{stem}",
                "dataset": name,
                "tier": tier,
                "rgb": f"{name}/rgb/{stem}.jpg",
                "gt": f"{name}/gt/{stem}.png",
                "gt_edge": f"{name}/gt_edge/{stem}.png",
                "original_size": [int(width), int(height)],
                "canonical_size": [int(target_w), int(target_h)],
                "scale": round(float(scale), 6),
                "split": "train",
                "instance_area_ratio": round(float((gt > 0).mean()), 6),
                "boundary_sharpness": round(sharpness, 6),
                "quality_weight": float(weight),
                "license": "unknown",
                "stroke_seed": int(stroke_seed),
                "phash": str(phash(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))),
            }
        )
        accepted += 1

        if accepted % 500 == 0:
            print(f"  {name}: {accepted} accepted, {len(rejected)} rejected")

    stats = {
        "dataset": name,
        "accepted": accepted,
        "rejected": len(rejected),
        "tier": tier,
        "short_side": short_side,
        "rejections": rejected[:50],
    }

    if not dry_run:
        update_manifest(root, name, samples)
        (out_dir / "meta.json").write_text(
            json.dumps(stats, indent=2), encoding="utf-8"
        )

    return stats


def update_manifest(root: Path, dataset: str, samples: list[dict]) -> None:
    """Replace this dataset's entries in the manifest, keeping the others."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = {"version": 1, "samples": []}

    data["samples"] = [s for s in data["samples"] if s.get("dataset") != dataset]
    data["samples"].extend(samples)
    data["samples"].sort(key=lambda s: s["id"])
    data["datasets"] = sorted({s["dataset"] for s in data["samples"]})
    data["counts"] = {
        name: sum(1 for s in data["samples"] if s["dataset"] == name)
        for name in data["datasets"]
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Normalise a dataset into BAM layout")
    parser.add_argument("--name", required=True, help="dataset name, e.g. duts")
    parser.add_argument("--source", required=True, help="folder with images/ and masks/")
    parser.add_argument("--root", default="datasets")
    parser.add_argument("--short-side", type=int, default=640)
    parser.add_argument("--tier", default="A", choices=["A", "B"])
    parser.add_argument("--quality-weight", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-sharpness", type=float, default=0.0)
    parser.add_argument("--stroke-seed", type=int, default=0)
    parser.add_argument("--no-overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    stats = normalize_dataset(
        name=args.name,
        source=Path(args.source),
        root=Path(args.root),
        short_side=args.short_side,
        tier=args.tier,
        quality_weight=args.quality_weight,
        limit=args.limit,
        stroke_seed=args.stroke_seed,
        min_sharpness=args.min_sharpness,
        overwrite=not args.no_overwrite,
        dry_run=args.dry_run,
    )
    print(json.dumps(stats, indent=2))
    return 0 if stats["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
