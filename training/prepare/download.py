"""Dataset download and import helpers.

Only COCO is automated, because it is the only large instance-segmentation corpus
with a stable, documented URL layout and a licence that permits commercial use of
the annotations.

Everything else (Open Images, DIS5K, DUTS, ...) is imported from a folder you
have downloaded yourself, laid out as::

    <dataset>/train/im  <dataset>/train/gt
    <dataset>/val/im    <dataset>/val/gt
    <dataset>/test/im   <dataset>/test/gt

That is deliberate: several of those datasets have no commercial licence, and a
one-command downloader would make it too easy to contaminate a shippable
checkpoint.

    python -m training.prepare.download --check
    python -m training.prepare.download --coco --limit 5000

COCO images are copied byte-for-byte; only the instance masks are rendered.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from imgio import imwrite  # noqa: E402

COCO_BASE = "http://images.cocodataset.org"
COCO_URLS = {
    "annotations": f"{COCO_BASE}/annotations/annotations_trainval2017.zip",
    "train2017": f"{COCO_BASE}/zips/train2017.zip",
    "val2017": f"{COCO_BASE}/zips/val2017.zip",
}

LICENSE_MATRIX = """
Dataset license matrix
======================
SHIPPABLE (tier A)
  Open Images V7   images CC BY 2.0 (per-image, verify) | masks CC BY 4.0 (Google)
                   -> commercial use allowed with attribution
  COCO 2017        annotations CC BY 4.0 | images under Flickr terms (check per image)
                   -> annotations are usable; images must be verified individually

RESEARCH ONLY (tier B) -- never ship weights trained on these
  DIS5K            no explicit licence  -> all rights reserved by default
  DUTS             "All rights reserved by the original authors"
  ECSSD / HKU-IS / DUT-OMRON   research datasets, no commercial grant
  SA-1B            SA-1B Dataset Research License
  Cityscapes / Mapillary / BDD100K / Objects365 / xView   non-commercial or research

Attribution is a legal obligation, not a courtesy. See README.md.
""".strip()


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #


def download(url: str, dest: Path, chunk_size: int = 1 << 20) -> Path:
    if dest.is_file() and dest.stat().st_size > 0:
        print(f"  exists, skipping: {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    print(f"  downloading {url}")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as handle:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            handle.write(chunk)
            done += len(chunk)
            if total:
                pct = 100.0 * done / total
                print(f"\r    {done / 1e6:8.1f} / {total / 1e6:.1f} MB ({pct:5.1f}%)", end="")
        print()
    tmp.replace(dest)
    return dest


def unzip(archive: Path, dest: Path) -> None:
    print(f"  extracting {archive.name}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)


# --------------------------------------------------------------------------- #
# COCO
# --------------------------------------------------------------------------- #


def _polygon_mask(segmentation, height: int, width: int) -> np.ndarray | None:
    """Render a COCO polygon segmentation. RLE (crowd) is skipped."""
    if isinstance(segmentation, dict):
        return None
    mask = np.zeros((height, width), np.uint8)
    drew = False
    for polygon in segmentation:
        pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] < 3:
            continue
        cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 255)
        drew = True
    return mask if drew else None


def _best_instances(ann_path: Path) -> tuple[dict, dict]:
    """Largest non-crowd polygon instance per image."""
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    images = {img["id"]: img for img in data["images"]}

    best: dict[int, dict] = {}
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        if not isinstance(ann.get("segmentation"), list):
            continue
        if ann.get("area", 0) <= 0:
            continue
        current = best.get(ann["image_id"])
        if current is None or ann["area"] > current["area"]:
            best[ann["image_id"]] = ann
    return images, best


def convert_coco_split(
    coco_root: Path,
    source_split: str,
    out_dir: Path,
    limit: int | None = None,
    val_fraction: float = 0.5,
) -> dict:
    """Convert one COCO split into ``im/`` and ``gt/`` folders.

    ``val2017`` is divided into ``val`` and ``test`` so the dataset ends up with
    all three splits. ``train2017`` becomes ``train``.
    """
    ann_path = coco_root / "annotations" / f"instances_{source_split}.json"
    image_dir = coco_root / source_split
    if not ann_path.is_file():
        raise FileNotFoundError(f"missing annotations: {ann_path}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"missing images: {image_dir}")

    print(f"  reading {ann_path.name}")
    images, best = _best_instances(ann_path)
    image_ids = sorted(best)

    if source_split == "val2017":
        cut = int(round(len(image_ids) * val_fraction))
        targets = {"val": image_ids[:cut], "test": image_ids[cut:]}
    else:
        targets = {"train": image_ids}

    stats: dict[str, dict] = {}
    for split, ids in targets.items():
        if limit is not None:
            ids = ids[:limit]
        im_dir = out_dir / split / "im"
        gt_dir = out_dir / split / "gt"
        im_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)

        written = 0
        skipped = 0
        for image_id in ids:
            meta = images.get(image_id)
            ann = best.get(image_id)
            if meta is None or ann is None:
                skipped += 1
                continue
            source = image_dir / meta["file_name"]
            if not source.is_file():
                skipped += 1
                continue

            mask_path = gt_dir / f"{image_id:012d}.png"
            if not mask_path.is_file():
                mask = _polygon_mask(ann["segmentation"], meta["height"], meta["width"])
                if mask is None:
                    skipped += 1
                    continue
                imwrite(mask_path, mask)

            target = im_dir / f"{image_id:012d}{source.suffix.lower()}"
            if not target.is_file():
                shutil.copy2(source, target)

            written += 1
            if written % 1000 == 0:
                print(f"    {split}: {written}")

        stats[split] = {"images": written, "skipped": skipped}

    return {"source": source_split, "splits": stats, "out": str(out_dir)}


def prepare_coco(
    coco_root: Path,
    out_dir: Path,
    limit: int | None,
    download_images: bool,
) -> dict:
    if download_images:
        for split in ("train2017", "val2017"):
            archive = coco_root / f"{split}.zip"
            download(COCO_URLS[split], archive)
            unzip(archive, coco_root)

    ann_archive = coco_root / "annotations_trainval2017.zip"
    if not (coco_root / "annotations" / "instances_val2017.json").is_file():
        download(COCO_URLS["annotations"], ann_archive)
        unzip(ann_archive, coco_root)

    results = [
        convert_coco_split(coco_root, split, out_dir, limit=limit)
        for split in ("train2017", "val2017")
    ]
    return {"converted": results, "out": str(out_dir)}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download or import datasets")
    parser.add_argument("--check", action="store_true", help="print the licence matrix")
    parser.add_argument("--coco", action="store_true", help="prepare COCO instances")
    parser.add_argument("--coco-root", default="datasets/_raw/coco")
    parser.add_argument("--out", default="datasets/_raw/coco")
    parser.add_argument("--limit", type=int, default=None, help="max images per split")
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="assume the archives are already extracted",
    )
    args = parser.parse_args(argv)

    if args.check or not args.coco:
        print(LICENSE_MATRIX)
        if not args.coco:
            return 0

    stats = prepare_coco(
        Path(args.coco_root),
        Path(args.out),
        limit=args.limit,
        download_images=not args.no_download,
    )
    print(json.dumps(stats, indent=2))
    print(f"\nnext:\n  python -m training.prepare.preprocess --dataset {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
