"""Preprocess a dataset into the layout the trainer expects.

Source layout (any size, ``gt_edge`` optional)::

    <source>/train/im/  <source>/train/gt/
    <source>/val/im/    <source>/val/gt/
    <source>/test/im/   <source>/test/gt/

Output layout (canonical size + synthetic strokes)::

    <out>/train/im/  <out>/train/gt/  <out>/train/gt_edge/
    ...

What it does per split:

1. Pairs ``im/`` and ``gt/`` by filename stem.
2. Rejects samples whose mask is empty, tiny, or geometrically broken.
3. Downscales image and mask to a canonical short side (never upscales). Both are
   resized from the same dimensions, so they cannot drift apart.
4. Writes ``gt_edge/``: one synthetic user stroke per mask, seed 0, so the
   validation set is reproducible.
5. Writes ``meta.json`` with counts and rejection reasons.

Cropping to 256 happens at training time, not here: a 640x640 image would yield
dozens of overlapping crops and the sampler deliberately varies scale and
position. Materialising them would multiply disk usage for no benefit.

    python -m training.prepare.preprocess --dataset datasets/DIS5K
    # -> datasets/DIS5K_prepared

    python -m training.prepare.preprocess --dataset datasets/DIS5K --inplace
    # -> rewrites DIS5K itself; originals are lost unless you have a copy

By default the output is a sibling ``<name>_prepared`` folder, so a 4K source is
never modified by accident. ``--inplace`` opts into the destructive path.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import Config, add_config_args  # noqa: E402
from imgio import (  # noqa: E402
    canonical_size,
    imread_gray,
    imread_rgb,
    imwrite,
    resize,
    to_binary_mask,
)
from training.paths import SPLITS, pair_split  # noqa: E402
from training.stroke_synth import (  # noqa: E402
    StrokeParams,
    boundary_from_mask,
    ensure_binary,
    synthesize_stroke,
)


def validate_pair(rgb: np.ndarray, gt: np.ndarray) -> tuple[bool, str]:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return False, "image is not 3-channel"
    if rgb.shape[:2] != gt.shape[:2]:
        return False, "image and mask shapes differ"
    area = float((gt > 0).mean())
    if area < 0.002:
        return False, f"foreground too small ({area:.4f})"
    if area > 0.98:
        return False, f"foreground too large ({area:.4f})"
    if int((boundary_from_mask(gt) > 0).sum()) < 32:
        return False, "boundary too short"
    return True, "ok"


def backup_split(root: Path, split: str) -> int:
    """Copy im/ and gt/ to _original/ before an in-place run."""
    copied = 0
    for sub in ("im", "gt"):
        source = root / split / sub
        target = root / split / "_original" / sub
        if not source.is_dir() or target.is_dir():
            continue
        target.mkdir(parents=True, exist_ok=True)
        for path in source.iterdir():
            if path.is_file():
                shutil.copy2(path, target / path.name)
                copied += 1
    return copied


def preprocess_split(
    source: Path,
    dest: Path,
    split: str,
    short_side: int,
    force: bool = False,
    dry_run: bool = False,
    limit: int | None = None,
    make_edge: bool = True,
    backup: bool = False,
) -> dict:
    if not (source / split / "im").is_dir():
        return {"split": split, "skipped": "no im/ folder"}

    pairs = pair_split(source, split)
    if limit:
        pairs = pairs[:limit]
    if not pairs:
        return {"split": split, "skipped": "no pairs"}

    in_place = source.resolve() == dest.resolve()
    if backup and in_place and not dry_run:
        copied = backup_split(source, split)
        if copied:
            print(f"  {split}: backed up {copied} file(s) to _original/")

    im_out = dest / split / "im"
    gt_out = dest / split / "gt"
    edge_out = dest / split / "gt_edge"

    params = StrokeParams()
    processed = 0
    resized = 0
    edges = 0
    rejected: list[dict] = []

    for item in pairs:
        try:
            rgb = imread_rgb(item["im"])
            gt = to_binary_mask(imread_gray(item["gt"]))
        except Exception as exc:
            rejected.append({"image": item["im"].name, "reason": f"read error: {exc}"})
            continue

        if gt.shape[:2] != rgb.shape[:2]:
            gt = cv2.resize(gt, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

        height, width = rgb.shape[:2]
        target_w, target_h, _ = canonical_size(height, width, short_side)
        needs_resize = (target_w, target_h) != (width, height)

        if needs_resize:
            rgb = resize(rgb, target_w, target_h, is_mask=False)
            gt = resize(gt, target_w, target_h, is_mask=True)

        gt = ensure_binary(gt)
        ok, reason = validate_pair(rgb, gt)
        if not ok:
            rejected.append({"image": item["im"].name, "reason": reason})
            continue

        if not dry_run:
            # In place: only rewrite what changed. Separate output: write everything.
            if needs_resize or not in_place:
                new_im = im_out / f"{item['stem']}.jpg"
                imwrite(new_im, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                if in_place and new_im != item["im"] and item["im"].is_file():
                    item["im"].unlink()  # drop the stale .png copy
                imwrite(gt_out / f"{item['stem']}.png", gt)
                if needs_resize:
                    resized += 1

            edge_path = edge_out / f"{item['stem']}.png"
            if make_edge and (force or not edge_path.is_file()):
                stroke = synthesize_stroke(gt, rgb=rgb, params=params, rng=0)
                imwrite(edge_path, stroke)
                edges += 1

        processed += 1
        if processed % 500 == 0:
            print(f"  {split}: {processed}/{len(pairs)}")

    return {
        "split": split,
        "images": len(pairs),
        "processed": processed,
        "resized": resized,
        "edges_written": edges,
        "rejected": len(rejected),
        "rejections": rejected[:20],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preprocess a BAM dataset")
    add_config_args(parser)
    parser.add_argument("--dataset", default=None, help="source path (default: config dataset)")
    parser.add_argument("--out", default=None, help="output path (default: <dataset>_prepared)")
    parser.add_argument("--inplace", action="store_true", help="rewrite the dataset itself")
    parser.add_argument("--short-side", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="max images per split")
    parser.add_argument("--force", action="store_true", help="regenerate gt_edge")
    parser.add_argument("--no-edge", action="store_true", help="skip gt_edge creation")
    parser.add_argument("--backup", action="store_true", help="copy im/gt to _original/ first")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    cfg = Config.load(args.config, args.overrides)
    source = Path(args.dataset) if args.dataset else cfg.dataset_path
    if not source.is_dir():
        raise SystemExit(f"dataset folder not found: {source}")

    if args.inplace:
        dest = source
    elif args.out:
        dest = Path(args.out)
    else:
        dest = source.with_name(f"{source.name}_prepared")

    short_side = int(args.short_side or cfg.short_side)
    print(f"source: {source}")
    print(f"output: {dest} (short side {short_side})")
    if args.dry_run:
        print("dry run: nothing will be written")
    if dest.resolve() == source.resolve():
        print("WARNING: in-place mode rewrites im/ and gt/. Keep a copy if you need the originals.")

    reports = []
    for split in SPLITS:
        report = preprocess_split(
            source,
            dest,
            split,
            short_side,
            force=args.force,
            dry_run=args.dry_run,
            limit=args.limit,
            make_edge=not args.no_edge,
            backup=args.backup,
        )
        reports.append(report)
        summary = {k: v for k, v in report.items() if k != "rejections"}
        print(f"  {split}: {json.dumps(summary)}")

    meta = {
        "dataset": dest.name,
        "source": str(source),
        "short_side": short_side,
        "splits": {r["split"]: {k: v for k, v in r.items() if k != "rejections"} for r in reports},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if not args.dry_run:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"wrote {dest / 'meta.json'}")

    total = sum(r.get("processed", 0) for r in reports)
    rejected = sum(r.get("rejected", 0) for r in reports)
    print(f"done: {total} samples processed, {rejected} rejected")
    if total and not args.dry_run:
        print(f"next: set 'dataset: {dest.name}' in config.yaml, then python -m training.train")
    return 0 if total else 1


if __name__ == "__main__":
    raise SystemExit(main())
