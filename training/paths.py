"""Dataset path helpers, free of torch so preprocessing stays lightweight.

A dataset looks like this:

    datasets/<name>/
        train/im/*.jpg     train/gt/*.png
        train/gt_edge/*.png    train/gt_brush/*.png
        val/im/*.jpg       val/gt/*.png       val/gt_edge/*.png   val/gt_brush/*.png
        test/im/*.jpg      test/gt/*.png      test/gt_edge/*.png  test/gt_brush/*.png

``gt_edge`` is the clean boundary line the model predicts. ``gt_brush`` is the
rough brush stroke the model receives. Both are produced by preprocessing;
either may be absent, and the trainer falls back to generating it from ``gt``.
Images and masks are paired by filename stem, case-insensitively.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

SPLITS = ("train", "val", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def pair_split(
    root: str | Path,
    split: str,
    require_gt: bool = True,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Pair every image in ``<root>/<split>/im`` with ``<root>/<split>/gt``.

    ``gt_edge`` and ``gt_brush`` are attached when present; a missing pair is
    not an error, because both rasters are reproducible from ``gt``.
    """
    root = Path(root)
    im_dir = root / split / "im"
    gt_dir = root / split / "gt"
    edge_dir = root / split / "gt_edge"
    brush_dir = root / split / "gt_brush"

    if not im_dir.is_dir():
        raise FileNotFoundError(f"missing image folder: {im_dir}")
    if require_gt and not gt_dir.is_dir():
        raise FileNotFoundError(f"missing mask folder: {gt_dir}")

    masks: dict[str, Path] = {}
    if gt_dir.is_dir():
        for path in sorted(gt_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                masks.setdefault(path.stem.lower(), path)

    files: list[dict[str, Any]] = []
    missing = 0
    for path in sorted(im_dir.iterdir()):
        if not (path.is_file() and path.suffix.lower() in IMAGE_EXTS):
            continue
        mask = masks.get(path.stem.lower())
        if mask is None:
            missing += 1
            continue
        edge = edge_dir / f"{path.stem}.png"
        brush = brush_dir / f"{path.stem}.png"
        stat = path.stat()
        files.append(
            {
                "stem": path.stem,
                "im": path,
                "gt": mask,
                "edge": edge if edge.is_file() else None,
                "brush": brush if brush.is_file() else None,
                "size": int(stat.st_size),
                "mtime": int(stat.st_mtime_ns),
            }
        )

    if missing and verbose:
        print(f"[data] {split}: {missing} image(s) had no matching mask, skipped")
    return files


def split_counts(root: str | Path) -> dict[str, int]:
    """Image counts per split, ignoring missing splits."""
    root = Path(root)
    counts: dict[str, int] = {}
    for split in SPLITS:
        im_dir = root / split / "im"
        if not im_dir.is_dir():
            continue
        counts[split] = sum(
            1 for p in im_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
    return counts


__all__ = ["SPLITS", "IMAGE_EXTS", "pair_split", "split_counts"]
