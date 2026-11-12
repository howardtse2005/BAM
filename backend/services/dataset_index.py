"""Dataset discovery and cached image loading.

A dataset is any folder under ``datasets/`` that contains at least one split:

    <dataset>/train/im/   <dataset>/train/gt/
    <dataset>/train/gt_edge/   <dataset>/train/gt_brush/
    <dataset>/val/im/     ...
    <dataset>/test/im/    ...

Images are addressed by ``(dataset, split, stem)``. The mask, edge and brush
folders are optional: a fresh folder with only ``im/`` can still be labelled.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from urllib.parse import quote
from pathlib import Path

import cv2
import numpy as np

from imgio import imread_gray, imread_rgb, to_binary_mask
from training.paths import IMAGE_EXTS, SPLITS

from .cache import LRUCache


@dataclass
class DatasetEntry:
    name: str
    root: Path
    images: dict[str, dict[str, Path]] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return sum(len(v) for v in self.images.values())

    def split_counts(self) -> dict[str, int]:
        return {split: len(v) for split, v in self.images.items()}


class DatasetIndex:
    def __init__(self, root: str | Path, cache_size: int = 64) -> None:
        self.root = Path(root)
        self._lock = threading.RLock()
        self._entries: dict[str, DatasetEntry] = {}
        self._stamp: tuple | None = None
        self._images = LRUCache(cache_size)

    # -- scanning ---------------------------------------------------------- #

    def _current_stamp(self) -> tuple:
        if not self.root.is_dir():
            return ()
        stamps = []
        for path in sorted(self.root.iterdir()):
            if not path.is_dir():
                continue
            for split in SPLITS:
                im_dir = path / split / "im"
                if im_dir.is_dir():
                    try:
                        stamps.append((path.name, split, int(im_dir.stat().st_mtime)))
                    except OSError:
                        continue
        return tuple(stamps)

    def refresh(self, force: bool = False) -> None:
        with self._lock:
            stamp = self._current_stamp()
            if not force and stamp == self._stamp and self._entries:
                return

            entries: dict[str, DatasetEntry] = {}
            for path in sorted(self.root.iterdir()) if self.root.is_dir() else []:
                if not path.is_dir():
                    continue
                splits: dict[str, dict[str, Path]] = {}
                for split in SPLITS:
                    im_dir = path / split / "im"
                    if not im_dir.is_dir():
                        continue
                    images = {
                        file.stem: file
                        for file in sorted(im_dir.iterdir())
                        if file.is_file() and file.suffix.lower() in IMAGE_EXTS
                    }
                    if images:
                        splits[split] = images
                if splits:
                    entries[path.name] = DatasetEntry(name=path.name, root=path, images=splits)

            self._entries = entries
            self._stamp = stamp
            self._images.clear()

    # -- queries ----------------------------------------------------------- #

    def list_datasets(self) -> list[dict]:
        self.refresh()
        out = []
        for entry in self._entries.values():
            splits = entry.split_counts()
            primary = next(
                (s for s in ("train", "val", "test") if entry.images.get(s)),
                next(iter(entry.images), None),
            )
            thumb = None
            if primary:
                first = entry.images[primary][next(iter(entry.images[primary]))]
                thumb = (
                    f"/media/{quote(entry.name, safe='')}/{primary}/im/"
                    f"{quote(first.name, safe='')}"
                )
            out.append(
                {
                    "name": entry.name,
                    "count": entry.count,
                    "splits": splits,
                    "has_gt": any((entry.root / s / "gt").is_dir() for s in splits),
                    "has_edge": any((entry.root / s / "gt_edge").is_dir() for s in splits),
                    "has_brush": any((entry.root / s / "gt_brush").is_dir() for s in splits),
                    "thumb": thumb,
                }
            )
        return out

    def names(self) -> list[str]:
        self.refresh()
        return list(self._entries)

    def _entry(self, dataset: str) -> DatasetEntry:
        entry = self._entries.get(dataset)
        if entry is None:
            raise KeyError(f"unknown dataset: {dataset}")
        return entry

    def splits(self, dataset: str) -> list[str]:
        self.refresh()
        return list(self._entry(dataset).images)

    def list_images(self, dataset: str, split: str) -> list[str]:
        self.refresh()
        entry = self._entry(dataset)
        images = entry.images.get(split)
        if images is None:
            raise KeyError(f"dataset {dataset!r} has no split {split!r}")
        return sorted(images)

    # -- pixels ------------------------------------------------------------ #

    def resolve(self, dataset: str, split: str, sub: str, name: str) -> Path | None:
        """Resolve a file in a dataset split, with or without an extension."""
        self.refresh()
        entry = self._entry(dataset)
        folder = entry.root / split / sub
        if not folder.is_dir():
            return None

        direct = folder / name
        if direct.is_file():
            return direct

        stem = Path(name).stem
        for ext in IMAGE_EXTS:
            candidate = folder / f"{stem}{ext}"
            if candidate.is_file():
                return candidate
        return None

    def image_path(self, dataset: str, split: str, name: str) -> Path:
        path = self.resolve(dataset, split, "im", name)
        if path is None:
            raise KeyError(f"unknown image {name!r} in {dataset}/{split}")
        return path

    def mask_path(self, dataset: str, split: str, name: str) -> Path | None:
        return self.resolve(dataset, split, "gt", name)

    def load_rgb(self, dataset: str, split: str, name: str) -> np.ndarray:
        path = self.image_path(dataset, split, name)
        key = ("rgb", dataset, split, name, path.stat().st_mtime_ns)
        return self._images.get_or_create(key, lambda: imread_rgb(path))

    def load_mask(self, dataset: str, split: str, name: str) -> np.ndarray | None:
        path = self.mask_path(dataset, split, name)
        if path is None:
            return None
        key = ("gt", dataset, split, name, path.stat().st_mtime_ns)
        return self._images.get_or_create(key, lambda: to_binary_mask(imread_gray(path)))

    def load_edge(self, dataset: str, split: str, name: str) -> np.ndarray | None:
        """The clean edge line of the object, as stored by preprocessing."""
        path = self.resolve(dataset, split, "gt_edge", name)
        if path is None:
            return None
        key = ("edge", dataset, split, name, path.stat().st_mtime_ns)
        return self._images.get_or_create(key, lambda: imread_gray(path))

    def load_brush(self, dataset: str, split: str, name: str) -> np.ndarray | None:
        """The synthetic brush stroke paired with the image, if one exists."""
        path = self.resolve(dataset, split, "gt_brush", name)
        if path is None:
            return None
        key = ("brush", dataset, split, name, path.stat().st_mtime_ns)
        return self._images.get_or_create(key, lambda: imread_gray(path))

    def thumbnail(self, dataset: str, split: str, name: str, size: int = 160) -> np.ndarray:
        rgb = self.load_rgb(dataset, split, name)
        h, w = rgb.shape[:2]
        scale = size / max(h, w)
        resized = cv2.resize(
            rgb, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA
        )
        return cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)

    def stats(self) -> dict:
        self.refresh()
        return {
            "root": str(self.root),
            "datasets": len(self._entries),
            "images": sum(e.count for e in self._entries.values()),
        }


__all__ = ["DatasetIndex", "DatasetEntry"]
