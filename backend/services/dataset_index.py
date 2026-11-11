"""Dataset discovery and cached image loading.

A dataset is any directory under ``datasets/`` that contains ``rgb/``. The GT
and stroke folders are optional: a user can label a fresh folder that has only
images.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from imgio import imread_gray, imread_rgb, to_binary_mask
from .cache import LRUCache

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class DatasetEntry:
    name: str
    root: Path
    images: dict[str, Path] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.images)


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
            rgb = path / "rgb"
            if path.is_dir() and rgb.is_dir():
                try:
                    stamps.append((path.name, int(rgb.stat().st_mtime)))
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
                rgb = path / "rgb"
                if not (path.is_dir() and rgb.is_dir()):
                    continue
                images: dict[str, Path] = {}
                for file in sorted(rgb.iterdir()):
                    if file.is_file() and file.suffix.lower() in IMAGE_EXTS:
                        images[file.stem] = file
                entries[path.name] = DatasetEntry(name=path.name, root=path, images=images)

            self._entries = entries
            self._stamp = stamp
            self._images.clear()

    # -- queries ----------------------------------------------------------- #

    def list_datasets(self) -> list[dict]:
        self.refresh()
        out = []
        for entry in self._entries.values():
            out.append(
                {
                    "name": entry.name,
                    "count": entry.count,
                    "has_gt": (entry.root / "gt").is_dir(),
                    "has_edge": (entry.root / "gt_edge").is_dir(),
                    "thumb": (
                        f"/media/{entry.name}/rgb/{entry.images[next(iter(entry.images))].name}"
                        if entry.images
                        else None
                    ),
                }
            )
        return out

    def names(self) -> list[str]:
        self.refresh()
        return list(self._entries)

    def list_images(self, dataset: str) -> list[str]:
        self.refresh()
        entry = self._entry(dataset)
        return sorted(entry.images)

    def _entry(self, dataset: str) -> DatasetEntry:
        entry = self._entries.get(dataset)
        if entry is None:
            raise KeyError(f"unknown dataset: {dataset}")
        return entry

    def image_path(self, dataset: str, name: str) -> Path:
        self.refresh()
        entry = self._entry(dataset)
        path = entry.images.get(name)
        if path is None:
            raise KeyError(f"unknown image {name!r} in dataset {dataset!r}")
        return path

    def mask_path(self, dataset: str, name: str) -> Path | None:
        self.refresh()
        entry = self._entry(dataset)
        stem = Path(name).stem
        for sub in ("gt", "labels"):
            folder = entry.root / sub
            if not folder.is_dir():
                continue
            for ext in IMAGE_EXTS:
                candidate = folder / f"{stem}{ext}"
                if candidate.is_file():
                    return candidate
        return None

    def resolve(self, dataset: str, sub: str, name: str) -> Path | None:
        """Resolve a file in a dataset subfolder, with or without an extension."""
        self.refresh()
        entry = self._entry(dataset)
        folder = entry.root / sub
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

    # -- pixels ------------------------------------------------------------ #

    def load_rgb(self, dataset: str, name: str) -> np.ndarray:
        path = self.image_path(dataset, name)
        key = ("rgb", dataset, name, path.stat().st_mtime_ns)
        return self._images.get_or_create(key, lambda: imread_rgb(path))

    def load_mask(self, dataset: str, name: str) -> np.ndarray | None:
        path = self.mask_path(dataset, name)
        if path is None:
            return None
        key = ("gt", dataset, name, path.stat().st_mtime_ns)
        return self._images.get_or_create(key, lambda: to_binary_mask(imread_gray(path)))

    def load_stroke(self, dataset: str, name: str) -> np.ndarray | None:
        path = self.resolve(dataset, "gt_edge", name)
        if path is None:
            return None
        key = ("edge", dataset, name, path.stat().st_mtime_ns)
        return self._images.get_or_create(key, lambda: imread_gray(path))

    def thumbnail(self, dataset: str, name: str, size: int = 160) -> np.ndarray:
        rgb = self.load_rgb(dataset, name)
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
