"""Crop sampling for training.

Sampling only boundary-centred crops teaches a degenerate shortcut ("draw a line
through the middle"). This sampler enforces a mixture that also includes
off-centre boundaries, crops with no boundary at all, and cluttered crops with
several boundaries. Without the empty bucket the model hallucinates edges; with
it, it learns to stay silent.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .stroke_synth import boundary_from_mask, ensure_binary


@dataclass
class CropBox:
    x0: int
    y0: int
    size: int

    @property
    def x1(self) -> int:
        return self.x0 + self.size

    @property
    def y1(self) -> int:
        return self.y0 + self.size

    @property
    def center(self) -> tuple[int, int]:
        return self.x0 + self.size // 2, self.y0 + self.size // 2


@dataclass
class ImageState:
    """Precomputed per-image maps shared across all crops of one image."""

    height: int
    width: int
    boundary: np.ndarray  # uint8 0/255
    dist: np.ndarray  # float32 distance to nearest boundary pixel
    density: np.ndarray  # float32 local boundary density in [0, 1]
    boundary_ys: np.ndarray
    boundary_xs: np.ndarray
    dense_ys: np.ndarray
    dense_xs: np.ndarray


class CropSampler:
    def __init__(
        self,
        crop_size: int = 256,
        mix: dict | None = None,
        jitter: int = 24,
        near_range: tuple[int, int] = (32, 96),
        empty_clearance: int = 32,
        dense_percentile: float = 80.0,
    ) -> None:
        self.crop_size = int(crop_size)
        mix = mix or {"boundary": 0.50, "near": 0.20, "empty": 0.15, "dense": 0.15}
        total = float(sum(mix.values()))
        if total <= 0:
            raise ValueError("crop mix must have a positive total")
        self.buckets = ["boundary", "near", "empty", "dense"]
        self.weights = np.array([mix[k] for k in self.buckets], np.float64) / total
        self.jitter = int(jitter)
        self.near_range = near_range
        self.empty_clearance = int(empty_clearance)
        self.dense_percentile = float(dense_percentile)

    # -- preparation ------------------------------------------------------- #

    def prepare(self, gt: np.ndarray) -> ImageState:
        binary = ensure_binary(gt)
        boundary = boundary_from_mask(binary)
        dist = cv2.distanceTransform(255 - boundary, cv2.DIST_L2, 5).astype(np.float32)

        density = cv2.blur(
            (boundary > 0).astype(np.float32), (17, 17), borderType=cv2.BORDER_REPLICATE
        )
        ys, xs = np.nonzero(boundary > 0)
        if ys.size == 0:
            ys = np.array([binary.shape[0] // 2], np.int64)
            xs = np.array([binary.shape[1] // 2], np.int64)

        dense_ys, dense_xs = ys, xs
        if density.size:
            thresh = np.percentile(density[ys, xs], self.dense_percentile)
            keep = density[ys, xs] >= thresh
            if keep.any():
                dense_ys, dense_xs = ys[keep], xs[keep]

        return ImageState(
            height=binary.shape[0],
            width=binary.shape[1],
            boundary=boundary,
            dist=dist,
            density=density,
            boundary_ys=ys,
            boundary_xs=xs,
            dense_ys=dense_ys,
            dense_xs=dense_xs,
        )

    # -- sampling ---------------------------------------------------------- #

    def sample(self, state: ImageState, rng: np.random.Generator) -> CropBox:
        bucket = self.buckets[int(rng.choice(len(self.buckets), p=self.weights))]
        self.last_bucket = bucket
        if bucket == "boundary":
            cx, cy = self._boundary_center(state, rng)
            cx += int(rng.integers(-self.jitter, self.jitter + 1))
            cy += int(rng.integers(-self.jitter, self.jitter + 1))
        elif bucket == "near":
            cx, cy = self._boundary_center(state, rng)
            radius = float(rng.uniform(*self.near_range))
            angle = float(rng.uniform(0, 2 * np.pi))
            cx += int(round(radius * np.cos(angle)))
            cy += int(round(radius * np.sin(angle)))
        elif bucket == "dense":
            cx, cy = self._dense_center(state, rng)
            cx += int(rng.integers(-self.jitter, self.jitter + 1))
            cy += int(rng.integers(-self.jitter, self.jitter + 1))
        else:
            cx, cy = self._empty_center(state, rng)

        half = self.crop_size // 2
        cx = int(np.clip(cx, 0, state.width - 1))
        cy = int(np.clip(cy, 0, state.height - 1))
        return CropBox(x0=cx - half, y0=cy - half, size=self.crop_size)

    # -- internals --------------------------------------------------------- #

    @staticmethod
    def _boundary_center(state: ImageState, rng: np.random.Generator) -> tuple[int, int]:
        i = int(rng.integers(0, state.boundary_ys.size))
        return int(state.boundary_xs[i]), int(state.boundary_ys[i])

    @staticmethod
    def _dense_center(state: ImageState, rng: np.random.Generator) -> tuple[int, int]:
        i = int(rng.integers(0, state.dense_ys.size))
        return int(state.dense_xs[i]), int(state.dense_ys[i])

    def _empty_center(self, state: ImageState, rng: np.random.Generator) -> tuple[int, int]:
        for _ in range(30):
            cx = int(rng.integers(0, state.width))
            cy = int(rng.integers(0, state.height))
            if state.dist[cy, cx] > self.empty_clearance:
                return cx, cy
        # Deterministic fallback: the single emptiest pixel in the image.
        flat = int(np.argmax(state.dist))
        cy, cx = divmod(flat, state.width)
        return int(cx), int(cy)


__all__ = ["CropBox", "ImageState", "CropSampler"]
