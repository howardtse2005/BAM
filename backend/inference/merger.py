"""Merge per-crop predictions onto the full canvas.

This preserves the reference ``PredictionMerger`` semantics:

    average the overlapping predictions  ->  boost repeatedly brushed pixels
    ->  threshold  ->  denoise  ->  morphology

Two documented deltas:

  * ``weighting="gaussian"`` optionally down-weights crop borders, where the
    model has the least context. Default stays ``uniform`` for parity.
  * ``sensitivity`` is a float in [1, 3] rather than an unbounded 0-100 integer.
    The original ``base ** (n-1)`` with ``base=5`` saturates after two overlaps,
    which makes the threshold stop mattering.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CropPlacement:
    x0: int
    y0: int
    size: int

    @property
    def x1(self) -> int:
        return self.x0 + self.size

    @property
    def y1(self) -> int:
        return self.y0 + self.size


@dataclass
class MergeResult:
    mask: np.ndarray  # uint8 {0, 255}
    edge: np.ndarray  # float32 [0, 1]
    covered: int  # pixels that received at least one prediction


def disk(radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def _gaussian_weight(size: int, sigma_frac: float = 0.45) -> np.ndarray:
    axis = np.arange(size, dtype=np.float32) - (size - 1) / 2.0
    sigma = max(1.0, size * sigma_frac / 2.0)
    profile = np.exp(-(axis**2) / (2.0 * sigma**2))
    weight = np.outer(profile, profile)
    peak = float(weight.max())
    return (weight / peak).astype(np.float32) if peak > 0 else weight


class PredictionMerger:
    def __init__(
        self,
        height: int,
        width: int,
        weighting: str = "uniform",
        sigma_frac: float = 0.45,
    ) -> None:
        if height <= 0 or width <= 0:
            raise ValueError("canvas dimensions must be positive")
        if weighting not in ("uniform", "gaussian"):
            raise ValueError("weighting must be 'uniform' or 'gaussian'")

        self.height = int(height)
        self.width = int(width)
        self.weighting = weighting
        self.values = np.zeros((self.height, self.width), np.float64)
        self.weights = np.zeros((self.height, self.width), np.float64)
        self.brush = np.zeros((self.height, self.width), np.int32)
        self._weight_cache: dict[int, np.ndarray] = {}
        self._sigma_frac = float(sigma_frac)

    # -- accumulation ------------------------------------------------------ #

    def add(
        self,
        placement: CropPlacement,
        probability: np.ndarray,
        valid: np.ndarray | None = None,
    ) -> bool:
        prob = np.asarray(probability, np.float32)
        if prob.ndim != 2:
            raise ValueError("probability map must be 2D")
        if prob.shape[0] != placement.size or prob.shape[1] != placement.size:
            raise ValueError(
                f"crop size mismatch: got {prob.shape}, expected {placement.size}"
            )

        x0, y0 = placement.x0, placement.y0
        x1, y1 = x0 + placement.size, y0 + placement.size

        cx0, cy0 = max(0, x0), max(0, y0)
        cx1, cy1 = min(self.width, x1), min(self.height, y1)
        if cx1 <= cx0 or cy1 <= cy0:
            return False

        sx0, sy0 = cx0 - x0, cy0 - y0
        sx1, sy1 = sx0 + (cx1 - cx0), sy0 + (cy1 - cy0)
        sub = prob[sy0:sy1, sx0:sx1]

        if valid is None:
            valid_sub = np.ones_like(sub, np.float32)
        else:
            valid_sub = np.asarray(valid, np.float32)[sy0:sy1, sx0:sx1]
            if valid_sub.shape != sub.shape:
                raise ValueError("validity mask shape mismatch")

        if self.weighting == "gaussian":
            if placement.size not in self._weight_cache:
                self._weight_cache[placement.size] = _gaussian_weight(
                    placement.size, self._sigma_frac
                )
            weight = self._weight_cache[placement.size][sy0:sy1, sx0:sx1]
        else:
            weight = np.ones_like(sub, np.float32)

        effective = weight * valid_sub
        self.values[cy0:cy1, cx0:cx1] += sub * effective
        self.weights[cy0:cy1, cx0:cx1] += effective
        self.brush[cy0:cy1, cx0:cx1] += (valid_sub > 0).astype(np.int32)
        return True

    # -- finalisation ------------------------------------------------------ #

    def finalize(
        self,
        threshold: float = 0.5,
        sensitivity: float = 1.0,
        apply_morphology: bool = True,
        morph_kernel_size: int = 3,
        morph_iterations: int = 2,
        apply_dbscan: bool = False,
        db_eps: float = 10.0,
        db_min_samples: int = 5,
        min_component_area: int = 24,
    ) -> MergeResult:
        covered_mask = self.weights > 0
        edge = np.zeros_like(self.values, dtype=np.float64)
        np.divide(self.values, self.weights, out=edge, where=covered_mask)

        sensitivity = float(max(1.0, sensitivity))
        if sensitivity > 1.0 and np.any(self.brush > 1):
            exponent = np.clip(self.brush - 1, 0, 8)
            boost = np.power(sensitivity, exponent.astype(np.float64))
            edge = np.where(covered_mask, edge * boost, edge)

        edge = np.clip(edge, 0.0, 1.0).astype(np.float32)
        binary = ((edge > float(threshold)).astype(np.uint8)) * 255

        if apply_dbscan and binary.any():
            binary = _dbscan_filter(binary, db_eps, db_min_samples)
        elif binary.any():
            binary = _component_filter(binary, min_component_area)

        if apply_morphology and binary.any():
            binary = apply_morphological_filtering(
                binary, morph_kernel_size, morph_iterations
            )

        return MergeResult(
            mask=binary.astype(np.uint8),
            edge=edge,
            covered=int(covered_mask.sum()),
        )


def _component_filter(mask: np.ndarray, min_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    out = np.zeros_like(mask)
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def _dbscan_filter(mask: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Optional DBSCAN denoise, matching the reference implementation.

    Falls back to a component filter when scikit-learn is unavailable.
    """
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        return _component_filter(mask, max(1, int(min_samples)))

    coords = np.column_stack(np.nonzero(mask))
    if coords.shape[0] == 0:
        return mask
    labels = DBSCAN(eps=float(eps), min_samples=int(min_samples)).fit_predict(coords)
    out = np.zeros_like(mask)
    keep = labels != -1
    out[coords[keep, 0], coords[keep, 1]] = 255
    return out


def apply_morphological_filtering(
    mask: np.ndarray, kernel_size: int = 3, iterations: int = 2
) -> np.ndarray:
    """Close -> open -> median, on an elliptical kernel."""
    binary = (mask > 127).astype(np.uint8)
    kernel_size = max(1, int(kernel_size))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    closed = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, kernel, iterations=max(1, int(iterations))
    )
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    if kernel_size >= 3:
        opened = cv2.medianBlur(opened, min(kernel_size if kernel_size % 2 else kernel_size + 1, 5))
    return (opened * 255).astype(np.uint8)


__all__ = [
    "CropPlacement",
    "MergeResult",
    "PredictionMerger",
    "apply_morphological_filtering",
    "disk",
]
