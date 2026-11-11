"""Evaluation metrics.

Boundary metrics are the primary signal: the model predicts an edge, and a
1-2 px localisation error matters more than a handful of stray pixels.
"""

from __future__ import annotations

import cv2
import numpy as np

from .stroke_synth import boundary_from_mask, ensure_binary


def _as_binary(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.dtype == np.bool_:
        return (arr.astype(np.uint8)) * 255
    if arr.max(initial=0) <= 1.0:
        return (arr > threshold).astype(np.uint8) * 255
    return ensure_binary(arr)


def _boundary(mask: np.ndarray) -> np.ndarray:
    return boundary_from_mask(_as_binary(mask))


def _dilate(binary: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return binary
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(binary, k)


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    p = _as_binary(pred) > 0
    g = _as_binary(gt) > 0
    union = np.logical_or(p, g).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(p, g).sum() / union)


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p = _as_binary(pred) > 0
    g = _as_binary(gt) > 0
    total = p.sum() + g.sum()
    if total == 0:
        return 1.0
    return float(2.0 * np.logical_and(p, g).sum() / total)


def boundary_f_score(
    pred: np.ndarray, gt: np.ndarray, tolerance: int = 2
) -> tuple[float, float, float]:
    """Boundary precision / recall / F with a pixel tolerance."""
    pb = _boundary(pred) > 0
    gb = _boundary(gt) > 0

    if not pb.any() and not gb.any():
        return 1.0, 1.0, 1.0
    if not pb.any() or not gb.any():
        return 0.0, 0.0, 0.0

    dist_to_gt = cv2.distanceTransform((~gb).astype(np.uint8), cv2.DIST_L2, 5)
    dist_to_pred = cv2.distanceTransform((~pb).astype(np.uint8), cv2.DIST_L2, 5)

    precision = float((dist_to_gt[pb] <= tolerance).mean())
    recall = float((dist_to_pred[gb] <= tolerance).mean())
    if precision + recall <= 0:
        return 0.0, 0.0, 0.0
    f = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f


def boundary_iou(pred: np.ndarray, gt: np.ndarray, dilation: int = 2) -> float:
    """IoU of boundary bands (Cheng et al., 2021)."""
    pb = _dilate(_boundary(pred), dilation) > 0
    gb = _dilate(_boundary(gt), dilation) > 0
    union = np.logical_or(pb, gb).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pb, gb).sum() / union)


def mae(pred: np.ndarray, gt: np.ndarray) -> float:
    p = np.asarray(pred, np.float32)
    g = np.asarray(gt, np.float32)
    if p.max(initial=0) > 1.0:
        p = p / 255.0
    if g.max(initial=0) > 1.0:
        g = g / 255.0
    return float(np.abs(p - g).mean())


def evaluate_masks(
    pred: np.ndarray, gt: np.ndarray, tolerance: int = 2
) -> dict[str, float]:
    p, r, f = boundary_f_score(pred, gt, tolerance=tolerance)
    return {
        "iou": iou(pred, gt),
        "dice": dice(pred, gt),
        "boundary_iou": boundary_iou(pred, gt),
        "bf_precision": p,
        "bf_recall": r,
        "bf": f,
        "mae": mae(pred, gt),
    }


class MetricAccumulator:
    """Running mean of per-image metrics."""

    def __init__(self) -> None:
        self._sums: dict[str, float] = {}
        self._count = 0

    def add(self, values: dict[str, float]) -> None:
        for key, value in values.items():
            self._sums[key] = self._sums.get(key, 0.0) + float(value)
        self._count += 1

    def mean(self) -> dict[str, float]:
        if self._count == 0:
            return {}
        return {k: v / self._count for k, v in self._sums.items()}

    def __len__(self) -> int:
        return self._count


__all__ = [
    "iou",
    "dice",
    "boundary_f_score",
    "boundary_iou",
    "mae",
    "evaluate_masks",
    "MetricAccumulator",
]
