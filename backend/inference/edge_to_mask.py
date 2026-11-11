"""Convert a predicted edge map into a filled object mask.

The network predicts an edge. A product needs a mask. This module bridges the
two deterministically:

    thin the edge -> build a barrier -> flood the outside -> invert
    -> keep components that overlap the stroke loop -> refine -> smooth

The stroke loop provides the interior seed, so the model never has to resolve
which side of the boundary is foreground. Holes are preserved automatically,
because an enclosed region is not reachable from the frame corners.

Three fallbacks keep the result usable when the loop is open or the barrier
leaks: a morphological loop closure, GrabCut, and finally the raw loop refined
by a guided filter. ``mode="line"`` skips all of this and returns an edge ribbon,
which is the original masker behaviour for cracks and thin structures.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FillResult:
    mask: np.ndarray  # uint8 {0, 255}
    mode: str  # flood | closed_gap | grabcut | loop | line | empty
    confidence: float
    low_confidence: bool
    edge: np.ndarray  # uint8 {0, 255} thinned edge


# --------------------------------------------------------------------------- #
# Morphology helpers
# --------------------------------------------------------------------------- #


def disk(radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def thin(binary: np.ndarray, max_iter: int = 40) -> np.ndarray:
    """Zhang-Suen thinning to a 1-px skeleton.

    Implemented directly so the backend does not depend on scikit-image.
    """
    img = (np.asarray(binary) > 0).astype(np.uint8)
    if not img.any():
        return (img * 255).astype(np.uint8)

    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            padded = np.pad(img, 1, mode="constant")
            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]

            sequence = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            neighbours = (
                p2.astype(np.uint8)
                + p3
                + p4
                + p5
                + p6
                + p7
                + p8
                + p9
            )
            transitions = np.zeros_like(img, dtype=np.uint8)
            for i in range(8):
                transitions += ((sequence[i] == 0) & (sequence[i + 1] == 1)).astype(
                    np.uint8
                )

            if step == 0:
                c1 = (p2 * p4 * p6) == 0
                c2 = (p4 * p6 * p8) == 0
            else:
                c1 = (p2 * p4 * p8) == 0
                c2 = (p2 * p6 * p8) == 0

            remove = (
                (img == 1)
                & (neighbours >= 2)
                & (neighbours <= 6)
                & (transitions == 1)
                & c1
                & c2
            )
            if remove.any():
                img[remove] = 0
                changed = True

        if not changed:
            break

    return (img * 255).astype(np.uint8)


def fill_holes(binary: np.ndarray) -> np.ndarray:
    """Fill enclosed background regions, leaving the outer border alone."""
    b = (np.asarray(binary) > 0).astype(np.uint8) * 255
    padded = cv2.copyMakeBorder(b, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 255)
    holes = (flood == 0) & (padded == 0)
    out = padded.copy()
    out[holes] = 255
    return out[1:-1, 1:-1]


def _remove_small_holes(mask: np.ndarray, max_area: int) -> np.ndarray:
    height, width = mask.shape
    inverse = (mask == 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, connectivity=4)
    out = mask.copy()
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if x == 0 or y == 0 or x + w >= width or y + h >= height:
            continue
        if area <= max_area:
            out[labels == i] = 255
    return out


def _free_components(barrier: np.ndarray) -> tuple[np.ndarray, set[int]]:
    """Label the free space around ``barrier`` and find what touches the frame.

    Returns ``(labels, outside_labels)`` at the original resolution. Free-space
    components that are not in ``outside_labels`` are enclosed by the edge map:
    they are object interiors or holes.
    """
    padded = np.pad((barrier > 0).astype(np.uint8), 1, constant_values=0)
    free = (padded == 0).astype(np.uint8)
    count, labels = cv2.connectedComponents(free, connectivity=8)
    border = np.concatenate(
        [labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]
    )
    outside_labels = {int(v) for v in np.unique(border) if v != 0}
    return labels[1:-1, 1:-1], outside_labels


def _select_object_region(
    barrier: np.ndarray, stroke_bin: np.ndarray
) -> np.ndarray | None:
    """Pick the enclosed free-space component that belongs to the object.

    The stroke ring is drawn along the outer boundary, so the object component is
    the one adjacent to it. Holes inside the object are separate components and
    are therefore excluded, which is what preserves donuts and windows.
    """
    labels, outside = _free_components(barrier)
    enclosed = [i for i in range(1, int(labels.max()) + 1) if i not in outside]
    if not enclosed:
        return None

    stroke_near = cv2.dilate(stroke_bin, disk(4))
    scored: list[tuple[bool, int, int]] = []
    for index in enclosed:
        component = labels == index
        area = int(component.sum())
        adjacent = bool(np.logical_and(component, stroke_near > 0).any())
        scored.append((adjacent, area, index))

    adjacent_only = [item for item in scored if item[0]]
    _, _, chosen = max(adjacent_only or scored, key=lambda item: item[1])

    mask = ((labels == chosen).astype(np.uint8)) * 255
    mask[barrier > 0] = 255
    return mask


def guided_filter(
    guide: np.ndarray, src: np.ndarray, radius: int = 8, eps: float = 1e-3
) -> np.ndarray:
    """Edge-preserving refinement with a grayscale guide.

    A local linear model per window: q = a * I + b. Snaps the mask boundary to
    image edges without the cost of a full optimisation.
    """
    guide = np.asarray(guide, np.float32)
    if guide.max(initial=0) > 1.5:
        guide = guide / 255.0
    src = np.asarray(src, np.float32)
    if src.max(initial=0) > 1.5:
        src = src / 255.0

    radius = max(1, int(radius))
    size = (2 * radius + 1, 2 * radius + 1)

    mean_i = cv2.blur(guide, size)
    mean_p = cv2.blur(src, size)
    corr_i = cv2.blur(guide * guide, size)
    corr_ip = cv2.blur(guide * src, size)

    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p

    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i

    mean_a = cv2.blur(a, size)
    mean_b = cv2.blur(b, size)
    return mean_a * guide + mean_b


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #


def _thin_edge(edge_prob: np.ndarray, edge_threshold: float) -> np.ndarray:
    smoothed = cv2.GaussianBlur(np.asarray(edge_prob, np.float32), (0, 0), 0.8)
    binary = (smoothed > float(edge_threshold)).astype(np.uint8)
    if not binary.any():
        return np.zeros(binary.shape, np.uint8)

    # Thinning is the most expensive step and is local, so run it on the band's
    # bounding box instead of the whole canvas.
    ys, xs = np.nonzero(binary)
    y0 = max(0, int(ys.min()) - 2)
    y1 = min(binary.shape[0], int(ys.max()) + 3)
    x0 = max(0, int(xs.min()) - 2)
    x1 = min(binary.shape[1], int(xs.max()) + 3)

    skeleton_sub = thin(binary[y0:y1, x0:x1])
    skeleton = np.zeros_like(binary)
    skeleton[y0:y1, x0:x1] = skeleton_sub

    return cv2.morphologyEx(skeleton, cv2.MORPH_CLOSE, disk(1))


def _largest_component(binary: np.ndarray) -> np.ndarray | None:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (binary > 0).astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return None
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return ((labels == index).astype(np.uint8)) * 255


def _plausible_loop(
    loop: np.ndarray | None,
    stroke_bin: np.ndarray,
    total: float,
    min_frac: float,
    max_frac: float,
) -> bool:
    """A real loop encloses most of the stroke's bounding box.

    A partial arc that happens to self-intersect produces a tiny filled region.
    Accepting it would return a small blob and silently ignore the object, so the
    ratio of filled area to stroke bbox area is used to reject it.
    """
    if loop is None:
        return False
    area = float((loop > 0).sum())
    if not (min_frac * total < area < max_frac * total):
        return False

    ys, xs = np.nonzero(stroke_bin)
    if ys.size == 0:
        return False
    bbox = float((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
    return area >= 0.25 * bbox


def _stroke_loop(
    stroke_bin: np.ndarray, min_frac: float = 0.002, max_frac: float = 0.95
) -> tuple[np.ndarray | None, str]:
    """Filled stroke loop, closing small gaps if needed."""
    height, width = stroke_bin.shape
    total = float(height * width)

    loop = _largest_component(fill_holes(stroke_bin))
    if _plausible_loop(loop, stroke_bin, total, min_frac, max_frac):
        return loop, "closed"

    closed = cv2.morphologyEx(stroke_bin, cv2.MORPH_CLOSE, disk(20))
    loop = _largest_component(fill_holes(closed))
    if _plausible_loop(loop, closed, total, min_frac, max_frac):
        return loop, "closed_gap"

    return None, "open"


def _grabcut(
    rgb: np.ndarray, loop: np.ndarray, barrier: np.ndarray
) -> np.ndarray | None:
    if rgb is None:
        return None
    height, width = loop.shape
    if rgb.ndim == 2:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
    else:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if bgr.shape[:2] != (height, width):
        return None

    inner = cv2.erode(loop, disk(3))
    if not inner.any():
        return None

    gc = np.full((height, width), cv2.GC_BGD, np.uint8)
    gc[cv2.dilate(loop, disk(10)) > 0] = cv2.GC_PR_BGD
    gc[loop > 0] = cv2.GC_PR_FGD
    gc[inner > 0] = cv2.GC_FGD
    gc[barrier > 0] = cv2.GC_PR_BGD

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, gc, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None

    out = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8) * 255
    return out


def edge_to_mask(
    edge_prob: np.ndarray,
    stroke: np.ndarray,
    rgb: np.ndarray | None = None,
    edge_threshold: float = 0.45,
    threshold: float = 0.5,
    expand: int = 0,
    smooth: int = 2,
    mode: str = "object",
    min_area_frac: float = 0.002,
) -> FillResult:
    edge_prob = np.asarray(edge_prob, np.float32)
    if edge_prob.ndim != 2:
        raise ValueError("edge_prob must be 2D")
    height, width = edge_prob.shape
    total = float(height * width)

    stroke_bin = (np.asarray(stroke) > 127).astype(np.uint8)
    if stroke_bin.shape != edge_prob.shape:
        raise ValueError("stroke and edge_prob must have the same shape")

    edge = _thin_edge(edge_prob, edge_threshold)

    if mode == "line":
        ribbon = cv2.dilate(edge, disk(max(1, 2 + int(round(smooth)))))
        return FillResult(ribbon, "line", 1.0, False, edge)

    if not edge.any():
        return FillResult(np.zeros((height, width), np.uint8), "empty", 0.0, True, edge)

    barrier = cv2.dilate(edge, disk(1))

    loop, loop_kind = _stroke_loop(stroke_bin)
    if loop is None:
        # The user has not closed the loop. Returning a filled blob would be a
        # guess; returning the edge is honest and actionable.
        return FillResult(edge, "open_loop", 0.15, True, edge)

    loop_area = float((loop > 0).sum())
    roi = cv2.dilate(loop, disk(8))

    mask_u8 = _select_object_region(barrier, stroke_bin)
    if mask_u8 is None:
        mask_u8 = np.zeros((height, width), np.uint8)
    mask_u8 = cv2.bitwise_and(mask_u8, roi)

    mask_u8 = _keep_loop_components(mask_u8, loop, min_area_frac * total)
    used_mode = "flood" if loop_kind == "closed" else "closed_gap"

    coverage = float((mask_u8 > 0).sum()) / total
    if coverage <= 0.0 or coverage > 0.90:
        grabbed = _grabcut(rgb, loop, barrier)
        if grabbed is not None and 0.0 < float((grabbed > 0).mean()) <= 0.90:
            mask_u8 = grabbed
            used_mode = "grabcut"
        else:
            mask_u8 = loop.copy()
            used_mode = "loop"

    mask_u8 = _remove_small_holes(mask_u8, max(16, int(0.0005 * total)))

    if smooth > 0 and rgb is not None:
        guide = (
            cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            if rgb.ndim == 3
            else rgb.astype(np.uint8)
        )
        if guide.shape[:2] == (height, width):
            refined = guided_filter(
                guide, mask_u8.astype(np.float32) / 255.0, radius=4 * int(smooth)
            )
            candidate = ((refined > float(threshold)).astype(np.uint8)) * 255
            if 0.0 < float((candidate > 0).mean()) <= 0.95:
                mask_u8 = candidate

    expand = int(expand)
    if expand > 0:
        mask_u8 = cv2.dilate(mask_u8, disk(expand))
    elif expand < 0:
        mask_u8 = cv2.erode(mask_u8, disk(-expand))

    area = float((mask_u8 > 0).sum())
    confidence = float(np.clip(1.0 - abs(area - loop_area) / max(loop_area, 1.0), 0.0, 1.0))
    if used_mode in ("grabcut", "loop"):
        confidence = min(confidence, 0.45)

    return FillResult(
        mask=mask_u8.astype(np.uint8),
        mode=used_mode,
        confidence=round(confidence, 4),
        low_confidence=confidence < 0.40,
        edge=edge,
    )


def _keep_loop_components(
    mask: np.ndarray, loop: np.ndarray, min_area: float
) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    out = np.zeros_like(mask)
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        component = labels == i
        if np.logical_and(component, loop > 0).any():
            out[component] = 255
    return out


__all__ = [
    "FillResult",
    "edge_to_mask",
    "thin",
    "fill_holes",
    "guided_filter",
    "disk",
]
