"""Synthetic brush strokes and clean edge targets.

Every binary mask yields two rasters:

``gt_brush``
    What a person dragging a brush along the boundary produces: simplified,
    smoothed, wobbling, variable width, and several pixels thick. This is the
    model's 4th input channel.

``gt_edge``
    The clean boundary line of the same object, produced by the same contour
    walk and the same rasteriser with every source of imprecision removed: full
    perimeter coverage, no simplification, no smoothing, no normal-direction
    displacement, constant radius. This is what the model has to predict.

Both cover the same boundary — every contour, holes included — so the input and
the target always describe the same thing. The brush is *imprecise*, not
*partial*: it wanders by up to ``max_offset`` px and only a minority of its
pixels land on the edge, which is what keeps the task from being a copy of the
input channel.

``BrushParams`` still carries the knobs that make a stroke deliberately
incomplete (``coverage`` below 1, ``p_partial``, ``p_two_arcs``, ``p_no_brush``,
``p_distractor``, ``p_interior``, ``p_dry_brush``). They all default to off; turn
them on only if you want a model that also copes with half-drawn strokes.

This module is deliberately dependency-light: NumPy + OpenCV only. It is used
both offline (to write ``gt_edge/`` and ``gt_brush/``) and online (to regenerate
a fresh brush variant per training sample).

All lengths are in canonical pixels (short side 640 unless configured otherwise).
"""

from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #


@dataclass
class BrushParams:
    radius_min: float = 1.5
    radius_max: float = 6.0
    jitter_sigma: float = 3.0
    wobble_amp: tuple = (1.0, 4.0)
    wobble_len: tuple = (20.0, 180.0)
    coverage: tuple = (1.00, 1.00)
    simplify_eps: tuple = (2.0, 10.0)
    smooth_window: tuple = (5, 21)
    max_offset: float = 12.0
    # Off by default: each of these deliberately makes the brush describe less
    # than the edge, which would leave the input and the target misaligned.
    p_no_brush: float = 0.0
    p_distractor: float = 0.0
    p_interior: float = 0.0
    p_partial: float = 0.0
    p_two_arcs: float = 0.0
    p_dry_brush: float = 0.0

    # Config field -> BrushParams field. Kept explicit so the flat config can
    # use readable names (brush_radius_min) without renaming the dataclass.
    _CONFIG_MAP = {
        "brush_radius_min": "radius_min",
        "brush_radius_max": "radius_max",
        "brush_jitter": "jitter_sigma",
        "brush_wobble_amp": "wobble_amp",
        "brush_wobble_len": "wobble_len",
        "brush_coverage": "coverage",
        "brush_simplify": "simplify_eps",
        "brush_smooth": "smooth_window",
        "brush_max_offset": "max_offset",
        "p_no_brush": "p_no_brush",
        "p_distractor": "p_distractor",
        "p_interior": "p_interior",
        "p_partial": "p_partial",
        "p_two_arcs": "p_two_arcs",
        "p_dry_brush": "p_dry_brush",
    }

    @classmethod
    def from_config(cls, cfg) -> "BrushParams":
        """Build from a Config (flat fields) or any object with those names."""
        if cfg is None:
            return cls()
        kwargs = {
            target: getattr(cfg, source)
            for source, target in cls._CONFIG_MAP.items()
            if hasattr(cfg, source)
        }
        return cls(**kwargs)


# --------------------------------------------------------------------------- #
# Mask utilities
# --------------------------------------------------------------------------- #


def ensure_binary(mask: np.ndarray) -> np.ndarray:
    """Return a uint8 mask with values strictly in {0, 255}."""
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.max(initial=0) <= 1:
        mask = mask * 255
    return np.where(mask >= 128, 255, 0).astype(np.uint8)


def _find_contours(binary: np.ndarray) -> list[np.ndarray]:
    """cv2.findContours across OpenCV 3/4/5 return signatures."""
    result = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours = result[0] if len(result) == 2 else result[1]
    return list(contours)


def boundary_from_mask(mask: np.ndarray, thickness: int = 1) -> np.ndarray:
    """1-px boundary raster of a binary mask (outer and inner contours)."""
    binary = ensure_binary(mask)
    contours = _find_contours(binary)
    out = np.zeros_like(binary)
    if contours:
        cv2.drawContours(out, contours, -1, 255, thickness, lineType=cv2.LINE_8)
    return out


def clean_edge(
    mask: np.ndarray,
    radius: float = 1.0,
    min_length: float = 32.0,
) -> np.ndarray:
    """The clean boundary line of a mask: ``gt_edge``.

    The same construction as a synthetic brush stroke, minus every source of
    imprecision: the full perimeter is covered, nothing is simplified or
    smoothed, nothing is displaced along the normal, and the radius is constant.
    What remains is the object boundary drawn as a line.

    Deterministic: no RNG is involved, so the same mask always yields the same
    raster. That is what makes it usable as a training target.
    """
    binary = ensure_binary(mask)
    out = np.zeros_like(binary)
    thickness = max(1, int(round(2.0 * float(radius))))
    for path in mask_to_paths(binary, min_length=min_length):
        points = np.round(path).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [points], True, 255, thickness, lineType=cv2.LINE_AA)
    return out


def edge_band(edge: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Soft training target: a blurred edge line normalised to peak 1.0.

    A 1-px line is 1-3% positive pixels and a 1-px miss is a total loss. The
    band keeps the peak exactly on the edge while giving a graded penalty
    nearby, which is what makes the target trainable under class imbalance.
    """
    arr = np.asarray(edge)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    line = (arr >= 128).astype(np.float32)
    if line.max() <= 0:
        return line
    band = cv2.GaussianBlur(line, (0, 0), max(float(sigma), 1e-3))
    peak = float(band.max())
    if peak > 1e-6:
        band = band / peak
    return np.clip(band, 0.0, 1.0).astype(np.float32)


def boundary_band(mask: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Soft band around the clean edge of ``mask``; the model's target."""
    return edge_band(clean_edge(mask), sigma=sigma)


def brush_distance(brush: np.ndarray, clip_px: float = 16.0) -> np.ndarray:
    """Distance-to-brush ramp in [0, 1]: 1 on the brush, 0 beyond ``clip_px``.

    Used when ``brush_encoding == "sdf"``. A brush stroke is a curve, not a
    closed region, so this is an unsigned distance rather than a signed one.
    """
    binary = ensure_binary(brush)
    dist = cv2.distanceTransform(255 - binary, cv2.DIST_L2, 5)
    ramp = 1.0 - np.clip(dist / max(clip_px, 1e-3), 0.0, 1.0)
    return ramp.astype(np.float32)


def fill_holes(binary: np.ndarray) -> np.ndarray:
    """Fill enclosed background regions without touching the outer border."""
    b = ensure_binary(binary)
    padded = cv2.copyMakeBorder(b, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ff = padded.copy()
    mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(ff, mask, (0, 0), 255)
    holes = (ff == 0) & (padded == 0)
    out = padded.copy()
    out[holes] = 255
    return out[1:-1, 1:-1]


# --------------------------------------------------------------------------- #
# Polyline utilities
# --------------------------------------------------------------------------- #


def mask_to_paths(mask: np.ndarray, min_length: float = 32.0) -> list[np.ndarray]:
    """Ordered contour polylines as float arrays of shape (N, 2) in (x, y)."""
    binary = ensure_binary(mask)
    paths: list[np.ndarray] = []
    for contour in _find_contours(binary):
        pts = contour.reshape(-1, 2).astype(np.float32)
        if pts.shape[0] < 3:
            continue
        if polyline_length(pts, closed=True) < min_length:
            continue
        paths.append(pts)
    paths.sort(key=lambda p: polyline_length(p, closed=True), reverse=True)
    return paths


def polyline_length(pts: np.ndarray, closed: bool = False) -> float:
    if pts.shape[0] < 2:
        return 0.0
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(d.sum())
    if closed:
        total += float(np.linalg.norm(pts[0] - pts[-1]))
    return total


def resample_polyline(pts: np.ndarray, step: float) -> np.ndarray:
    """Resample by arc length, preserving endpoints."""
    pts = np.asarray(pts, dtype=np.float32)
    if pts.shape[0] < 2:
        return pts.copy()
    step = max(float(step), 1e-3)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cumulative[-1])
    if total < step:
        return np.stack([pts[0], pts[-1]]).astype(np.float32)
    n = max(int(round(total / step)) + 1, 2)
    targets = np.linspace(0.0, total, n)
    x = np.interp(targets, cumulative, pts[:, 0])
    y = np.interp(targets, cumulative, pts[:, 1])
    return np.stack([x, y], axis=1).astype(np.float32)


def smooth_polyline(pts: np.ndarray, window: int) -> np.ndarray:
    """Hann-window moving average with edge padding."""
    n = pts.shape[0]
    window = int(max(3, min(window, n if n % 2 == 1 else n - 1)))
    if window < 3 or n < 3:
        return pts.copy()
    if window % 2 == 0:
        window -= 1
    kernel = np.hanning(window + 2)[1:-1]
    kernel = kernel / kernel.sum()
    pad = window // 2
    out = np.empty_like(pts)
    for axis in range(2):
        padded = np.pad(pts[:, axis], (pad, pad), mode="edge")
        out[:, axis] = np.convolve(padded, kernel, mode="valid")
    return out.astype(np.float32)


def simplify_polyline(pts: np.ndarray, eps: float) -> np.ndarray:
    """Ramer-Douglas-Peucker: users cut corners."""
    if pts.shape[0] < 3 or eps <= 0:
        return pts.copy()
    approx = cv2.approxPolyDP(
        pts.reshape(-1, 1, 2).astype(np.float32), float(eps), False
    )
    out = approx.reshape(-1, 2).astype(np.float32)
    return out if out.shape[0] >= 2 else pts.copy()


def polyline_normals(pts: np.ndarray) -> np.ndarray:
    tangent = np.gradient(pts, axis=0)
    normals = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    return (normals / np.maximum(norm, 1e-6)).astype(np.float32)


def _fractal_noise(n: int, rng: np.random.Generator, octaves: int = 3) -> np.ndarray:
    """Smooth 1-D noise in roughly [-1, 1], used for wobble."""
    if n < 2:
        return np.zeros(n, np.float32)
    out = np.zeros(n, np.float32)
    for o in range(octaves):
        k = int(max(3, min(n, n // (2 ** (octaves - o)))))
        base = rng.normal(size=k).astype(np.float32)
        out += np.interp(
            np.linspace(0, 1, n), np.linspace(0, 1, k), base
        ) * (0.5**o)
    out -= out.mean()
    peak = float(np.abs(out).max())
    return (out / peak) if peak > 1e-6 else out


def deform_polyline(
    pts: np.ndarray,
    rng: np.random.Generator,
    params: BrushParams,
    normals: np.ndarray | None = None,
) -> np.ndarray:
    """Displace along the local normal: global offset + multi-scale wobble."""
    if pts.shape[0] < 2:
        return pts.copy()
    normals = polyline_normals(pts) if normals is None else normals

    n = pts.shape[0]
    offset = rng.normal(0.0, params.jitter_sigma)

    amp_lo, amp_hi = params.wobble_amp
    len_lo, len_hi = params.wobble_len
    disp = np.full(n, offset, dtype=np.float32)
    for _ in range(2):
        amp = rng.uniform(amp_lo, amp_hi)
        wavelength = rng.uniform(len_lo, len_hi)
        phase = rng.uniform(0, 2 * np.pi)
        t = np.arange(n, dtype=np.float32)
        disp += amp * np.sin(2 * np.pi * t / max(wavelength / 6.0, 1.0) + phase)

    disp += _fractal_noise(n, rng) * rng.uniform(0.8, 3.5)
    disp = np.clip(disp, -params.max_offset, params.max_offset)

    return (pts + normals * disp[:, None]).astype(np.float32)


# --------------------------------------------------------------------------- #
# Arc selection
# --------------------------------------------------------------------------- #


def _arc_slices(
    n: int, coverage: float, n_arcs: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """Indices of 1..n_arcs disjoint arcs covering roughly ``coverage`` of n."""
    n_arcs = int(max(1, min(n_arcs, 3)))
    coverage = float(np.clip(coverage, 0.05, 1.0))
    per = coverage / n_arcs
    # Keep arcs separated so they do not overlap into a continuous blob.
    gap = max(2, int((1.0 - coverage) * n / (n_arcs + 1)))
    cursor = int(rng.integers(0, max(1, n // 4)))
    out: list[np.ndarray] = []
    for i in range(n_arcs):
        length = int(max(4, round(per * n)))
        if i == n_arcs - 1 and coverage >= 0.95:
            length = n
        start = (cursor + i * gap) % n
        idx = (start + np.arange(length)) % n
        out.append(idx)
    return out


# --------------------------------------------------------------------------- #
# Rasterisation
# --------------------------------------------------------------------------- #


def _radii_profile(
    n: int, rng: np.random.Generator, params: BrushParams
) -> np.ndarray:
    base = rng.uniform(params.radius_min, params.radius_max)
    period = rng.uniform(18.0, 55.0)
    phase = rng.uniform(0, 2 * np.pi)
    t = np.arange(n, dtype=np.float32)
    modulation = 1.0 + 0.25 * np.sin(2 * np.pi * t / period + phase)
    return np.clip(base * modulation, 1.0, 14.0).astype(np.float32)


def rasterize_polyline(
    shape: tuple[int, int],
    pts: np.ndarray,
    radii: np.ndarray,
    rng: np.random.Generator | None = None,
    dry_brush_p: float = 0.0,
    line_type: int = cv2.LINE_AA,
) -> np.ndarray:
    """Draw a variable-width anti-aliased stroke onto a uint8 canvas."""
    canvas = np.zeros(shape, np.uint8)
    n = pts.shape[0]
    if n == 0:
        return canvas

    radii = np.asarray(radii, dtype=np.float32)
    if radii.shape[0] != n:
        radii = np.full(n, float(radii.reshape(-1)[0]) if radii.size else 2.0)

    skip = np.zeros(max(n - 1, 0), dtype=bool)
    if rng is not None and dry_brush_p > 0 and n > 8:
        gaps = int(rng.integers(1, 4))
        for _ in range(gaps):
            start = int(rng.integers(0, max(1, n - 4)))
            skip[start : start + int(rng.integers(1, 4))] = True

    for i in range(n - 1):
        if skip[i]:
            continue
        p0 = (int(round(float(pts[i, 0]))), int(round(float(pts[i, 1]))))
        p1 = (int(round(float(pts[i + 1, 0]))), int(round(float(pts[i + 1, 1]))))
        thickness = int(max(1, round(float(radii[i]) * 2.0)))
        cv2.line(canvas, p0, p1, 255, thickness, lineType=line_type)
        cv2.circle(
            canvas,
            p0,
            int(max(1, round(float(radii[i])))),
            255,
            -1,
            lineType=line_type,
        )

    last = (int(round(float(pts[-1, 0]))), int(round(float(pts[-1, 1]))))
    cv2.circle(
        canvas, last, int(max(1, round(float(radii[-1])))), 255, -1, line_type
    )
    return canvas


# --------------------------------------------------------------------------- #
# Distractor and interior strokes
# --------------------------------------------------------------------------- #


def _distractor_stroke(
    rgb: np.ndarray,
    boundary: np.ndarray,
    rng: np.random.Generator,
    params: BrushParams,
) -> np.ndarray | None:
    """A stroke along a strong image gradient that is *not* the object edge."""
    if rgb is None:
        return None
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    # Do not place a distractor right on top of the true boundary.
    near_boundary = cv2.dilate(
        (boundary > 0).astype(np.uint8), np.ones((41, 41), np.uint8)
    )
    mag[near_boundary > 0] = 0.0
    mag[:8, :] = mag[-8:, :] = mag[:, :8] = mag[:, -8:] = 0.0

    flat = mag.reshape(-1)
    if float(flat.max()) <= 1e-3:
        return None
    k = max(1, int(flat.size * 0.001))
    top = np.argpartition(flat, -k)[-k:]
    seed = int(rng.choice(top))
    h, w = mag.shape
    y, x = divmod(seed, w)

    angle = rng.uniform(0, 2 * np.pi)
    length = rng.uniform(60, 140)
    step = 4.0
    pts = [(float(x), float(y))]
    for _ in range(int(length / step)):
        angle += rng.normal(0, 0.18)
        x += step * np.cos(angle)
        y += step * np.sin(angle)
        if not (2 <= x < w - 2 and 2 <= y < h - 2):
            break
        pts.append((x, y))
    if len(pts) < 6:
        return None

    arr = np.asarray(pts, np.float32)
    arr = smooth_polyline(arr, 9)
    radii = _radii_profile(arr.shape[0], rng, params) * 0.8
    return rasterize_polyline((h, w), arr, radii)


def _interior_stroke(
    mask: np.ndarray, rng: np.random.Generator, params: BrushParams
) -> np.ndarray | None:
    binary = ensure_binary(mask)
    eroded = cv2.erode(binary, np.ones((9, 9), np.uint8))
    ys, xs = np.nonzero(eroded)
    if ys.size < 50:
        return None

    idx = rng.integers(0, ys.size)
    x0, y0 = float(xs[idx]), float(ys[idx])
    angle = rng.uniform(0, 2 * np.pi)
    length = rng.uniform(40, 110)
    pts = [(x0, y0)]
    for _ in range(int(length / 5)):
        angle += rng.normal(0, 0.25)
        x0 += 5 * np.cos(angle)
        y0 += 5 * np.sin(angle)
        pts.append((x0, y0))

    arr = smooth_polyline(np.asarray(pts, np.float32), 7)
    radii = _radii_profile(arr.shape[0], rng, params) * 0.9
    return rasterize_polyline(binary.shape, arr, radii)


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #


def synthesize_brush(
    mask: np.ndarray,
    rgb: np.ndarray | None = None,
    params: BrushParams | None = None,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Generate one synthetic brush stroke for a binary mask.

    Returns a uint8 raster (0-255) the same size as ``mask``. This is the
    ``gt_brush`` artefact and the model's 4th input channel.

    Every contour is stroked end to end, so the brush covers the same boundary
    that :func:`clean_edge` draws. It is imprecise — simplified, smoothed,
    displaced along the normal, variable width — but it is not *partial*: an
    input that described only part of the object while the target described all
    of it would be asking the model to invent the rest.
    """
    params = params or BrushParams()
    if isinstance(rng, np.random.Generator):
        gen = rng
    else:
        gen = np.random.default_rng(rng)

    binary = ensure_binary(mask)
    h, w = binary.shape
    out = np.zeros((h, w), np.uint8)

    if gen.random() < params.p_no_brush:
        return out

    paths = mask_to_paths(binary)
    if not paths:
        return out

    lo, hi = params.coverage
    coverage = float(gen.uniform(lo, hi))
    if gen.random() < params.p_partial:
        coverage = float(gen.uniform(0.12, 0.32))
    n_arcs = 1
    if gen.random() < params.p_two_arcs:
        n_arcs = 2

    eps = float(gen.uniform(*params.simplify_eps))
    window = int(gen.integers(params.smooth_window[0], params.smooth_window[1] + 1))

    for path in paths:
        # Rotate the start point so the contour seam never sits at a fixed place.
        path = np.roll(path, int(gen.integers(0, path.shape[0])), axis=0)

        closed = np.vstack([path, path[:1]])
        resampled = resample_polyline(closed, step=6.0)
        if resampled.shape[0] < 8:
            continue
        # Drop the duplicated closing point so arc wrapping behaves.
        resampled = resampled[:-1]

        n = resampled.shape[0]
        for idx in _arc_slices(n, coverage, n_arcs, gen):
            arc = resampled[idx]
            if arc.shape[0] < 4:
                continue
            arc = simplify_polyline(arc, eps)
            arc = resample_polyline(arc, step=6.0)
            if arc.shape[0] < 3:
                continue
            arc = smooth_polyline(arc, window)
            arc = deform_polyline(arc, gen, params)
            if len(idx) == n:
                # Full ring: close the polyline so the rasteriser draws the last
                # segment and the loop has no seam gap. Deformation is smooth,
                # so the closing segment stays short.
                arc = np.vstack([arc, arc[:1]])

            radii = _radii_profile(arc.shape[0], gen, params)
            dry = params.p_dry_brush if gen.random() < 0.5 else 0.0
            layer = rasterize_polyline((h, w), arc, radii, rng=gen, dry_brush_p=dry)
            out = np.maximum(out, layer)

    if gen.random() < params.p_distractor:
        layer = _distractor_stroke(rgb, boundary_from_mask(binary), gen, params)
        if layer is not None:
            out = np.maximum(out, layer)

    if gen.random() < params.p_interior:
        layer = _interior_stroke(binary, gen, params)
        if layer is not None:
            out = np.maximum(out, layer)

    return out


def encode_brush(brush: np.ndarray, encoding: str = "binary", clip_px: float = 16.0):
    """Convert a brush raster to the model's 4th channel."""
    if encoding == "sdf":
        return brush_distance(brush, clip_px=clip_px)
    if encoding != "binary":
        raise ValueError(f"unknown brush encoding: {encoding}")
    return (ensure_binary(brush).astype(np.float32) / 255.0)


__all__ = [
    "BrushParams",
    "ensure_binary",
    "boundary_from_mask",
    "clean_edge",
    "edge_band",
    "boundary_band",
    "brush_distance",
    "fill_holes",
    "mask_to_paths",
    "polyline_length",
    "resample_polyline",
    "smooth_polyline",
    "simplify_polyline",
    "polyline_normals",
    "deform_polyline",
    "rasterize_polyline",
    "synthesize_brush",
    "encode_brush",
]
