"""Magic brush endpoints.

The magic brush is a two-step interaction:

1. ``POST /api/segment`` — the user drags a rough stroke along the object edge.
   The server slices small square crops across that stroke, batches them through
   the UNet with the stroke itself, merges the per-crop predictions, and returns
   the predicted **clean line**. Nothing is committed yet: the line replaces the
   wobbly stroke on screen so the user can judge it, re-stroke, or accept it.

2. ``POST /api/fill`` — once the line is accepted and encloses the object, the
   enclosed region is flooded into a filled mask.

``/api/magic_brush/predict_crops`` and ``/api/magic_brush/predict_single_crop``
keep the reference repository's request/response shape for clients that send
crop images instead of crop boxes. They return the raw edge mask, not a line.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
from flask import Blueprint, request

from ..deps import cfg, fail, index, ok, registry
from ..inference.edge_to_mask import edge_to_mask, line_is_closed, line_to_mask, ribbon
from ..inference.merger import (
    CropPlacement,
    PredictionMerger,
    apply_morphological_filtering,
)
from ..inference.registry import ModelUnavailable
from ..schemas import (
    CropSpec,
    ValidationError,
    parse_fill,
    parse_legacy_crops,
    parse_segment,
    pseudo_stroke,
)
from ..services.storage import StorageError, encode_png_base64

bp = Blueprint("magic_brush", __name__)


# --------------------------------------------------------------------------- #
# Crop extraction
# --------------------------------------------------------------------------- #


def _validity(placement: CropPlacement, height: int, width: int) -> np.ndarray:
    valid = np.zeros((placement.size, placement.size), np.float32)
    x0 = max(0, placement.x0)
    y0 = max(0, placement.y0)
    x1 = min(width, placement.x1)
    y1 = min(height, placement.y1)
    if x1 > x0 and y1 > y0:
        valid[
            y0 - placement.y0 : y1 - placement.y0,
            x0 - placement.x0 : x1 - placement.x0,
        ] = 1.0
    return valid


def _slice_crops(
    rgb: np.ndarray, stroke: np.ndarray, crops: list[CropSpec]
) -> tuple[list[np.ndarray], list[np.ndarray], list[CropPlacement], list[np.ndarray]]:
    """Slice crops from replicate-padded arrays so edges never get fake black.

    The RGB patch and the stroke patch are cut with identical integer offsets,
    which is what keeps the 4th channel pixel-aligned with the image.
    """
    size = crops[0].size
    half = size // 2

    rgb_padded = cv2.copyMakeBorder(rgb, half, half, half, half, cv2.BORDER_REPLICATE)
    stroke_padded = cv2.copyMakeBorder(
        stroke, half, half, half, half, cv2.BORDER_CONSTANT, value=0
    )

    height, width = rgb.shape[:2]
    rgb_crops: list[np.ndarray] = []
    stroke_crops: list[np.ndarray] = []
    placements: list[CropPlacement] = []
    valids: list[np.ndarray] = []

    for crop in crops:
        x = crop.x0 + half
        y = crop.y0 + half
        patch = rgb_padded[y : y + size, x : x + size]
        stroke_patch = stroke_padded[y : y + size, x : x + size]
        if patch.shape[:2] != (size, size):
            continue
        rgb_crops.append(np.ascontiguousarray(patch))
        stroke_crops.append(np.ascontiguousarray(stroke_patch))
        placement = CropPlacement(crop.x0, crop.y0, size)
        placements.append(placement)
        valids.append(_validity(placement, height, width))

    return rgb_crops, stroke_crops, placements, valids


def _merge(
    placements: list[CropPlacement],
    probabilities: np.ndarray,
    valids: list[np.ndarray],
    height: int,
    width: int,
    params: dict,
) -> PredictionMerger:
    merger = PredictionMerger(
        height, width, weighting=params.get("merge_weighting", "uniform")
    )
    for placement, prob, valid in zip(placements, probabilities, valids):
        merger.add(placement, prob[0] if prob.ndim == 3 else prob, valid)
    return merger


def _load_canvas_rgb(req) -> np.ndarray:
    """Load the stored image resized to the canvas the client is drawing on."""
    rgb = index().load_rgb(req.dataset, req.split, req.image)
    target = (req.canvas_height, req.canvas_width)
    if rgb.shape[:2] != target:
        rgb = cv2.resize(rgb, (target[1], target[0]), interpolation=cv2.INTER_AREA)
    return rgb


# --------------------------------------------------------------------------- #
# /api/segment — stroke -> predicted line
# --------------------------------------------------------------------------- #


@bp.post("/api/segment")
def segment():
    payload = request.get_json(silent=True)
    try:
        req = parse_segment(payload, cfg())
    except ValidationError as exc:
        return fail(exc, 400)

    try:
        rgb = _load_canvas_rgb(req)
    except KeyError as exc:
        return fail(exc, 404)
    except Exception as exc:
        return fail(f"could not load image: {exc}", 500)

    rgb_crops, stroke_crops, placements, valids = _slice_crops(rgb, req.stroke, req.crops)
    if not rgb_crops:
        return fail("no valid crops after clamping", 400)

    started = time.perf_counter()
    try:
        probabilities = registry().predict(rgb_crops, stroke_crops)
    except ModelUnavailable as exc:
        return fail(str(exc), 503)
    except Exception as exc:
        return fail(f"inference failed: {exc}", 500)
    infer_ms = (time.perf_counter() - started) * 1000.0

    post_started = time.perf_counter()
    params = req.params
    merger = _merge(
        placements, probabilities, valids, rgb.shape[0], rgb.shape[1], params
    )
    merged = merger.finalize(
        threshold=params["threshold"],
        sensitivity=params["sensitivity"],
        apply_morphology=False,
    )

    fill = edge_to_mask(
        merged.edge,
        req.stroke,
        rgb=rgb,
        edge_threshold=params["edge_threshold"],
        threshold=params["threshold"],
        expand=params["expand"],
        smooth=params["smooth"],
        mode=params["mode"],
        line_width=params["line_width"],
    )

    if params["mode"] == "line":
        line = fill.mask
        filled = None
    else:
        # The line is always available, even when a filled mask was requested:
        # the UI shows it while the region is being judged.
        line = ribbon(fill.edge, params["line_width"])
        filled = fill.mask
        if params["apply_morphology"] and filled.any():
            filled = apply_morphological_filtering(
                filled, params["morph_kernel_size"], params["morph_iterations"]
            )

    closed = line_is_closed(line, gap=params["fill_close"])
    post_ms = (time.perf_counter() - post_started) * 1000.0

    return ok(
        line_base64=encode_png_base64(line),
        mask_base64=encode_png_base64(filled) if filled is not None else None,
        edge_base64=encode_png_base64(fill.edge),
        closed=closed,
        fillable=bool(closed and params["mode"] == "line"),
        confidence=fill.confidence,
        low_confidence=bool(fill.low_confidence),
        mode_used=fill.mode,
        crop_size=int(req.crops[0].size),
        canvas={"width": req.canvas_width, "height": req.canvas_height},
        stats={
            "num_crops": len(rgb_crops),
            "requested_crops": len(req.crops),
            "ms_infer": round(infer_ms, 1),
            "ms_post": round(post_ms, 1),
            "ms_total": round(infer_ms + post_ms, 1),
            "backend": registry().meta.get("backend", "unknown"),
            "coverage": round(merged.covered / max(1, rgb.shape[0] * rgb.shape[1]), 4),
        },
    )


# --------------------------------------------------------------------------- #
# /api/fill — accepted line -> filled region
# --------------------------------------------------------------------------- #


@bp.post("/api/fill")
def fill():
    payload = request.get_json(silent=True)
    try:
        req = parse_fill(payload, cfg())
    except ValidationError as exc:
        return fail(exc, 400)

    try:
        rgb = index().load_rgb(req.dataset, req.split, req.image)
    except KeyError as exc:
        return fail(exc, 404)
    except Exception as exc:
        return fail(f"could not load image: {exc}", 500)

    if rgb.shape[:2] != (req.height, req.width):
        rgb = cv2.resize(rgb, (req.width, req.height), interpolation=cv2.INTER_AREA)

    params = req.params
    started = time.perf_counter()
    result = line_to_mask(
        req.line,
        rgb=rgb,
        close=params["fill_close"],
        smooth=params["smooth"],
        expand=params["expand"],
        threshold=params["threshold"],
    )
    elapsed = (time.perf_counter() - started) * 1000.0

    return ok(
        mask_base64=encode_png_base64(result.mask),
        grew=result.grew,
        added_px=result.added_px,
        mode_used=result.mode,
        confidence=result.confidence,
        canvas={"width": req.width, "height": req.height},
        stats={"ms_fill": round(elapsed, 1)},
    )


# --------------------------------------------------------------------------- #
# Reference-client endpoints
# --------------------------------------------------------------------------- #


@bp.post("/api/magic_brush/predict_crops")
def predict_crops():
    payload = request.get_json(silent=True)
    try:
        crops, params, canvas_width, canvas_height = parse_legacy_crops(payload, cfg())
    except ValidationError as exc:
        return fail(exc, 400)

    global_stroke = None
    raw_stroke = (payload or {}).get("stroke_mask")
    if isinstance(raw_stroke, str) and raw_stroke:
        from ..services.storage import decode_data_url, decode_mask_bytes

        try:
            global_stroke = decode_mask_bytes(decode_data_url(raw_stroke))
            global_stroke = cv2.resize(
                global_stroke,
                (canvas_width, canvas_height),
                interpolation=cv2.INTER_NEAREST,
            )
        except StorageError as exc:
            return fail(f"stroke_mask: {exc}", 400)

    degraded = global_stroke is None
    rgb_crops: list[np.ndarray] = []
    stroke_crops: list[np.ndarray] = []
    placements: list[CropPlacement] = []
    valids: list[np.ndarray] = []

    for crop in crops:
        rgb_crops.append(crop.image)
        if global_stroke is not None:
            half = crop.size // 2
            padded = cv2.copyMakeBorder(
                global_stroke, half, half, half, half, cv2.BORDER_CONSTANT, value=0
            )
            stroke_crops.append(
                padded[
                    crop.y0 + half : crop.y0 + half + crop.size,
                    crop.x0 + half : crop.x0 + half + crop.size,
                ]
            )
        else:
            stroke_crops.append(pseudo_stroke(crop.image))

        placement = CropPlacement(crop.x0, crop.y0, crop.size)
        placements.append(placement)
        valids.append(_validity(placement, canvas_height, canvas_width))

    started = time.perf_counter()
    try:
        probabilities = registry().predict(rgb_crops, stroke_crops)
    except ModelUnavailable as exc:
        return fail(str(exc), 503)
    except Exception as exc:
        return fail(f"inference failed: {exc}", 500)
    infer_ms = (time.perf_counter() - started) * 1000.0

    merger = _merge(
        placements, probabilities, valids, canvas_height, canvas_width, params
    )
    merged = merger.finalize(
        threshold=params["threshold"],
        sensitivity=params["sensitivity"],
        apply_morphology=params["apply_morphology"],
        morph_kernel_size=params["morph_kernel_size"],
        morph_iterations=params["morph_iterations"],
        apply_dbscan=params["apply_dbscan"],
    )

    return ok(
        message=f"processed {len(rgb_crops)} of {len(crops)} crops",
        merged_mask_base64=encode_png_base64(merged.mask),
        num_crops_processed=len(rgb_crops),
        canvas_dimensions=[canvas_width, canvas_height],
        degraded=degraded,
        sensitivity=params["sensitivity"],
        threshold=params["threshold"],
        stats={"ms_infer": round(infer_ms, 1), "coverage": merged.covered},
    )


@bp.post("/api/magic_brush/predict_single_crop")
def predict_single_crop():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return fail("body must be a JSON object", 400)

    from ..services.storage import decode_data_url

    raw = payload.get("image_base64")
    if not isinstance(raw, str) or not raw:
        return fail("image_base64 is required", 400)

    try:
        buffer = np.frombuffer(decode_data_url(raw), np.uint8)
        bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except StorageError as exc:
        return fail(exc, 400)
    if bgr is None:
        return fail("crop image could not be decoded", 400)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    stroke = pseudo_stroke(rgb)

    started = time.perf_counter()
    try:
        probabilities = registry().predict([rgb], [stroke])
    except ModelUnavailable as exc:
        return fail(str(exc), 503)
    except Exception as exc:
        return fail(f"inference failed: {exc}", 500)
    elapsed = (time.perf_counter() - started) * 1000.0

    prob = probabilities[0, 0]
    encoded = np.clip(prob * 255.0, 0, 255).astype(np.uint8)
    return ok(
        prediction_base64=encode_png_base64(encoded),
        crop_info={
            "width": int(rgb.shape[1]),
            "height": int(rgb.shape[0]),
            "centerX": payload.get("centerX"),
            "centerY": payload.get("centerY"),
        },
        stats={"ms": round(elapsed, 1), "degraded": True},
    )


__all__ = ["bp"]
