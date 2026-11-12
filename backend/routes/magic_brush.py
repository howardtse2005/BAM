"""Magic brush endpoints.

The magic brush is a two-step interaction:

1. ``POST /api/segment`` — the user drags a rough stroke along the object edge.
   The stroke *is* the request: the server takes its bounding box, grows it into
   the smallest square that holds the whole stroke, runs that one square through
   the UNet with the stroke itself, and returns the predicted **clean line**.
   One square, one forward pass — no tiling along the path and nothing to merge.
   Nothing is committed yet: the line replaces the wobbly stroke on screen so
   the user can judge it, re-stroke, or accept it.

   The stroke arrives in two widths. What the user draws and sees is a *band*;
   what the model is conditioned on is a thinner *prompt* stroke centred in it.
   The predicted edge is clipped to the band before it becomes a line, so the
   prediction can only appear under the stroke the user actually drew — a
   prediction that wanders off is trimmed rather than shown somewhere the user
   never brushed.

   A window larger than ``crop_size`` is downscaled to it rather than upsampled
   by the network: the model only ever saw 256 px crops, and a 600 px square
   would otherwise be a picture of the whole object. The stroke prompt shrinks
   with it, which keeps it inside the 3-12 px range it was trained on.

2. ``POST /api/fill`` — once the line is accepted and encloses the object, the
   enclosed region is flooded into a filled mask.

``/api/magic_brush/predict_crops`` and ``/api/magic_brush/predict_single_crop``
keep the reference repository's request/response shape for clients that send
crop images instead of a stroke. They return the raw edge mask, not a line.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
from flask import Blueprint, request

from ..deps import cfg, fail, index, ok, registry
from ..inference.crop_window import CropWindow, stroke_window
from ..inference.edge_to_mask import (
    edge_to_mask,
    line_is_closed,
    line_to_mask,
    ribbon,
    stroke_band,
)
from ..inference.merger import (
    CropPlacement,
    PredictionMerger,
    apply_morphological_filtering,
)
from ..inference.registry import ModelUnavailable
from ..schemas import (
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


def _window_patch(
    array: np.ndarray, window: CropWindow, border: int, fill: int = 0
) -> np.ndarray:
    """Cut the window out of an array, padding whatever falls outside it.

    A stroke that runs along the frame edge puts part of its square outside the
    image. The RGB is padded by replicating the border, so the model sees more
    of the same edge instead of a black wall it would happily predict around;
    the stroke is padded with zeros, because a stroke that was not drawn there
    must not become a prompt there.
    """
    height, width = array.shape[:2]
    top = max(0, -window.y0)
    left = max(0, -window.x0)
    bottom = max(0, window.y0 + window.size - height)
    right = max(0, window.x0 + window.size - width)
    if top or left or bottom or right:
        array = cv2.copyMakeBorder(array, top, bottom, left, right, border, value=fill)
    x = window.x0 + left
    y = window.y0 + top
    return np.ascontiguousarray(array[y : y + window.size, x : x + window.size])


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

    params = req.params
    try:
        window = stroke_window(
            req.stroke,
            # The window has to hold the band, not just the prompt inside it:
            # the band is what the user drew, and a window that cuts it would
            # clip the prediction before it was ever made.
            pad=max(2, int(round(params["brush_width"] / 2.0))),
            min_side=cfg().crop_min,
            max_side=cfg().crop_size,
        )
    except ValueError:
        return fail("stroke_mask is empty — draw a stroke first", 400)

    rgb_patch = _window_patch(rgb, window, cv2.BORDER_REPLICATE)
    stroke_patch = _window_patch(req.stroke, window, cv2.BORDER_CONSTANT)
    if window.downscaled:
        target = (window.model_size, window.model_size)
        rgb_patch = cv2.resize(rgb_patch, target, interpolation=cv2.INTER_AREA)
        stroke_patch = cv2.resize(stroke_patch, target, interpolation=cv2.INTER_AREA)
        # Keep every pixel the stroke touched: a 3 px prompt shrunk six times
        # survives as partial coverage, and thresholding that away would delete
        # the prompt the whole prediction is conditioned on.
        stroke_patch = ((stroke_patch > 0).astype(np.uint8)) * 255

    started = time.perf_counter()
    try:
        probabilities = registry().predict([rgb_patch], [stroke_patch])
    except ModelUnavailable as exc:
        return fail(str(exc), 503)
    except Exception as exc:
        return fail(f"inference failed: {exc}", 500)
    infer_ms = (time.perf_counter() - started) * 1000.0

    post_started = time.perf_counter()
    probability = probabilities[0]
    if probability.ndim == 3:
        probability = probability[0]
    if window.downscaled:
        probability = cv2.resize(
            probability, (window.size, window.size), interpolation=cv2.INTER_LINEAR
        )

    height, width = rgb.shape[:2]
    placement = CropPlacement(window.x0, window.y0, window.size)
    merger = _merge(
        [placement],
        [probability],
        [_validity(placement, height, width)],
        height,
        width,
        params,
    )
    merged = merger.finalize(
        threshold=params["threshold"],
        sensitivity=params["sensitivity"],
        apply_morphology=False,
    )

    # Clip the prediction to the brush the user actually drew. The model sees the
    # thin prompt stroke, so its edge can sit a few px outside it; the band is
    # what was on screen, and a line drawn outside the stroke is a line the user
    # cannot attribute to anything. Insetting by the ribbon radius keeps the
    # painted line inside the band, not just the skeleton it grew from.
    band = stroke_band(
        req.stroke,
        params["brush_width"],
        params["prompt_width"],
        inset=params["line_width"] // 2,
    )
    predicted = merged.edge > float(params["edge_threshold"])
    clipped = int(np.count_nonzero(predicted & (band == 0)))
    edge_prob = merged.edge * (band > 0)

    fill = edge_to_mask(
        edge_prob,
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
        crop_size=window.model_size,
        window={"x0": window.x0, "y0": window.y0, "size": window.size},
        scale=round(window.scale, 4),
        band_width=params["brush_width"],
        prompt_width=params["prompt_width"],
        canvas={"width": req.canvas_width, "height": req.canvas_height},
        stats={
            "num_crops": 1,
            "window_px": window.size,
            "ms_infer": round(infer_ms, 1),
            "ms_post": round(post_ms, 1),
            "ms_total": round(infer_ms + post_ms, 1),
            "backend": registry().meta.get("backend", "unknown"),
            "coverage": round(merged.covered / max(1, height * width), 4),
            "predicted_px": int(predicted.sum()),
            "clipped_px": clipped,
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
