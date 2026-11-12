"""Request parsing and validation.

Hand-rolled instead of pydantic: the payload is small, the error messages need to
be actionable in the UI, and every numeric field is clamped rather than trusted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .services.storage import StorageError, decode_data_url, decode_mask_bytes

FORBIDDEN_CHARS = set("/\\\x00")
MAX_NAME = 200
VALID_SPLITS = {"train", "val", "test"}


class ValidationError(ValueError):
    """Client error; maps to HTTP 400."""


@dataclass
class CropSpec:
    x0: int
    y0: int
    size: int


@dataclass
class SegmentRequest:
    dataset: str
    split: str
    image: str
    canvas_width: int
    canvas_height: int
    crops: list[CropSpec]
    stroke: np.ndarray
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class FillRequest:
    dataset: str
    split: str
    image: str
    width: int
    height: int
    line: np.ndarray
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class LegacyCrop:
    image: np.ndarray  # RGB uint8
    x0: int
    y0: int
    size: int


def _require(payload: dict, key: str, kind: type):
    if key not in payload:
        raise ValidationError(f"missing field: {key}")
    value = payload[key]
    if not isinstance(value, kind):
        raise ValidationError(f"field {key} must be {kind.__name__}")
    return value


def _clamp(value: Any, low: float, high: float, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(number):
        return default
    return float(min(high, max(low, number)))


def safe_name(value: str, field_name: str) -> str:
    """Reject path traversal without rejecting legal filename characters.

    An allowlist would break real datasets: DIS5K filenames contain ``#``, and
    ``#`` also starts a URL fragment, so it must survive to the route.
    """
    if not isinstance(value, str) or not value:
        raise ValidationError(f"invalid {field_name}")
    if len(value) > MAX_NAME:
        raise ValidationError(f"{field_name} is too long")
    if any(ch in FORBIDDEN_CHARS for ch in value):
        raise ValidationError(f"invalid {field_name}")
    if ".." in value or value.startswith("."):
        raise ValidationError(f"invalid {field_name}")
    return value


def safe_split(value: Any) -> str:
    split = str(value or "train").strip().lower()
    if split not in VALID_SPLITS:
        raise ValidationError(f"invalid split: {value!r}")
    return split


def parse_params(raw: dict | None, cfg) -> dict[str, Any]:
    raw = raw or {}
    mode = str(raw.get("mode", cfg.mode))
    if mode not in ("object", "line"):
        mode = "object"
    weighting = str(raw.get("merge_weighting", cfg.merge_weighting))
    if weighting not in ("uniform", "gaussian"):
        weighting = "uniform"

    # The visible band and the prompt stroke inside it. The client sends both
    # because it is the side that painted them; the server only checks that the
    # prompt is a real stroke inside the band it is supposed to sit in.
    band = int(
        _clamp(
            raw.get("brush_width", cfg.magic_brush_width),
            4,
            160,
            cfg.magic_brush_width,
        )
    )
    default_prompt = cfg.prompt_width_for(band)
    prompt = int(_clamp(raw.get("prompt_width", default_prompt), 1, 64, default_prompt))
    prompt = max(1, min(prompt, band))

    return {
        "sensitivity": _clamp(raw.get("sensitivity", cfg.sensitivity), 1.0, 3.0, 1.0),
        "threshold": _clamp(raw.get("threshold", cfg.threshold), 0.05, 0.95, 0.5),
        "edge_threshold": _clamp(raw.get("edge_threshold", cfg.edge_threshold), 0.05, 0.95, 0.45),
        "expand": int(_clamp(raw.get("expand", cfg.expand), -16, 16, 0)),
        "smooth": int(_clamp(raw.get("smooth", cfg.smooth), 0, 4, 2)),
        "mode": mode,
        "line_width": int(_clamp(raw.get("line_width", cfg.line_width), 1, 32, cfg.line_width)),
        "fill_close": int(_clamp(raw.get("fill_close", cfg.fill_close), 0, 16, cfg.fill_close)),
        "brush_width": band,
        "prompt_width": prompt,
        "merge_weighting": weighting,
        "apply_morphology": bool(raw.get("apply_morphology", cfg.apply_morphology)),
        "morph_kernel_size": int(_clamp(raw.get("morph_kernel_size", cfg.morph_kernel), 1, 15, 3)),
        "morph_iterations": int(
            _clamp(raw.get("morph_iterations", cfg.morph_iterations), 1, 8, 2)
        ),
        "apply_dbscan": bool(raw.get("apply_dbscan", cfg.apply_dbscan)),
    }


def _parse_crop_size(raw: Any, cfg) -> int:
    """Validate an inference crop size.

    The crop is not fixed at 256: the client picks it from the canvas size so a
    crop always covers the same fraction of the object. It must stay a multiple
    of 32 (four pooling stages) and inside ``[crop_min, crop_size]``.
    """
    size = int(_clamp(raw, 32, 512, cfg.crop_size))
    if size % 32 != 0:
        raise ValidationError(f"crop size must be a multiple of 32, got {size}")
    if not cfg.crop_min <= size <= cfg.crop_size:
        raise ValidationError(
            f"crop size must be in [{cfg.crop_min}, {cfg.crop_size}], got {size}"
        )
    return size


def parse_segment(payload: dict, cfg) -> SegmentRequest:
    if not isinstance(payload, dict):
        raise ValidationError("body must be a JSON object")

    dataset = safe_name(_require(payload, "dataset", str), "dataset")
    split = safe_split(payload.get("split"))
    image = safe_name(_require(payload, "image", str), "image")

    canvas = payload.get("canvas") or {}
    if not isinstance(canvas, dict):
        raise ValidationError("canvas must be an object")
    width = int(_clamp(canvas.get("width", 0), 1, cfg.max_canvas_side, 0))
    height = int(_clamp(canvas.get("height", 0), 1, cfg.max_canvas_side, 0))
    if width <= 0 or height <= 0:
        raise ValidationError("canvas dimensions are required")

    raw_crops = payload.get("crops")
    if not isinstance(raw_crops, list) or not raw_crops:
        raise ValidationError("crops must be a non-empty list")
    if len(raw_crops) > cfg.max_crops:
        raise ValidationError(f"too many crops (max {cfg.max_crops})")

    crops: list[CropSpec] = []
    for item in raw_crops:
        if not isinstance(item, dict):
            raise ValidationError("each crop must be an object")
        size = _parse_crop_size(item.get("size", cfg.crop_size), cfg)
        crops.append(
            CropSpec(
                x0=int(_clamp(item.get("x0", 0), -4096, 4096, 0)),
                y0=int(_clamp(item.get("y0", 0), -4096, 4096, 0)),
                size=size,
            )
        )

    sizes = {crop.size for crop in crops}
    if len(sizes) != 1:
        raise ValidationError("all crops must share one size")

    stroke_b64 = payload.get("stroke_mask")
    if not isinstance(stroke_b64, str) or not stroke_b64:
        raise ValidationError("stroke_mask is required")
    try:
        stroke = decode_mask_bytes(decode_data_url(stroke_b64))
    except StorageError as exc:
        raise ValidationError(f"stroke_mask: {exc}") from exc

    if stroke.shape[:2] != (height, width):
        stroke = cv2.resize(stroke, (width, height), interpolation=cv2.INTER_NEAREST)

    return SegmentRequest(
        dataset=dataset,
        split=split,
        image=image,
        canvas_width=width,
        canvas_height=height,
        crops=crops,
        stroke=stroke,
        params=parse_params(payload.get("params"), cfg),
    )


def parse_fill(payload: dict, cfg) -> FillRequest:
    """Parse the body of ``/api/fill``: an accepted line plus fill knobs."""
    if not isinstance(payload, dict):
        raise ValidationError("body must be a JSON object")

    dataset = safe_name(_require(payload, "dataset", str), "dataset")
    split = safe_split(payload.get("split"))
    image = safe_name(_require(payload, "image", str), "image")

    canvas = payload.get("canvas") or {}
    if not isinstance(canvas, dict):
        raise ValidationError("canvas must be an object")
    width = int(_clamp(canvas.get("width", 0), 1, cfg.max_canvas_side, 0))
    height = int(_clamp(canvas.get("height", 0), 1, cfg.max_canvas_side, 0))
    if width <= 0 or height <= 0:
        raise ValidationError("canvas dimensions are required")

    line_b64 = payload.get("line_mask")
    if not isinstance(line_b64, str) or not line_b64:
        raise ValidationError("line_mask is required")
    try:
        line = decode_mask_bytes(decode_data_url(line_b64))
    except StorageError as exc:
        raise ValidationError(f"line_mask: {exc}") from exc

    if line.shape[:2] != (height, width):
        line = cv2.resize(line, (width, height), interpolation=cv2.INTER_NEAREST)

    return FillRequest(
        dataset=dataset,
        split=split,
        image=image,
        width=width,
        height=height,
        line=line,
        params=parse_params(payload.get("params"), cfg),
    )


def parse_legacy_crops(payload: dict, cfg) -> tuple[list[LegacyCrop], dict, int, int]:
    """Parse the reference client's ``predict_crops`` body."""
    if not isinstance(payload, dict):
        raise ValidationError("body must be a JSON object")

    raw_crops = payload.get("crops")
    if not isinstance(raw_crops, list) or not raw_crops:
        raise ValidationError("crops must be a non-empty list")
    if len(raw_crops) > cfg.max_crops:
        raise ValidationError(f"too many crops (max {cfg.max_crops})")

    crops: list[LegacyCrop] = []
    for item in raw_crops:
        if not isinstance(item, dict):
            raise ValidationError("each crop must be an object")
        data = item.get("image_base64")
        if not isinstance(data, str) or not data:
            raise ValidationError("each crop needs image_base64")
        try:
            buffer = np.frombuffer(decode_data_url(data), np.uint8)
            bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        except StorageError as exc:
            raise ValidationError(f"crop image: {exc}") from exc
        if bgr is None:
            raise ValidationError("crop image could not be decoded")

        width = int(item.get("width", bgr.shape[1]))
        height = int(item.get("height", bgr.shape[0]))
        if (width, height) != (bgr.shape[1], bgr.shape[0]):
            bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)

        center_x = int(_clamp(item.get("centerX", 0), -8192, 8192, 0))
        center_y = int(_clamp(item.get("centerY", 0), -8192, 8192, 0))
        crops.append(
            LegacyCrop(
                image=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                x0=center_x - width // 2,
                y0=center_y - height // 2,
                size=max(width, height),
            )
        )

    canvas_width = int(_clamp(raw_crops[0].get("canvas_width", 0), 1, 8192, 0))
    canvas_height = int(_clamp(raw_crops[0].get("canvas_height", 0), 1, 8192, 0))
    if canvas_width <= 0 or canvas_height <= 0:
        raise ValidationError("canvas_width and canvas_height are required")

    params = parse_params(
        {
            "sensitivity": payload.get("sensitivity", cfg.sensitivity),
            "threshold": payload.get("threshold", cfg.threshold),
            "apply_morphology": payload.get("apply_morphology", True),
            "morph_kernel_size": payload.get("morph_kernel_size", 3),
            "morph_iterations": payload.get("morph_iterations", 2),
            "apply_dbscan": payload.get("apply_dbscan", False),
            "mode": payload.get("mode", cfg.mode),
        },
        cfg,
    )
    return crops, params, canvas_width, canvas_height


def pseudo_stroke(rgb: np.ndarray) -> np.ndarray:
    """Weak stroke estimate from image gradients.

    Only used by the legacy endpoint when the caller does not send a stroke
    channel. Degraded mode: the response sets ``degraded=true``.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    peak = float(magnitude.max())
    if peak <= 1e-3:
        return np.zeros(gray.shape, np.uint8)
    normalized = magnitude / peak
    threshold = float(np.percentile(normalized, 88))
    return ((normalized >= threshold).astype(np.uint8)) * 255


__all__ = [
    "ValidationError",
    "CropSpec",
    "SegmentRequest",
    "FillRequest",
    "LegacyCrop",
    "parse_segment",
    "parse_fill",
    "parse_legacy_crops",
    "parse_params",
    "pseudo_stroke",
    "safe_name",
    "safe_split",
]
