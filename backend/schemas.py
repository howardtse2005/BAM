"""Request parsing and validation.

Hand-rolled instead of pydantic: the payload is small, the error messages need to
be actionable in the UI, and adding a dependency for six fields is not worth it.
Every numeric field is clamped to a safe range rather than trusted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .services.storage import StorageError, decode_data_url, decode_mask_bytes

SAFE_NAME = re.compile(r"^[A-Za-z0-9._\- ]{1,128}$")


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
    image: str
    canvas_width: int
    canvas_height: int
    crops: list[CropSpec]
    stroke: np.ndarray
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
    if not isinstance(value, str) or not SAFE_NAME.match(value):
        raise ValidationError(f"invalid {field_name}")
    if ".." in value or value.startswith("."):
        raise ValidationError(f"invalid {field_name}")
    return value


def parse_params(raw: dict | None, cfg) -> dict[str, Any]:
    raw = raw or {}
    infer = cfg.infer
    mode = str(raw.get("mode", infer.mode))
    if mode not in ("object", "line"):
        mode = "object"
    weighting = str(raw.get("merge_weighting", infer.merge_weighting))
    if weighting not in ("uniform", "gaussian"):
        weighting = "uniform"

    return {
        "sensitivity": _clamp(raw.get("sensitivity", infer.sensitivity), 1.0, 3.0, 1.0),
        "threshold": _clamp(raw.get("threshold", infer.threshold), 0.05, 0.95, 0.5),
        "edge_threshold": _clamp(
            raw.get("edge_threshold", infer.edge_threshold), 0.05, 0.95, 0.45
        ),
        "expand": int(_clamp(raw.get("expand", infer.expand), -16, 16, 0)),
        "smooth": int(_clamp(raw.get("smooth", infer.smooth), 0, 4, 2)),
        "mode": mode,
        "merge_weighting": weighting,
        "apply_morphology": bool(raw.get("apply_morphology", infer.apply_morphology)),
        "morph_kernel_size": int(
            _clamp(raw.get("morph_kernel_size", infer.morph_kernel_size), 1, 15, 3)
        ),
        "morph_iterations": int(
            _clamp(raw.get("morph_iterations", infer.morph_iterations), 1, 8, 2)
        ),
        "apply_dbscan": bool(raw.get("apply_dbscan", infer.apply_dbscan)),
    }


def parse_segment(payload: dict, cfg) -> SegmentRequest:
    if not isinstance(payload, dict):
        raise ValidationError("body must be a JSON object")

    dataset = safe_name(_require(payload, "dataset", str), "dataset")
    image = safe_name(_require(payload, "image", str), "image")

    canvas = payload.get("canvas") or {}
    if not isinstance(canvas, dict):
        raise ValidationError("canvas must be an object")
    width = int(_clamp(canvas.get("width", 0), 1, cfg.server.max_canvas_side, 0))
    height = int(_clamp(canvas.get("height", 0), 1, cfg.server.max_canvas_side, 0))
    if width <= 0 or height <= 0:
        raise ValidationError("canvas dimensions are required")

    raw_crops = payload.get("crops")
    if not isinstance(raw_crops, list) or not raw_crops:
        raise ValidationError("crops must be a non-empty list")
    if len(raw_crops) > cfg.infer.max_crops:
        raise ValidationError(f"too many crops (max {cfg.infer.max_crops})")

    crops: list[CropSpec] = []
    for item in raw_crops:
        if not isinstance(item, dict):
            raise ValidationError("each crop must be an object")
        size = int(_clamp(item.get("size", cfg.infer.crop_size), 32, 512, cfg.infer.crop_size))
        if size != cfg.infer.crop_size:
            raise ValidationError(
                f"crop size must be {cfg.infer.crop_size}, got {size}"
            )
        crops.append(
            CropSpec(
                x0=int(_clamp(item.get("x0", 0), -4096, 4096, 0)),
                y0=int(_clamp(item.get("y0", 0), -4096, 4096, 0)),
                size=size,
            )
        )

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
        image=image,
        canvas_width=width,
        canvas_height=height,
        crops=crops,
        stroke=stroke,
        params=parse_params(payload.get("params"), cfg),
    )


def parse_legacy_crops(payload: dict, cfg) -> tuple[list[LegacyCrop], dict, int, int]:
    """Parse the masker-compatible ``/magic_pen/predict_crops`` body."""
    if not isinstance(payload, dict):
        raise ValidationError("body must be a JSON object")

    raw_crops = payload.get("crops")
    if not isinstance(raw_crops, list) or not raw_crops:
        raise ValidationError("crops must be a non-empty list")
    if len(raw_crops) > cfg.infer.max_crops:
        raise ValidationError(f"too many crops (max {cfg.infer.max_crops})")

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
            "sensitivity": payload.get("sensitivity", cfg.infer.sensitivity),
            "threshold": payload.get("threshold", cfg.infer.threshold),
            "apply_morphology": payload.get("apply_morphology", True),
            "morph_kernel_size": payload.get("morph_kernel_size", 3),
            "morph_iterations": payload.get("morph_iterations", 2),
            "apply_dbscan": payload.get("apply_dbscan", False),
            "mode": payload.get("mode", cfg.infer.mode),
        },
        cfg,
    )
    return crops, params, canvas_width, canvas_height


def pseudo_stroke(rgb: np.ndarray) -> np.ndarray:
    """Weak stroke estimate from image gradients.

    Only used by the legacy endpoint when the caller does not send a stroke
    channel. It is a degraded mode: the response sets ``degraded=true`` so the
    caller knows the model was not given the real pen path.
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
    "LegacyCrop",
    "parse_segment",
    "parse_legacy_crops",
    "parse_params",
    "pseudo_stroke",
    "safe_name",
]
