"""Mask load/save and image upload."""

from __future__ import annotations

from pathlib import Path

import cv2
from flask import Blueprint, request

from imgio import canonical_size, imwrite, resize

from ..deps import cfg, fail, index, ok
from ..schemas import ValidationError, safe_name
from ..services.storage import (
    StorageError,
    decode_data_url,
    decode_image_bytes,
    decode_mask_bytes,
    encode_png_base64,
    safe_join,
    sniff_image,
)

bp = Blueprint("masks_api", __name__)


def _require_json() -> dict:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ValidationError("body must be a JSON object")
    return payload


@bp.get("/api/mask/<dataset>/<image>")
def get_mask(dataset: str, image: str):
    try:
        safe_name(dataset, "dataset")
        safe_name(image, "image")
        path = index().mask_path(dataset, image)
    except ValidationError as exc:
        return fail(exc, 400)
    except KeyError as exc:
        return fail(exc, 404)

    if path is None:
        return fail("no mask for this image", 404)
    mask = index().load_mask(dataset, image)
    return ok(mask_base64=encode_png_base64(mask), path=str(path))


@bp.post("/api/mask/save")
def save_mask():
    try:
        payload = _require_json()
        dataset = safe_name(payload.get("dataset", ""), "dataset")
        image = safe_name(payload.get("image", ""), "image")
        raw = payload.get("mask_base64")
        if not isinstance(raw, str) or not raw:
            raise ValidationError("mask_base64 is required")
        mask = decode_mask_bytes(decode_data_url(raw))
    except ValidationError as exc:
        return fail(exc, 400)
    except StorageError as exc:
        return fail(exc, 400)

    try:
        rgb = index().load_rgb(dataset, image)
    except KeyError as exc:
        return fail(exc, 404)

    height, width = rgb.shape[:2]
    if mask.shape[:2] != (height, width):
        mask = resize(mask, width, height, is_mask=True)

    try:
        path = safe_join(cfg().dataset_root, dataset, "gt", f"{image}.png")
    except StorageError as exc:
        return fail(exc, 400)

    imwrite(path, mask)
    index().refresh(force=True)
    return ok(path=str(path), width=width, height=height)


@bp.post("/api/image/upload")
def upload_image():
    try:
        payload = _require_json()
        dataset = safe_name(payload.get("dataset", ""), "dataset")
        raw_name = payload.get("name", "")
        if not isinstance(raw_name, str) or not raw_name:
            raise ValidationError("name is required")
        stem = safe_name(Path(raw_name).stem, "name")
        raw = payload.get("image_base64")
        if not isinstance(raw, str) or not raw:
            raise ValidationError("image_base64 is required")

        data = decode_data_url(raw)
        sniff_image(data)
        bgr = decode_image_bytes(data)
    except ValidationError as exc:
        return fail(exc, 400)
    except StorageError as exc:
        return fail(exc, 400)

    height, width = bgr.shape[:2]
    target_w, target_h, scale = canonical_size(
        height, width, cfg().data.canonical_short_side
    )
    if (target_w, target_h) != (width, height):
        bgr = cv2.resize(bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)

    try:
        path = safe_join(cfg().dataset_root, dataset, "rgb", f"{stem}.jpg")
    except StorageError as exc:
        return fail(exc, 400)

    imwrite(path, bgr)
    index().refresh(force=True)
    return ok(
        dataset=dataset,
        image=stem,
        width=int(target_w),
        height=int(target_h),
        scale=round(float(scale), 6),
        path=str(path),
    )


@bp.delete("/api/mask/<dataset>/<image>")
def delete_mask(dataset: str, image: str):
    try:
        safe_name(dataset, "dataset")
        safe_name(image, "image")
        path = index().mask_path(dataset, image)
    except ValidationError as exc:
        return fail(exc, 400)
    except KeyError as exc:
        return fail(exc, 404)

    if path is None:
        return fail("no mask for this image", 404)
    path.unlink()
    index().refresh(force=True)
    return ok(deleted=str(path))


__all__ = ["bp"]
