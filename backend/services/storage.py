"""Filesystem helpers: safe paths, data-URL decoding, atomic image writes."""

from __future__ import annotations

import base64
import binascii
import os
from pathlib import Path

import cv2
import numpy as np


class StorageError(ValueError):
    """Raised for invalid paths or malformed payloads."""


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #


def safe_join(root: str | Path, *parts: str) -> Path:
    """Join under ``root``, refusing anything that escapes it."""
    root_path = Path(root).resolve()
    candidate = root_path.joinpath(*[str(p) for p in parts]).resolve()
    try:
        candidate.relative_to(root_path)
    except ValueError as exc:
        raise StorageError(f"path escapes dataset root: {candidate}") from exc
    return candidate


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_write(path: str | Path, data: bytes) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# Base64
# --------------------------------------------------------------------------- #

_DATA_PREFIX = "data:image/"


def decode_data_url(value: str, max_bytes: int = 32 * 1024 * 1024) -> bytes:
    """Decode a data URL or a bare base64 string into bytes."""
    if not isinstance(value, str) or not value:
        raise StorageError("empty image payload")
    if len(value) > max_bytes * 2:
        raise StorageError("image payload too large")

    payload = value
    if value.startswith(_DATA_PREFIX):
        _, _, payload = value.partition(",")
    elif "," in value and "base64" in value.split(",", 1)[0]:
        payload = value.split(",", 1)[1]

    try:
        data = base64.b64decode(payload, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise StorageError("invalid base64 payload") from exc

    if not data:
        raise StorageError("decoded payload is empty")
    if len(data) > max_bytes:
        raise StorageError("image payload too large")
    return data


def encode_png_base64(image: np.ndarray) -> str:
    """Encode an array as a PNG data URL."""
    if image is None:
        raise StorageError("cannot encode None")
    arr = np.ascontiguousarray(image)
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise StorageError("PNG encoding failed")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def encode_png_bytes(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", np.ascontiguousarray(image))
    if not ok:
        raise StorageError("PNG encoding failed")
    return buf.tobytes()


def decode_image_bytes(data: bytes, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Decode image bytes, verifying that they really are an image."""
    if not data:
        raise StorageError("empty image bytes")
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, flags)
    if image is None:
        raise StorageError("payload is not a decodable image")
    return image


def decode_mask_bytes(data: bytes) -> np.ndarray:
    """Decode a mask, accepting grayscale or alpha-channel PNGs."""
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise StorageError("payload is not a decodable image")

    if image.ndim == 3:
        if image.shape[2] == 4:
            image = image[..., 3]
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.where(image >= 128, 255, 0).astype(np.uint8)


def sniff_image(data: bytes) -> str:
    """Identify an image by magic bytes, not by filename."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[:2] == b"BM":
        return "bmp"
    raise StorageError("unsupported or unrecognised image format")


def image_size(data: bytes) -> tuple[int, int]:
    """(width, height) without a full decode where possible."""
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise StorageError("could not read image dimensions")
    return int(image.shape[1]), int(image.shape[0])


__all__ = [
    "StorageError",
    "safe_join",
    "ensure_dir",
    "atomic_write",
    "decode_data_url",
    "encode_png_base64",
    "encode_png_bytes",
    "decode_image_bytes",
    "decode_mask_bytes",
    "sniff_image",
    "image_size",
]
