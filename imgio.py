"""Unicode-safe image IO and canonical resizing, shared by training and serving.

``cv2.imread`` cannot open paths containing non-ASCII characters on Windows.
Every image read or write in this project goes through this module instead.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np


def round_half_up(value: float) -> int:
    """Round half away from zero, matching JavaScript's Math.round."""
    return int(np.floor(float(value) + 0.5))


def canonical_size(
    height: int, width: int, short_side: int
) -> tuple[int, int, float]:
    """Target (width, height, scale) for a canonical short side.

    Never upscales: an image already smaller than ``short_side`` is left alone.
    """
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    short_side = int(short_side)
    if short_side <= 0:
        raise ValueError("short_side must be positive")
    current = min(height, width)
    if current <= short_side:
        return int(width), int(height), 1.0
    scale = short_side / float(current)
    return round_half_up(width * scale), round_half_up(height * scale), scale


def imread(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Read an image from any path, including non-ASCII ones on Windows."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    buf = np.fromfile(str(path), dtype=np.uint8)
    if buf.size == 0:
        raise ValueError(f"empty image file: {path}")
    image = cv2.imdecode(buf, flags)
    if image is None:
        raise ValueError(f"could not decode image: {path}")
    return image


def imread_rgb(path: str | Path) -> np.ndarray:
    """Read as RGB uint8 HxWx3."""
    return cv2.cvtColor(imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def imread_gray(path: str | Path) -> np.ndarray:
    """Read as single-channel uint8."""
    return imread(path, cv2.IMREAD_GRAYSCALE)


def imwrite(path: str | Path, image: np.ndarray) -> None:
    """Write an image atomically, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".png"
    if suffix not in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        raise ValueError(f"unsupported image extension: {suffix}")

    params: list[int] = []
    if suffix in (".jpg", ".jpeg"):
        params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    elif suffix == ".png":
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), 6]
    elif suffix == ".webp":
        params = [int(cv2.IMWRITE_WEBP_QUALITY), 90]

    ok, buf = cv2.imencode(suffix, image, params)
    if not ok:
        raise ValueError(f"could not encode image: {path}")

    tmp = path.with_name(path.name + ".tmp")
    buf.tofile(str(tmp))
    _replace_with_retry(tmp, path)


def _replace_with_retry(source: Path, target: Path, attempts: int = 5) -> None:
    """``os.replace``, retried briefly.

    On Windows a reader (indexer, thumbnailer, editor preview) holding the
    destination open makes the rename fail with "access denied" for a few
    milliseconds. Retrying is the difference between a preprocessing run that
    finishes and one that dies on image 9 of 3000.
    """
    for attempt in range(attempts):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.05 * (attempt + 1))


def resize(
    image: np.ndarray, width: int, height: int, is_mask: bool = False
) -> np.ndarray:
    """Resize with the interpolation appropriate to the content type."""
    if image.shape[1] == width and image.shape[0] == height:
        return image
    shrinking = width * height < image.shape[1] * image.shape[0]
    if is_mask:
        interp = cv2.INTER_NEAREST
    else:
        interp = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    return cv2.resize(image, (int(width), int(height)), interpolation=interp)


def resize_canonical(
    image: np.ndarray, short_side: int, is_mask: bool = False
) -> tuple[np.ndarray, float]:
    """Resize so the short side equals ``short_side`` (downscale only)."""
    h, w = image.shape[:2]
    tw, th, scale = canonical_size(h, w, short_side)
    if (tw, th) == (w, h):
        return image, 1.0
    return resize(image, tw, th, is_mask=is_mask), scale


def to_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Normalise any mask to uint8 {0, 255}."""
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.size and mask.max() <= 1:
        mask = (mask > 0).astype(np.uint8) * 255
    return np.where(mask >= 128, 255, 0).astype(np.uint8)


__all__ = [
    "round_half_up",
    "canonical_size",
    "imread",
    "imread_rgb",
    "imread_gray",
    "imwrite",
    "resize",
    "resize_canonical",
    "to_binary_mask",
]
