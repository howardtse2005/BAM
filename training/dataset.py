"""Training dataset: RGB + synthetic stroke in, boundary band out.

Every geometric transform is applied to RGB, GT, stroke and validity mask with
the same matrix, so the four arrays can never drift apart. The target band is
derived from the *transformed* GT, which makes it aligned by construction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from imgio import imread_gray, imread_rgb, to_binary_mask

from .sampler import CropBox, CropSampler
from .stroke_synth import (
    StrokeParams,
    boundary_band,
    encode_stroke,
    ensure_binary,
    synthesize_stroke,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def load_manifest(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if "samples" not in data or not isinstance(data["samples"], list):
        raise ValueError("manifest must contain a 'samples' list")
    return data


def resolve_sample_paths(sample: dict, dataset_root: Path) -> tuple[Path, Path, Path | None]:
    rgb = dataset_root / sample["rgb"]
    gt = dataset_root / sample["gt"]
    edge = dataset_root / sample["gt_edge"] if sample.get("gt_edge") else None
    return rgb, gt, edge


def _pad_and_crop(array: np.ndarray, box: CropBox, pad_value: int | float) -> np.ndarray:
    """Crop from an array already padded by ``size // 2`` on every side."""
    half = box.size // 2
    x = box.x0 + half
    y = box.y0 + half
    return array[y : y + box.size, x : x + box.size]


def _validity(box: CropBox, height: int, width: int) -> np.ndarray:
    valid = np.zeros((box.size, box.size), np.uint8)
    x0, y0 = max(0, box.x0), max(0, box.y0)
    x1, y1 = min(width, box.x1), min(height, box.y1)
    if x1 > x0 and y1 > y0:
        valid[y0 - box.y0 : y1 - box.y0, x0 - box.x0 : x1 - box.x0] = 1
    return valid


def _contiguous(*arrays: np.ndarray) -> list[np.ndarray]:
    return [np.ascontiguousarray(a) for a in arrays]


# --------------------------------------------------------------------------- #
# Augmentation
# --------------------------------------------------------------------------- #


def geometric_augment(
    rgb: np.ndarray,
    gt: np.ndarray,
    stroke: np.ndarray,
    valid: np.ndarray,
    rng: np.random.Generator,
    p_flip: float = 0.5,
    p_rot90: float = 0.5,
    p_rotate: float = 0.3,
    max_angle: float = 12.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply one shared transform to all four arrays."""
    if rng.random() < p_flip:
        rgb, gt, stroke, valid = (a[:, ::-1] for a in (rgb, gt, stroke, valid))
    if rng.random() < p_flip:
        rgb, gt, stroke, valid = (a[::-1] for a in (rgb, gt, stroke, valid))

    if rng.random() < p_rot90:
        k = int(rng.integers(1, 4))
        rgb, gt, stroke, valid = (np.rot90(a, k) for a in (rgb, gt, stroke, valid))

    rgb, gt, stroke, valid = _contiguous(rgb, gt, stroke, valid)

    if rng.random() < p_rotate:
        h, w = rgb.shape[:2]
        angle = float(rng.uniform(-max_angle, max_angle))
        matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        rgb = cv2.warpAffine(
            rgb,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        gt = cv2.warpAffine(
            gt,
            matrix,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        stroke = cv2.warpAffine(
            stroke,
            matrix,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        valid = cv2.warpAffine(
            valid,
            matrix,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    return rgb, gt, stroke, valid


def photometric_augment(
    rgb: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Colour jitter on a float32 RGB image in [0, 1]."""
    if rng.random() < 0.7:
        brightness = float(rng.uniform(-0.2, 0.2))
        contrast = float(rng.uniform(0.8, 1.2))
        rgb = np.clip((rgb - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0)

    if rng.random() < 0.5:
        gray = rgb.mean(axis=2, keepdims=True)
        saturation = float(rng.uniform(0.8, 1.2))
        rgb = np.clip(gray + (rgb - gray) * saturation, 0.0, 1.0)

    if rng.random() < 0.3:
        shift = int(rng.integers(-8, 9))
        hsv = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        hsv[..., 0] = (hsv[..., 0].astype(np.int16) + shift) % 180
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

    if rng.random() < 0.2:
        sigma = float(rng.uniform(0.0, 1.2))
        if sigma > 0.05:
            rgb = cv2.GaussianBlur(rgb, (0, 0), sigma)

    if rng.random() < 0.2:
        noise = rng.normal(0.0, 6.0 / 255.0, rgb.shape).astype(np.float32)
        rgb = np.clip(rgb + noise, 0.0, 1.0)

    if rng.random() < 0.2:
        quality = int(rng.integers(60, 96))
        ok, buf = cv2.imencode(
            ".jpg", (rgb * 255).astype(np.uint8), [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        if ok:
            rgb = cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32) / 255.0

    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


class BAMDataset(Dataset):
    def __init__(
        self,
        samples: list[dict],
        dataset_root: str | Path,
        cfg,
        train: bool = True,
        length: int | None = None,
        seed: int | None = None,
    ) -> None:
        if not samples:
            raise ValueError("dataset split is empty")
        self.samples = samples
        self.root = Path(dataset_root)
        self.cfg = cfg
        self.train = bool(train)
        self.length = int(length) if length else len(samples)
        self.seed = int(seed if seed is not None else getattr(cfg, "seed", 1337))
        self.epoch = 0

        data = getattr(cfg, "data", cfg)
        self.crop_size = int(getattr(data, "crop_size", 256))
        self.target_sigma = float(getattr(data, "target_sigma", 1.0))
        self.stroke_source = str(getattr(data, "stroke_source", "online"))
        self.stroke_encoding = str(getattr(data, "stroke_encoding", "binary"))
        self.mean = np.asarray(getattr(data, "rgb_mean", (0.485, 0.456, 0.406, 0.0)), np.float32)
        self.std = np.asarray(getattr(data, "rgb_std", (0.229, 0.224, 0.225, 1.0)), np.float32)

        mix = getattr(data, "crop_mix", None)
        mix_dict = (
            {
                "boundary": getattr(mix, "boundary", 0.5),
                "near": getattr(mix, "near", 0.2),
                "empty": getattr(mix, "empty", 0.15),
                "dense": getattr(mix, "dense", 0.15),
            }
            if mix is not None
            else None
        )
        # Validation crops must contain the object. Sampling empty crops there
        # would score a free 1.0 whenever the prediction is also empty.
        if not self.train:
            mix_dict = {"boundary": 1.0, "near": 0.0, "empty": 0.0, "dense": 0.0}
        self.sampler = CropSampler(crop_size=self.crop_size, mix=mix_dict)
        self.stroke_params = StrokeParams.from_config(data)

    # -- torch plumbing ---------------------------------------------------- #

    def __len__(self) -> int:
        return self.length

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    # -- item -------------------------------------------------------------- #

    def _rng(self, index: int) -> np.random.Generator:
        seed = (self.seed + self.epoch * 1_000_003 + index * 9_176) % (2**32)
        return np.random.default_rng(seed)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = self._rng(index)
        sample = self.samples[index % len(self.samples)]

        rgb_path, gt_path, edge_path = resolve_sample_paths(sample, self.root)
        rgb = imread_rgb(rgb_path)
        gt = to_binary_mask(imread_gray(gt_path))

        if gt.shape[:2] != rgb.shape[:2]:
            gt = cv2.resize(
                gt, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST
            )

        if self.stroke_source == "disk" and edge_path is not None and edge_path.is_file():
            stroke = to_binary_mask(imread_gray(edge_path))
            if stroke.shape[:2] != rgb.shape[:2]:
                stroke = cv2.resize(
                    stroke, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST
                )
        else:
            stroke = synthesize_stroke(gt, rgb=rgb, params=self.stroke_params, rng=rng)

        # Optional scale jitter before cropping (compensates small source images).
        if self.train and rng.random() < 0.5:
            factor = float(rng.uniform(0.75, 1.35))
            h, w = rgb.shape[:2]
            nh = max(64, int(round(h * factor)))
            nw = max(64, int(round(w * factor)))
            interp = cv2.INTER_AREA if factor < 1.0 else cv2.INTER_LINEAR
            rgb = cv2.resize(rgb, (nw, nh), interpolation=interp)
            gt = cv2.resize(gt, (nw, nh), interpolation=cv2.INTER_NEAREST)
            stroke = cv2.resize(stroke, (nw, nh), interpolation=cv2.INTER_NEAREST)

        state = self.sampler.prepare(gt)
        box = self.sampler.sample(state, rng)

        half = self.crop_size // 2
        rgb_p = cv2.copyMakeBorder(
            rgb, half, half, half, half, cv2.BORDER_REPLICATE
        )
        gt_p = cv2.copyMakeBorder(gt, half, half, half, half, cv2.BORDER_CONSTANT, value=0)
        stroke_p = cv2.copyMakeBorder(
            stroke, half, half, half, half, cv2.BORDER_CONSTANT, value=0
        )

        rgb_c = _pad_and_crop(rgb_p, box, 0)
        gt_c = _pad_and_crop(gt_p, box, 0)
        stroke_c = _pad_and_crop(stroke_p, box, 0)
        valid = _validity(box, gt.shape[0], gt.shape[1])

        if self.train:
            rgb_c, gt_c, stroke_c, valid = geometric_augment(
                rgb_c, gt_c, stroke_c, valid, rng
            )

        rgb_f = rgb_c.astype(np.float32) / 255.0
        if self.train:
            rgb_f = photometric_augment(rgb_f, rng)

        gt_c = ensure_binary(gt_c)
        target = boundary_band(gt_c, sigma=self.target_sigma)

        stroke_ch = encode_stroke(stroke_c, encoding=self.stroke_encoding)

        rgb_n = (rgb_f - self.mean[:3]) / self.std[:3]
        image = np.concatenate([rgb_n.transpose(2, 0, 1), stroke_ch[None]], axis=0)
        image = np.ascontiguousarray(image, dtype=np.float32)

        quality = float(sample.get("quality_weight", 1.0))
        weight = (valid.astype(np.float32) * quality)[None]
        target = np.ascontiguousarray(target[None], dtype=np.float32)

        gt_bin = np.ascontiguousarray(gt_c[None].astype(np.float32) / 255.0)

        return {
            "image": torch.from_numpy(image),
            "target": torch.from_numpy(target),
            "weight": torch.from_numpy(weight),
            "gt": torch.from_numpy(gt_bin),
            "has_edge": torch.tensor(bool(target.max() > 0.1)),
            "index": torch.tensor(index, dtype=torch.long),
            "sample_id": sample.get("id", str(index)),
        }


# --------------------------------------------------------------------------- #
# Factories
# --------------------------------------------------------------------------- #


def build_dataset(
    cfg,
    split: str,
    train: bool | None = None,
    length: int | None = None,
    manifest: str | Path | None = None,
) -> BAMDataset:
    root = cfg.dataset_root
    manifest_path = Path(manifest) if manifest else root / "manifest.json"
    data = load_manifest(manifest_path)

    samples = [s for s in data["samples"] if s.get("split") == split]
    if not samples:
        raise ValueError(f"no samples with split={split!r} in {manifest_path}")

    if train is None:
        train = split == "train"

    return BAMDataset(
        samples=samples,
        dataset_root=root,
        cfg=cfg,
        train=train,
        length=length,
        seed=getattr(cfg, "seed", 1337),
    )


__all__ = [
    "BAMDataset",
    "build_dataset",
    "load_manifest",
    "geometric_augment",
    "photometric_augment",
]
