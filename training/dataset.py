"""Dataset scanning and the training Dataset.

Layout (see README step 2):

    datasets/<name>/train/im/*.jpg    datasets/<name>/train/gt/*.png
    datasets/<name>/val/im/*.jpg      datasets/<name>/val/gt/*.png
    datasets/<name>/test/im/*.jpg     datasets/<name>/test/gt/*.png

``gt_edge/`` is created next to ``im/`` and ``gt/`` by the preprocessing step and
is used for deterministic validation strokes. Training regenerates a fresh
stroke for every sample, so augmentation is effectively unbounded.

There is no manifest: the folder structure is the manifest. Splits are whatever
the user put in ``train/``, ``val/`` and ``test/``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from config import RGB_MEAN, RGB_STD
from imgio import imread_gray, imread_rgb, to_binary_mask

from .paths import SPLITS, pair_split, split_counts
from .sampler import CropBox, CropSampler
from .stroke_synth import (
    StrokeParams,
    boundary_band,
    encode_stroke,
    ensure_binary,
    synthesize_stroke,
)

def scan_split(root: str | Path, split: str) -> list[dict[str, Any]]:
    """Pair images and masks in ``<root>/<split>`` (implemented in training.paths)."""
    return pair_split(root, split)


def dataset_fingerprint(files_by_split: dict[str, list[dict]]) -> str:
    """Stable hash of the dataset contents, recorded in every checkpoint."""
    digest = hashlib.sha256()
    for split in SPLITS:
        for item in files_by_split.get(split, []):
            digest.update(
                f"{split}/{item['stem']}/{item['size']}/{item['mtime']}\n".encode()
            )
    return digest.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Augmentation
# --------------------------------------------------------------------------- #


def _contiguous(*arrays: np.ndarray) -> list[np.ndarray]:
    return [np.ascontiguousarray(a) for a in arrays]


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
            rgb, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        gt = cv2.warpAffine(
            gt, matrix, (w, h), flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        stroke = cv2.warpAffine(
            stroke, matrix, (w, h), flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        valid = cv2.warpAffine(
            valid, matrix, (w, h), flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )

    return rgb, gt, stroke, valid


def photometric_augment(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
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


def _crop(array: np.ndarray, box: CropBox) -> np.ndarray:
    half = box.size // 2
    return array[box.y0 + half : box.y0 + half + box.size, box.x0 + half : box.x0 + half + box.size]


def _validity(box: CropBox, height: int, width: int) -> np.ndarray:
    valid = np.zeros((box.size, box.size), np.uint8)
    x0, y0 = max(0, box.x0), max(0, box.y0)
    x1, y1 = min(width, box.x1), min(height, box.y1)
    if x1 > x0 and y1 > y0:
        valid[y0 - box.y0 : y1 - box.y0, x0 - box.x0 : x1 - box.x0] = 1
    return valid


class BAMDataset(Dataset):
    """Crops of ``<dataset>/<split>/im`` with a synthetic stroke channel."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        cfg,
        train: bool | None = None,
        length: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.cfg = cfg
        self.files = scan_split(self.root, split)
        self.train = (split == "train") if train is None else bool(train)
        self.length = int(length) if length else len(self.files)
        self.epoch = 0

        self.crop_size = int(cfg.crop_size)
        self.target_sigma = float(cfg.target_sigma)
        self.stroke_encoding = str(cfg.stroke_encoding)
        self.mean = np.asarray(RGB_MEAN, np.float32)
        self.std = np.asarray(RGB_STD, np.float32)
        self.seed = int(cfg.seed)

        mix = {
            "boundary": float(cfg.mix_boundary),
            "near": float(cfg.mix_near),
            "empty": float(cfg.mix_empty),
            "dense": float(cfg.mix_dense),
        }
        # Validation crops must contain the object; sampling empty crops there
        # would score a free 1.0 whenever the prediction is also empty.
        if not self.train:
            mix = {"boundary": 1.0, "near": 0.0, "empty": 0.0, "dense": 0.0}

        self.sampler = CropSampler(crop_size=self.crop_size, mix=mix)
        self.stroke_params = StrokeParams.from_config(cfg)

    def __len__(self) -> int:
        return self.length

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    # -- helpers ----------------------------------------------------------- #

    def _rng(self, index: int) -> np.random.Generator:
        seed = (self.seed + self.epoch * 1_000_003 + index * 9_176) % (2**32)
        return np.random.default_rng(seed)

    @staticmethod
    def _stem_seed(stem: str) -> int:
        return int(hashlib.sha256(stem.encode()).hexdigest()[:8], 16)

    def _stroke(self, item: dict, gt: np.ndarray, rgb: np.ndarray, rng) -> np.ndarray:
        if self.train:
            return synthesize_stroke(gt, rgb=rgb, params=self.stroke_params, rng=rng)

        # Validation must be reproducible: use the stored stroke when it exists,
        # otherwise synthesise once from a seed derived from the filename.
        if item["edge"] is not None:
            stroke = to_binary_mask(imread_gray(item["edge"]))
            if stroke.shape[:2] != gt.shape[:2]:
                stroke = cv2.resize(
                    stroke, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST
                )
            return stroke
        return synthesize_stroke(
            gt, rgb=rgb, params=self.stroke_params, rng=self._stem_seed(item["stem"])
        )

    # -- item -------------------------------------------------------------- #

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = self._rng(index)
        item = self.files[index % len(self.files)]

        rgb = imread_rgb(item["im"])
        gt = to_binary_mask(imread_gray(item["gt"]))
        if gt.shape[:2] != rgb.shape[:2]:
            gt = cv2.resize(gt, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

        stroke = self._stroke(item, gt, rgb, rng)

        if self.train and rng.random() < 0.5:
            factor = float(rng.uniform(0.75, 1.35))
            h, w = rgb.shape[:2]
            nh, nw = max(64, int(round(h * factor))), max(64, int(round(w * factor)))
            interp = cv2.INTER_AREA if factor < 1.0 else cv2.INTER_LINEAR
            rgb = cv2.resize(rgb, (nw, nh), interpolation=interp)
            gt = cv2.resize(gt, (nw, nh), interpolation=cv2.INTER_NEAREST)
            stroke = cv2.resize(stroke, (nw, nh), interpolation=cv2.INTER_NEAREST)

        state = self.sampler.prepare(gt)
        box = self.sampler.sample(state, rng)

        half = self.crop_size // 2
        rgb_p = cv2.copyMakeBorder(rgb, half, half, half, half, cv2.BORDER_REPLICATE)
        gt_p = cv2.copyMakeBorder(gt, half, half, half, half, cv2.BORDER_CONSTANT, value=0)
        stroke_p = cv2.copyMakeBorder(
            stroke, half, half, half, half, cv2.BORDER_CONSTANT, value=0
        )

        rgb_c = _crop(rgb_p, box)
        gt_c = _crop(gt_p, box)
        stroke_c = _crop(stroke_p, box)
        valid = _validity(box, gt.shape[0], gt.shape[1])

        if self.train:
            rgb_c, gt_c, stroke_c, valid = geometric_augment(rgb_c, gt_c, stroke_c, valid, rng)

        rgb_f = rgb_c.astype(np.float32) / 255.0
        if self.train:
            rgb_f = photometric_augment(rgb_f, rng)

        gt_c = ensure_binary(gt_c)
        target = boundary_band(gt_c, sigma=self.target_sigma)
        stroke_ch = encode_stroke(stroke_c, encoding=self.stroke_encoding)

        rgb_n = (rgb_f - self.mean[:3]) / self.std[:3]
        image = np.concatenate([rgb_n.transpose(2, 0, 1), stroke_ch[None]], axis=0)

        return {
            "image": torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)),
            "target": torch.from_numpy(np.ascontiguousarray(target[None], dtype=np.float32)),
            "weight": torch.from_numpy(np.ascontiguousarray(valid[None], dtype=np.float32)),
            "gt": torch.from_numpy(
                np.ascontiguousarray(gt_c[None].astype(np.float32) / 255.0)
            ),
            "has_edge": torch.tensor(bool(target.max() > 0.1)),
            "index": torch.tensor(index, dtype=torch.long),
            "stem": item["stem"],
        }


def build_dataset(
    cfg,
    split: str,
    train: bool | None = None,
    length: int | None = None,
) -> BAMDataset:
    return BAMDataset(
        root=cfg.dataset_path,
        split=split,
        cfg=cfg,
        train=train,
        length=length,
    )


__all__ = [
    "BAMDataset",
    "build_dataset",
    "scan_split",
    "split_counts",
    "dataset_fingerprint",
    "geometric_augment",
    "photometric_augment",
    "SPLITS",
]
