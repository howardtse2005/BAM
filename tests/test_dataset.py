import numpy as np
import pytest
import torch

from training.dataset import BAMDataset, build_dataset, geometric_augment
from training.sampler import CropSampler


def test_dataset_item_shapes(cfg):
    dataset = build_dataset(cfg, "train", train=True, length=4)
    item = dataset[0]
    assert item["image"].shape == (4, 256, 256)
    assert item["target"].shape == (1, 256, 256)
    assert item["weight"].shape == (1, 256, 256)
    assert item["gt"].shape == (1, 256, 256)
    assert item["image"].dtype == torch.float32
    assert item["target"].min() >= 0.0
    assert item["target"].max() <= 1.0
    assert item["weight"].max() <= 1.0


def test_dataset_stroke_channel_is_not_empty(cfg):
    dataset = build_dataset(cfg, "train", train=True, length=8)
    fractions = [float(dataset[i]["image"][3].mean()) for i in range(4)]
    assert max(fractions) > 0.0


def test_dataset_val_is_deterministic(cfg):
    dataset = build_dataset(cfg, "val", train=False, length=2)
    a = dataset[0]["image"]
    b = dataset[0]["image"]
    assert torch.equal(a, b)


def test_dataset_length_controls_epoch_size(cfg):
    dataset = build_dataset(cfg, "train", train=True, length=17)
    assert len(dataset) == 17
    assert dataset[20]["image"].shape == (4, 256, 256)


def test_geometric_augment_keeps_arrays_aligned():
    height, width = 64, 80
    rgb = np.zeros((height, width, 3), np.uint8)
    gt = np.zeros((height, width), np.uint8)
    gt[:, 40:42] = 255
    stroke = np.zeros((height, width), np.uint8)
    stroke[:, 38:44] = 255
    valid = np.ones((height, width), np.uint8)

    rng = np.random.default_rng(0)
    out_rgb, out_gt, out_stroke, out_valid = geometric_augment(
        rgb, gt, stroke, valid, rng, p_flip=1.0, p_rot90=0.0, p_rotate=0.0
    )

    assert out_rgb.shape[:2] == out_gt.shape == out_stroke.shape == out_valid.shape

    columns = set(np.nonzero(out_gt.any(axis=0))[0].tolist())
    # Horizontal flip maps column c to width-1-c: 40,41 -> 39,38.
    assert columns == {38, 39}
    # The stroke must have moved with it, not stayed behind.
    stroke_columns = set(np.nonzero(out_stroke.any(axis=0))[0].tolist())
    assert min(stroke_columns) <= 38 and max(stroke_columns) >= 39


def test_geometric_augment_rotation_keeps_shapes():
    rgb = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    gt = np.zeros((64, 64), np.uint8)
    gt[20:40, 20:40] = 255
    stroke = gt.copy()
    valid = np.ones((64, 64), np.uint8)
    rng = np.random.default_rng(1)
    out = geometric_augment(rgb, gt, stroke, valid, rng, p_flip=0.0, p_rot90=0.0, p_rotate=1.0)
    assert all(a.shape[:2] == (64, 64) for a in out)
    assert out[1].max() == 255


def test_crop_sampler_covers_all_buckets():
    mask = np.zeros((400, 400), np.uint8)
    mask[100:300, 100:300] = 255
    sampler = CropSampler(crop_size=128)
    state = sampler.prepare(mask)
    rng = np.random.default_rng(0)

    seen = set()
    for _ in range(400):
        box = sampler.sample(state, rng)
        assert box.size == 128
        seen.add(sampler.last_bucket)

    assert seen == {"boundary", "near", "empty", "dense"}


def test_crop_sampler_respects_weights():
    mask = np.zeros((300, 300), np.uint8)
    mask[100:200, 100:200] = 255
    sampler = CropSampler(
        crop_size=96, mix={"boundary": 1.0, "near": 0.0, "empty": 0.0, "dense": 0.0}
    )
    state = sampler.prepare(mask)
    rng = np.random.default_rng(2)
    for _ in range(50):
        sampler.sample(state, rng)
        assert sampler.last_bucket == "boundary"


def test_manifest_roundtrip(cfg):
    dataset = BAMDataset(
        samples=[{"id": "toy/000000000", "rgb": "toy/rgb/000000000.jpg", "gt": "toy/gt/000000000.png", "gt_edge": "toy/gt_edge/000000000.png"}],
        dataset_root=cfg.dataset_root,
        cfg=cfg,
        train=False,
        length=1,
    )
    item = dataset[0]
    assert item["image"].shape == (4, 256, 256)


def test_missing_split_raises(cfg):
    with pytest.raises(ValueError):
        build_dataset(cfg, "nonexistent")
