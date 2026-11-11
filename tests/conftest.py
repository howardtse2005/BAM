"""Shared fixtures: a tiny on-disk dataset and a Flask test client."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402
from training.stroke_synth import boundary_from_mask, synthesize_stroke  # noqa: E402


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, image)
    assert ok
    buf.tofile(str(path))


@pytest.fixture(scope="session")
def dataset_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("data") / "datasets"
    dataset = root / "toy"
    for sub in ("rgb", "gt", "gt_edge"):
        (dataset / sub).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(3)
    samples = []
    height, width = 260, 340

    for index in range(6):
        rgb = np.clip(rng.normal(120, 18, (height, width, 3)), 0, 255).astype(np.uint8)
        gt = np.zeros((height, width), np.uint8)
        cx = 120 + index * 18
        cy = 130 + (index % 3) * 12
        cv2.ellipse(gt, (cx, cy), (70, 48), index * 9, 0, 360, 255, -1)
        rgb[gt > 0] = (30 + index * 10, 150, 90)

        stroke = synthesize_stroke(gt, rgb=rgb, rng=index)
        stem = f"{index:09d}"

        _write_image(dataset / "rgb" / f"{stem}.jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        _write_image(dataset / "gt" / f"{stem}.png", gt)
        _write_image(dataset / "gt_edge" / f"{stem}.png", stroke)

        boundary = boundary_from_mask(gt)
        ys, xs = np.nonzero(boundary)
        samples.append(
            {
                "id": f"toy/{stem}",
                "dataset": "toy",
                "tier": "A",
                "rgb": f"toy/rgb/{stem}.jpg",
                "gt": f"toy/gt/{stem}.png",
                "gt_edge": f"toy/gt_edge/{stem}.png",
                "original_size": [width, height],
                "canonical_size": [width, height],
                "scale": 1.0,
                "split": "train" if index < 4 else ("val" if index == 4 else "test_a"),
                "instance_area_ratio": float((gt > 0).mean()),
                "boundary_sharpness": 0.3,
                "quality_weight": 0.9,
                "license": "test",
                "stroke_seed": index,
                "phash": str(index * 0x9E3779B97F4A7C15),
                "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            }
        )

    manifest = {"version": 1, "samples": samples, "datasets": ["toy"], "counts": {"toy": 6}}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return root


@pytest.fixture
def cfg(dataset_root: Path) -> Config:
    config = Config.from_dict(
        {
            "root": str(dataset_root.parent),
            "paths": {"datasets": str(dataset_root), "model": str(dataset_root.parent / "missing.pth")},
            "data": {"canonical_short_side": 640, "crop_size": 256, "num_workers": 0},
            "infer": {"backend": "mock", "device": "cpu", "allow_mock": True, "threads": 2},
        }
    )
    config.validate()
    return config


@pytest.fixture
def app(cfg: Config):
    from backend.app import create_app

    application = create_app(cfg)
    application.config.update(TESTING=True)
    return application


@pytest.fixture
def client(app):
    return app.test_client()
