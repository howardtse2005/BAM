"""Shared fixtures: a tiny on-disk dataset in the split layout and a test client."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402
from training.stroke_synth import clean_edge, synthesize_brush  # noqa: E402

SPLIT_SIZES = {"train": 4, "val": 1, "test": 1}


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, image)
    assert ok
    buf.tofile(str(path))


@pytest.fixture(scope="session")
def dataset_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("data") / "datasets"
    dataset = root / "toy"

    rng = np.random.default_rng(3)
    height, width = 260, 340
    index = 0

    for split, count in SPLIT_SIZES.items():
        for sub in ("im", "gt", "gt_edge", "gt_brush"):
            (dataset / split / sub).mkdir(parents=True, exist_ok=True)

        for _ in range(count):
            rgb = np.clip(rng.normal(120, 18, (height, width, 3)), 0, 255).astype(np.uint8)
            gt = np.zeros((height, width), np.uint8)
            cx = 120 + index * 14
            cy = 130 + (index % 3) * 10
            cv2.ellipse(gt, (cx, cy), (70, 48), index * 9, 0, 360, 255, -1)
            rgb[gt > 0] = (30 + index * 8, 150, 90)

            edge = clean_edge(gt)
            brush = synthesize_brush(gt, rgb=rgb, rng=index)
            stem = f"{index:09d}"

            _write_image(
                dataset / split / "im" / f"{stem}.jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            )
            _write_image(dataset / split / "gt" / f"{stem}.png", gt)
            _write_image(dataset / split / "gt_edge" / f"{stem}.png", edge)
            _write_image(dataset / split / "gt_brush" / f"{stem}.png", brush)
            index += 1

    return root


@pytest.fixture
def cfg(dataset_root: Path) -> Config:
    config = Config.from_dict(
        {
            "root": str(dataset_root.parent),
            "datasets_dir": str(dataset_root),
            "dataset": "toy",
            "model_path": str(dataset_root.parent / "missing.pth"),
            "short_side": 640,
            "crop_size": 256,
            "num_workers": 0,
            "backend": "mock",
            "infer_device": "cpu",
            "allow_mock": True,
            "threads": 2,
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
