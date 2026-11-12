from pathlib import Path

import pytest

from config import Config


def test_config_py_is_the_only_config_file():
    root = Path(__file__).resolve().parent.parent
    assert not (root / "config.yaml").exists(), (
        "config.yaml was merged into config.py; a second file drifts silently"
    )


def test_defaults_load_and_validate():
    cfg = Config.load()
    cfg.validate()
    assert cfg.crop_size == 256
    assert cfg.crop_stride == 128
    assert cfg.infer_device in ("cpu", "cuda")


def test_env_override_wins_over_defaults(monkeypatch):
    monkeypatch.setenv("BAM_INFER_DEVICE", "cuda")
    monkeypatch.setenv("BAM_CROP_SIZE", "320")
    cfg = Config.load(use_env=True)
    assert cfg.infer_device == "cuda"
    assert cfg.crop_size == 320


def test_cli_override_wins_over_env(monkeypatch):
    monkeypatch.setenv("BAM_INFER_DEVICE", "cuda")
    cfg = Config.load(use_env=True, overrides=["infer_device=cpu"])
    assert cfg.infer_device == "cpu"


def test_tuple_coercion():
    cfg = Config.from_dict({"brush_wobble_amp": [0.5, 2.5]})
    assert isinstance(cfg.brush_wobble_amp, tuple)
    assert cfg.brush_wobble_amp == (0.5, 2.5)


def test_int_and_float_coercion():
    cfg = Config.from_dict({"width": "24", "lr": "0.001"})
    assert cfg.width == 24 and isinstance(cfg.width, int)
    assert cfg.lr == 0.001 and isinstance(cfg.lr, float)


@pytest.mark.parametrize(
    "payload",
    [
        {"crop_size": 250},
        {"crop_stride": 512, "crop_size": 256},
        {"crop_min": 512},
        {"crop_min": 100},
        {"crop_ratio": 0.0},
        {"tier": "C"},
        {"block": "unknown"},
        {"mode": "banana"},
        {"port": 0},
        {"mix_boundary": 0.9},  # mixture must sum to 1
        {"depth": 0},
    ],
)
def test_validation_rejects(payload):
    with pytest.raises(ValueError):
        Config.from_dict(payload).validate()


@pytest.mark.parametrize(
    "height,width,expected",
    [
        (640, 640, 256),  # canonical short side keeps the training crop
        (640, 960, 256),
        (400, 400, 160),  # a smaller canvas gets a smaller crop
        (200, 300, 64),  # 80 snapped down to a multiple of 32
        (100, 100, 64),  # clamped to crop_min
        (4000, 3000, 256),  # never above the training crop
    ],
)
def test_crop_for_tracks_the_canvas_size(height, width, expected):
    cfg = Config.from_dict({})
    size = cfg.crop_for(height, width)
    assert size == expected
    assert size % 32 == 0
    assert cfg.crop_min <= size <= cfg.crop_size


def test_unknown_override_key_is_rejected():
    with pytest.raises(ValueError):
        Config.load(overrides=["nonsense=1"])


def test_fingerprint_is_stable():
    a = Config.from_dict({"seed": 7, "crop_size": 256})
    b = Config.from_dict({"seed": 7, "crop_size": 256})
    c = Config.from_dict({"seed": 8})
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != c.fingerprint()


def test_paths_resolve_against_root(tmp_path):
    cfg = Config.from_dict(
        {"root": str(tmp_path), "datasets_dir": "./data", "dataset": "demo"}
    )
    assert cfg.datasets_root == (tmp_path / "data").resolve()
    assert cfg.dataset_path == (tmp_path / "data" / "demo").resolve()
    assert cfg.model_file == (tmp_path / "model" / "bam_v1.pth").resolve()
