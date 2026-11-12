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
    assert cfg.crop_min == 64
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
        {"crop_min": 512},
        {"crop_min": 100},
        {"tier": "C"},
        {"block": "unknown"},
        {"mode": "banana"},
        {"port": 0},
        {"mix_boundary": 0.9},  # mixture must sum to 1
        {"depth": 0},
        {"magic_brush_width": 2},
        {"magic_prompt_ratio": 0.0},
        {"magic_prompt_min": 30, "magic_prompt_max": 20},
    ],
)
def test_validation_rejects(payload):
    with pytest.raises(ValueError):
        Config.from_dict(payload).validate()


def test_unknown_override_key_is_rejected():
    with pytest.raises(ValueError):
        Config.load(overrides=["nonsense=1"])


def test_prompt_width_tracks_the_band():
    cfg = Config.from_dict(
        {
            "magic_brush_width": 32,
            "magic_prompt_ratio": 0.625,
            "magic_prompt_min": 3,
            "magic_prompt_max": 20,
        }
    )
    assert cfg.prompt_width_for(32) == 20  # ratio, and the widest prompt allowed
    assert cfg.prompt_width_for(96) == 20  # capped: a thick band stays a thin prompt
    assert cfg.prompt_width_for(8) == 5  # below the cap the ratio decides
    assert cfg.prompt_width_for(4) == 3  # never thinner than the minimum


@pytest.mark.parametrize("band", [4, 8, 16, 32, 64, 96, 160])
def test_prompt_width_stays_inside_its_bounds(band):
    cfg = Config.from_dict({})
    prompt = cfg.prompt_width_for(band)
    assert 1 <= prompt <= band
    assert prompt <= cfg.magic_prompt_max
    assert prompt >= min(cfg.magic_prompt_min, band)


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
