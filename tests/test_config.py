import pytest

from config import Config


def test_defaults_load_and_validate():
    cfg = Config.load()
    cfg.validate()
    assert cfg.data.crop_size == 256
    assert cfg.model.in_channels == 4
    assert cfg.infer.device in ("cpu", "cuda")


def test_env_override_wins_over_yaml(monkeypatch):
    monkeypatch.setenv("BAM_INFER__DEVICE", "cuda")
    monkeypatch.setenv("BAM_DATA__CROP_SIZE", "320")
    cfg = Config.load(use_env=True)
    assert cfg.infer.device == "cuda"
    assert cfg.data.crop_size == 320


def test_cli_override_wins_over_env(monkeypatch):
    monkeypatch.setenv("BAM_INFER__DEVICE", "cuda")
    cfg = Config.load(use_env=True, overrides=["infer.device=cpu"])
    assert cfg.infer.device == "cpu"


def test_tuple_coercion():
    cfg = Config.from_dict({"data": {"rgb_mean": [0.1, 0.2, 0.3, 0.0]}})
    assert isinstance(cfg.data.rgb_mean, tuple)
    assert cfg.data.rgb_mean == (0.1, 0.2, 0.3, 0.0)


def test_nested_section_coercion():
    cfg = Config.from_dict({"data": {"stroke": {"radius_min": 2, "radius_max": 5}}})
    assert cfg.data.stroke.radius_min == 2.0
    assert isinstance(cfg.data.stroke.radius_min, float)
    assert cfg.data.stroke.radius_max == 5.0


@pytest.mark.parametrize(
    "payload",
    [
        {"data": {"crop_size": 250}},
        {"data": {"stride": 512, "crop_size": 256}},
        {"data": {"tier": "C"}},
        {"model": {"in_channels": 3}},
        {"model": {"block": "unknown"}},
        {"infer": {"mode": "banana"}},
        {"server": {"port": 0}},
    ],
)
def test_validation_rejects(payload):
    with pytest.raises(ValueError):
        Config.from_dict(payload).validate()


def test_fingerprint_is_stable():
    a = Config.from_dict({"seed": 7, "data": {"crop_size": 256}})
    b = Config.from_dict({"seed": 7, "data": {"crop_size": 256}})
    c = Config.from_dict({"seed": 8})
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != c.fingerprint()


def test_paths_resolve_against_root(tmp_path):
    cfg = Config.from_dict({"root": str(tmp_path), "paths": {"datasets": "./data"}})
    assert cfg.dataset_root == (tmp_path / "data").resolve()
