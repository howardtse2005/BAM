"""Configuration for BAM.

Single source of truth for training and inference. Resolution order:

    CLI overrides  >  environment  >  YAML file  >  dataclass defaults

Environment variables use a double underscore to separate sections:

    BAM_INFER__DEVICE=cpu
    BAM_DATA__CROP_SIZE=256
    BAM_TRAIN__BATCH_SIZE=32

CLI overrides use dot paths:

    python run.py --set infer.device=cpu --set server.port=8001

The resolved config is embedded in every checkpoint, so a model can always be
reproduced from the weights alone.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

try:  # PyYAML is a hard dependency of the app, but keep import errors readable
    import yaml
except Exception:  # pragma: no cover
    yaml = None


ENV_PREFIX = "BAM_"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #


@dataclass
class PathsCfg:
    datasets: str = "./datasets"
    model: str = "./model/bam_v1.pth"
    runs: str = "./runs"


@dataclass
class StrokeCfg:
    """Parameters of the synthetic user stroke. All lengths are canonical px."""

    radius_min: float = 1.5
    radius_max: float = 6.0
    jitter_sigma: float = 3.0
    wobble_amp: tuple = (1.0, 4.0)
    wobble_len: tuple = (20.0, 180.0)
    coverage: tuple = (0.45, 1.00)
    simplify_eps: tuple = (2.0, 10.0)
    smooth_window: tuple = (5, 21)
    max_offset: float = 12.0
    p_no_stroke: float = 0.08
    p_distractor: float = 0.15
    p_interior: float = 0.05
    p_partial: float = 0.15
    p_two_arcs: float = 0.15
    p_dry_brush: float = 0.15


@dataclass
class CropMixCfg:
    boundary: float = 0.50
    near: float = 0.20
    empty: float = 0.15
    dense: float = 0.15


@dataclass
class DataCfg:
    tier: str = "A"
    canonical_short_side: int = 640
    crop_size: int = 256
    stride: int = 128
    stroke_source: str = "online"  # online | disk
    stroke_encoding: str = "binary"  # binary | sdf
    target_sigma: float = 1.0
    num_workers: int = 8
    rgb_mean: tuple = (0.485, 0.456, 0.406, 0.0)
    rgb_std: tuple = (0.229, 0.224, 0.225, 1.0)
    stroke: StrokeCfg = field(default_factory=StrokeCfg)
    crop_mix: CropMixCfg = field(default_factory=CropMixCfg)


@dataclass
class ModelCfg:
    in_channels: int = 4
    out_channels: int = 1
    width: int = 20
    depth: int = 4
    max_width: int = 256
    block: str = "ds"  # ds | vanilla
    bilinear: bool = True


@dataclass
class TrainCfg:
    device: str = "auto"
    batch_size: int = 32
    crops_per_epoch: int = 24000
    epochs: int = 60
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 3
    amp: bool = True
    ema_decay: float = 0.999
    grad_clip: float = 1.0
    val_crops: int = 2000
    early_stop_patience: int = 15
    pos_weight: float = 6.0
    seed: int = 1337


@dataclass
class InferCfg:
    device: str = "cpu"
    backend: str = "auto"  # auto | onnx_int8 | onnx_fp32 | torch | mock
    working_short_side: int = 640
    crop_size: int = 256
    stride: int = 128
    max_crops: int = 64
    batch_size: int = 16
    threads: int = 8
    threshold: float = 0.5
    edge_threshold: float = 0.45
    sensitivity: float = 1.0
    expand: int = 0
    smooth: int = 2
    mode: str = "object"
    merge_weighting: str = "uniform"  # uniform | gaussian
    apply_dbscan: bool = False
    apply_morphology: bool = True
    morph_kernel_size: int = 3
    morph_iterations: int = 2
    cache_entries: int = 2048
    allow_mock: bool = True
    onnx_path: str = "./model/bam_v1.onnx"
    onnx_int8_path: str = "./model/bam_v1_int8.onnx"


@dataclass
class ServerCfg:
    host: str = "127.0.0.1"
    port: int = 8000
    max_content_mb: int = 24
    debug: bool = False
    max_canvas_side: int = 2048


@dataclass
class Config:
    root: str = ""
    seed: int = 1337
    paths: PathsCfg = field(default_factory=PathsCfg)
    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    infer: InferCfg = field(default_factory=InferCfg)
    server: ServerCfg = field(default_factory=ServerCfg)

    # -- paths ------------------------------------------------------------- #

    @property
    def project_root(self) -> Path:
        if self.root:
            return Path(self.root).expanduser().resolve()
        return Path(__file__).resolve().parent

    def path(self, value: str) -> Path:
        p = Path(value).expanduser()
        return p if p.is_absolute() else (self.project_root / p).resolve()

    @property
    def dataset_root(self) -> Path:
        return self.path(self.paths.datasets)

    @property
    def model_path(self) -> Path:
        return self.path(self.paths.model)

    @property
    def runs_root(self) -> Path:
        return self.path(self.paths.runs)

    # -- serialisation ----------------------------------------------------- #

    def to_dict(self) -> dict:
        return asdict(self)

    def to_yaml(self, path: str | Path | None = None) -> str:
        if yaml is None:
            raise RuntimeError("PyYAML is required to serialise the config")
        text = yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def fingerprint(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    # -- loading ----------------------------------------------------------- #

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        overrides: list[str] | None = None,
        use_env: bool = True,
    ) -> "Config":
        """Load defaults, then YAML, then env, then ``key=value`` overrides."""
        data: dict[str, Any] = {}

        cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
        if cfg_path.is_file():
            if yaml is None:
                raise RuntimeError("PyYAML is required to read config.yaml")
            loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"{cfg_path} must contain a mapping at the top level")
            _deep_update(data, loaded)

        if use_env:
            _deep_update(data, _env_overrides())

        for item in overrides or []:
            _apply_override(data, item)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict | None) -> "Config":
        return _from_dict(cls, data or {})

    # -- validation -------------------------------------------------------- #

    def validate(self) -> None:
        d, m, t, i, s = self.data, self.model, self.train, self.infer, self.server

        if d.tier not in ("A", "B"):
            raise ValueError("data.tier must be 'A' or 'B'")
        if d.crop_size % 32 != 0:
            raise ValueError("data.crop_size must be a multiple of 32")
        if not 0 < d.stride <= d.crop_size:
            raise ValueError("data.stride must be in (0, data.crop_size]")
        if d.canonical_short_side < 256:
            raise ValueError("data.canonical_short_side must be >= 256")
        if d.stroke_source not in ("online", "disk"):
            raise ValueError("data.stroke_source must be 'online' or 'disk'")
        if d.stroke_encoding not in ("binary", "sdf"):
            raise ValueError("data.stroke_encoding must be 'binary' or 'sdf'")
        if len(d.rgb_mean) != 4 or len(d.rgb_std) != 4:
            raise ValueError("data.rgb_mean / data.rgb_std must have 4 entries")

        if m.in_channels != 4:
            raise ValueError("model.in_channels must be 4 (RGB + stroke)")
        if m.depth < 1:
            raise ValueError("model.depth must be >= 1")
        if m.block not in ("ds", "vanilla"):
            raise ValueError("model.block must be 'ds' or 'vanilla'")
        if m.width < 4:
            raise ValueError("model.width must be >= 4")

        if t.device not in ("auto", "cpu", "cuda"):
            raise ValueError("train.device must be auto, cpu or cuda")
        if t.batch_size < 1:
            raise ValueError("train.batch_size must be >= 1")

        if i.device not in ("auto", "cpu", "cuda"):
            raise ValueError("infer.device must be auto, cpu or cuda")
        if i.backend not in ("auto", "onnx_int8", "onnx_fp32", "torch", "mock"):
            raise ValueError("infer.backend is invalid")
        if i.mode not in ("object", "line"):
            raise ValueError("infer.mode must be 'object' or 'line'")
        if i.merge_weighting not in ("uniform", "gaussian"):
            raise ValueError("infer.merge_weighting must be 'uniform' or 'gaussian'")
        if not 1 <= i.max_crops <= 256:
            raise ValueError("infer.max_crops must be in [1, 256]")
        if i.crop_size % 32 != 0:
            raise ValueError("infer.crop_size must be a multiple of 32")

        if not 1 <= s.port <= 65535:
            raise ValueError("server.port must be in [1, 65535]")
        if s.max_content_mb < 1:
            raise ValueError("server.max_content_mb must be >= 1")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _deep_update(base: dict, extra: dict) -> dict:
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _leaf_paths(cls, prefix: tuple[str, ...] = ()) -> dict[str, Any]:
    """Map dotted leaf paths to their dataclass field types."""
    hints = get_type_hints(cls)
    out: dict[str, Any] = {}
    for f in fields(cls):
        hint = hints[f.name]
        if is_dataclass(hint):
            out.update(_leaf_paths(hint, prefix + (f.name,)))
        else:
            out[".".join(prefix + (f.name,))] = hint
    return out


def _coerce(value: Any, hint: Any) -> Any:
    if is_dataclass(hint):
        return _from_dict(hint, value)
    origin = get_origin(hint)
    if origin in (tuple, list) or hint in (tuple, list):
        args = get_args(hint)
        seq = list(value)
        if args and not (len(args) == 2 and args[1] is Ellipsis):
            seq = [_coerce(v, a) for v, a in zip(seq, args)]
        return tuple(seq)
    if hint is bool:
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if hint is int:
        return int(value)
    if hint is float:
        return float(value)
    if hint is str:
        return str(value)
    return value


def _from_dict(cls, data: dict) -> Any:
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name in data:
            kwargs[f.name] = _coerce(data[f.name], hints[f.name])
    return cls(**kwargs)


def _parse_scalar(text: str) -> Any:
    if yaml is not None:
        try:
            return yaml.safe_load(text)
        except Exception:
            return text
    return text


def _env_overrides() -> dict:
    out: dict = {}
    for key, raw in os.environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        path = key[len(ENV_PREFIX) :].lower()
        parts = [p for p in path.split("__") if p]
        if not parts:
            continue
        node = out
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _parse_scalar(raw)
    return out


def _apply_override(data: dict, item: str) -> None:
    if "=" not in item:
        raise ValueError(f"override must look like key=value, got {item!r}")
    key, raw = item.split("=", 1)
    parts = [p for p in key.strip().split(".") if p]
    if not parts:
        raise ValueError(f"empty override key in {item!r}")
    node = data
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ValueError(f"override path {key!r} collides with a scalar")
    node[parts[-1]] = _parse_scalar(raw)


def add_config_args(parser) -> None:
    """Attach the standard config flags to an argparse parser."""
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set infer.device=cpu",
    )
