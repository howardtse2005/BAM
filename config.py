"""BAM configuration — the single source of truth.

Every setting and its default live in the dataclass below, one per line, next to
the comment that explains it. There is no second file to keep in sync.

    Config.load()                       # defaults + environment
    Config.load(overrides=["epochs=5"]) # plus CLI-style overrides

Resolution order: CLI override > environment > defaults.

Environment variables are the field name in upper case with a ``BAM_`` prefix:

    BAM_BATCH_SIZE=16 BAM_TRAIN_DEVICE=cuda python run.py

CLI overrides use the field name directly:

    python run.py --set port=8001

``Config.to_yaml`` still exists, but only as an *output*: training writes the
resolved values into ``runs/<id>/config.yaml`` as a record of what ran.

Dataset layout expected on disk (see README step 2):

    datasets/<name>/train/im/*.jpg   datasets/<name>/train/gt/*.png
    datasets/<name>/val/im/*.jpg     datasets/<name>/val/gt/*.png
    datasets/<name>/test/im/*.jpg    datasets/<name>/test/gt/*.png
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

try:
    import yaml
except Exception:  # pragma: no cover - keep the import error readable
    yaml = None


ENV_PREFIX = "BAM_"

# Normalisation constants shared by training and inference. Not configurable on
# purpose: a mismatch here silently degrades every prediction.
RGB_MEAN = (0.485, 0.456, 0.406, 0.0)
RGB_STD = (0.229, 0.224, 0.225, 1.0)

SPLITS = ("train", "val", "test")


@dataclass
class Config:
    # ------------------------------------------------------------------ paths
    datasets_dir: str = "./datasets"          # folder holding every dataset
    dataset: str = "demo"                     # which dataset to train on
    upload_dataset: str = "uploads"           # where the website stores uploaded images
    model_path: str = "./model/bam_v1.pth"
    onnx_path: str = "./model/bam_v1.onnx"
    onnx_int8_path: str = "./model/bam_v1_int8.onnx"
    runs_dir: str = "./runs"
    root: str = ""                            # "" = the folder containing this file

    # ------------------------------------------------------------- resolution
    short_side: int = 640                     # images are downscaled to this short side
    crop_size: int = 256                      # training crop, and the largest inference window
    crop_min: int = 64                        # smallest inference window
    target_sigma: float = 1.0                 # width of the soft edge target
    num_workers: int = 8                      # dataloader workers

    # -------------------------------------------------------- brush synthesis
    brush_radius_min: float = 1.5             # brush radius range, canonical px
    brush_radius_max: float = 6.0
    brush_jitter: float = 3.0                 # global offset sigma
    brush_wobble_amp: tuple = (1.0, 4.0)      # wobble amplitude range
    brush_wobble_len: tuple = (20.0, 180.0)   # wobble wavelength range
    brush_coverage: tuple = (1.00, 1.00)      # fraction of each contour covered
    brush_simplify: tuple = (2.0, 10.0)       # RDP epsilon range
    brush_smooth: tuple = (5, 21)             # smoothing window range
    brush_max_offset: float = 12.0            # displacement clip, px
    brush_encoding: str = "binary"            # binary | sdf
    # Off by default: these make the brush cover less than the edge, which would
    # leave the 4th channel and the target describing different things.
    p_no_brush: float = 0.0
    p_distractor: float = 0.0
    p_interior: float = 0.0
    p_partial: float = 0.0
    p_two_arcs: float = 0.0
    p_dry_brush: float = 0.0

    # ------------------------------------------------------------ crop mixture
    mix_boundary: float = 0.50
    mix_near: float = 0.20
    mix_empty: float = 0.15
    mix_dense: float = 0.15

    # ------------------------------------------------------------------- model
    width: int = 20                           # base channel width
    depth: int = 4                            # encoder/decoder levels
    max_width: int = 256                      # channel cap
    block: str = "ds"                         # ds | vanilla
    bilinear: bool = True                     # bilinear upsampling

    # ---------------------------------------------------------------- training
    seed: int = 1337
    train_device: str = "auto"                # auto | cpu | cuda
    batch_size: int = 32
    crops_per_epoch: int = 24000
    val_crops: int = 2000
    epochs: int = 60
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 3
    amp: bool = True
    ema_decay: float = 0.999
    grad_clip: float = 1.0
    early_stop: int = 15
    pos_weight: float = 6.0

    # --------------------------------------------------------------- inference
    infer_device: str = "cpu"                 # cpu | cuda
    backend: str = "auto"                     # auto | onnx_int8 | onnx_fp32 | torch | mock
    max_crops: int = 64
    infer_batch: int = 16
    threads: int = 8
    threshold: float = 0.5
    edge_threshold: float = 0.45
    sensitivity: float = 1.0
    expand: int = 0
    smooth: int = 2
    mode: str = "line"                        # line = predict the edge, object = also fill
    line_width: int = 3                       # predicted line thickness, canvas px
    fill_close: int = 5                       # px of gap bridged before filling a line

    # ------------------------------------------------------ magic brush geometry
    # The user draws a *band*; the model is given a thinner *prompt* stroke inside
    # it. The band is the tolerance the predicted line is allowed to move in, so a
    # prediction that sits a few px off the centreline is still under the stroke
    # instead of floating beside it.
    magic_brush_width: int = 32               # visible band width, canvas px (slider default)
    magic_prompt_ratio: float = 0.625         # prompt width / band width (20 px in a 32 px band)
    magic_prompt_min: int = 3                 # prompt is never thinner than this
    magic_prompt_max: int = 12                # ...nor wider; training brushes are 3-12 px
    merge_weighting: str = "uniform"          # uniform | gaussian
    apply_dbscan: bool = False
    apply_morphology: bool = True
    morph_kernel: int = 3
    morph_iterations: int = 2
    cache_entries: int = 2048
    allow_mock: bool = True

    # ------------------------------------------------------------------ server
    host: str = "127.0.0.1"
    port: int = 8000
    max_upload_mb: int = 24
    debug: bool = False
    max_canvas_side: int = 2048

    # -------------------------------------------------------------------- data
    tier: str = "A"                           # A = shippable, B = research only

    # ------------------------------------------------------------------- paths

    @property
    def project_root(self) -> Path:
        if self.root:
            return Path(self.root).expanduser().resolve()
        return Path(__file__).resolve().parent

    def path(self, value: str) -> Path:
        p = Path(value).expanduser()
        return p if p.is_absolute() else (self.project_root / p).resolve()

    @property
    def datasets_root(self) -> Path:
        return self.path(self.datasets_dir)

    @property
    def dataset_path(self) -> Path:
        return self.datasets_root / self.dataset

    @property
    def model_file(self) -> Path:
        return self.path(self.model_path)

    @property
    def onnx_file(self) -> Path:
        return self.path(self.onnx_path)

    @property
    def onnx_int8_file(self) -> Path:
        return self.path(self.onnx_int8_path)

    @property
    def runs_root(self) -> Path:
        return self.path(self.runs_dir)

    # --------------------------------------------------------------- geometry

    def prompt_width_for(self, band_width: int | None = None) -> int:
        """Width of the stroke actually handed to the model.

        The band is what the user draws and sees; the prompt is the thin core
        inside it that becomes the fourth channel. Keeping the prompt a fixed
        fraction of the band would let a thick band turn into a prompt far wider
        than any training brush, so the result is clamped to
        ``[magic_prompt_min, magic_prompt_max]`` and never exceeds the band it
        sits in. The frontend mirrors this formula (see ``EditorCanvas``).
        """
        band = int(band_width if band_width is not None else self.magic_brush_width)
        raw = int(round(band * float(self.magic_prompt_ratio)))
        clamped = max(int(self.magic_prompt_min), min(int(self.magic_prompt_max), raw))
        return max(1, min(band, clamped))

    # ------------------------------------------------------------ serialisation

    def to_dict(self) -> dict:
        return asdict(self)

    def to_yaml(self, path: str | Path | None = None) -> str:
        """Serialise the resolved config, for the run record only."""
        if yaml is None:
            raise RuntimeError("PyYAML is required to serialise the config")
        text = yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def fingerprint(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    # ------------------------------------------------------------------ loading

    @classmethod
    def load(
        cls,
        overrides: list[str] | None = None,
        use_env: bool = True,
    ) -> "Config":
        """Defaults, then ``BAM_*`` environment, then ``key=value`` overrides."""
        data: dict[str, Any] = {}

        if use_env:
            data.update(_env_overrides())

        for item in overrides or []:
            key, value = _parse_override(item)
            data[key] = value

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict | None) -> "Config":
        hints = get_type_hints(cls)
        kwargs: dict[str, Any] = {}
        for f in fields(cls):
            if f.name in (data or {}):
                kwargs[f.name] = _coerce(data[f.name], hints[f.name])
        return cls(**kwargs)

    # --------------------------------------------------------------- validation

    def validate(self) -> None:
        if not self.dataset:
            raise ValueError("dataset must name a folder under datasets_dir")
        if not self.upload_dataset:
            raise ValueError("upload_dataset must name a folder under datasets_dir")
        if self.tier not in ("A", "B"):
            raise ValueError("tier must be 'A' or 'B'")

        if self.short_side < 256:
            raise ValueError("short_side must be >= 256")
        if self.crop_size % 32 != 0:
            raise ValueError("crop_size must be a multiple of 32")
        if self.crop_min % 32 != 0 or not 32 <= self.crop_min <= self.crop_size:
            raise ValueError("crop_min must be a multiple of 32 in [32, crop_size]")
        if self.brush_encoding not in ("binary", "sdf"):
            raise ValueError("brush_encoding must be 'binary' or 'sdf'")
        if self.brush_radius_min <= 0 or self.brush_radius_max < self.brush_radius_min:
            raise ValueError("brush radius range is invalid")

        mix_total = self.mix_boundary + self.mix_near + self.mix_empty + self.mix_dense
        if abs(mix_total - 1.0) > 1e-6:
            raise ValueError(f"crop mixture must sum to 1.0, got {mix_total}")

        if self.width < 4:
            raise ValueError("width must be >= 4")
        if self.depth < 1:
            raise ValueError("depth must be >= 1")
        if self.block not in ("ds", "vanilla"):
            raise ValueError("block must be 'ds' or 'vanilla'")

        if self.train_device not in ("auto", "cpu", "cuda"):
            raise ValueError("train_device must be auto, cpu or cuda")
        if self.infer_device not in ("auto", "cpu", "cuda"):
            raise ValueError("infer_device must be auto, cpu or cuda")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        if self.backend not in ("auto", "onnx_int8", "onnx_fp32", "torch", "mock"):
            raise ValueError("backend is invalid")
        if self.mode not in ("object", "line"):
            raise ValueError("mode must be 'object' or 'line'")
        if not 1 <= self.line_width <= self.crop_size:
            raise ValueError("line_width must be in [1, crop_size]")
        if not 0 <= self.fill_close <= 16:
            raise ValueError("fill_close must be in [0, 16]")
        if not 4 <= self.magic_brush_width <= 160:
            raise ValueError("magic_brush_width must be in [4, 160]")
        if not 0.05 <= self.magic_prompt_ratio <= 1.0:
            raise ValueError("magic_prompt_ratio must be in (0.05, 1.0]")
        if not 1 <= self.magic_prompt_min <= self.magic_prompt_max <= 64:
            raise ValueError("magic prompt widths must satisfy 1 <= min <= max <= 64")
        if self.merge_weighting not in ("uniform", "gaussian"):
            raise ValueError("merge_weighting must be 'uniform' or 'gaussian'")
        if not 1 <= self.max_crops <= 256:
            raise ValueError("max_crops must be in [1, 256]")

        if not 1 <= self.port <= 65535:
            raise ValueError("port must be in [1, 65535]")
        if self.max_upload_mb < 1:
            raise ValueError("max_upload_mb must be >= 1")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _coerce(value: Any, hint: Any) -> Any:
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


def _parse_scalar(text: str) -> Any:
    if yaml is not None:
        try:
            return yaml.safe_load(text)
        except Exception:
            return text
    return text


def _env_overrides() -> dict:
    valid = {f.name for f in fields(Config)}
    out: dict[str, Any] = {}
    for key, raw in os.environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        name = key[len(ENV_PREFIX) :].lower()
        if name in valid:
            out[name] = _parse_scalar(raw)
    return out


def _parse_override(item: str) -> tuple[str, Any]:
    if "=" not in item:
        raise ValueError(f"override must look like key=value, got {item!r}")
    key, raw = item.split("=", 1)
    key = key.strip()
    if key not in {f.name for f in fields(Config)}:
        raise ValueError(f"unknown config field: {key!r}")
    return key, _parse_scalar(raw)


def add_config_args(parser) -> None:
    """Attach the standard config flag to an argparse parser."""
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set batch_size=16",
    )
