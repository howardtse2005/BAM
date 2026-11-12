"""BAM UNet: a 4-channel, CPU-friendly edge refiner.

The topology is a standard UNet (encoder / bottleneck / decoder with skip
connections). The only deviation is the convolution block: depthwise-separable
convolutions instead of dense 3x3 convolutions, which cuts spatial cost by
roughly 6-8x at equal width. That is what makes sub-40 ms CPU inference possible.

The model is deliberately self-contained: it imports nothing from ``config`` and
takes every parameter through the constructor. (The reference masker repository
has ``model/unet.py`` importing ``from config import Config`` while ``config.py``
is absent, which makes the model impossible to import on its own.)

Input:  ``[N, 4, H, W]`` float tensor -- RGB + brush channel.
Output: ``[N, 1, H, W]`` logits. Apply sigmoid for probabilities.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Blocks
# --------------------------------------------------------------------------- #


class DSConv(nn.Module):
    """Depthwise separable convolution: 3x3 depthwise -> 1x1 pointwise."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.dw = nn.Conv2d(
            in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False
        )
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        return x


class DenseConv(nn.Module):
    """Plain 3x3 convolution block, kept for ablation (``block: vanilla``)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DoubleBlock(nn.Module):
    """Two stacked conv blocks, the UNet workhorse."""

    def __init__(self, in_ch: int, out_ch: int, block: str = "ds") -> None:
        super().__init__()
        factory = DSConv if block == "ds" else DenseConv
        self.block1 = factory(in_ch, out_ch)
        self.block2 = factory(out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block2(self.block1(x))


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


class BAMUNet(nn.Module):
    """Lightweight UNet for brush-conditioned edge prediction."""

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        width: int = 20,
        depth: int = 4,
        max_width: int = 256,
        block: str = "ds",
        bilinear: bool = True,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if block not in ("ds", "vanilla"):
            raise ValueError("block must be 'ds' or 'vanilla'")
        if in_channels < 1:
            raise ValueError("in_channels must be >= 1")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.depth = depth
        self.block = block
        self.bilinear = bilinear

        widths: list[int] = [min(width * (2**i), max_width) for i in range(depth + 1)]
        self.widths = widths

        # Encoder
        enc: list[nn.Module] = []
        for i, w in enumerate(widths):
            enc.append(DoubleBlock(in_channels if i == 0 else widths[i - 1], w, block))
        self.encoders = nn.ModuleList(enc)
        self.pools = nn.ModuleList([nn.MaxPool2d(2) for _ in range(depth)])

        # Decoder
        ups: list[nn.Module] = []
        decs: list[nn.Module] = []
        for i in range(depth, 0, -1):
            cin, cout = widths[i], widths[i - 1]
            if bilinear:
                up: nn.Module = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(cin, cout, 1, bias=False),
                    nn.BatchNorm2d(cout),
                    nn.ReLU(inplace=True),
                )
            else:
                up = nn.Sequential(
                    nn.ConvTranspose2d(cin, cout, 2, stride=2),
                    nn.BatchNorm2d(cout),
                    nn.ReLU(inplace=True),
                )
            ups.append(up)
            decs.append(DoubleBlock(cout * 2, cout, block))
        self.ups = nn.ModuleList(ups)
        self.decoders = nn.ModuleList(decs)

        self.head = nn.Conv2d(widths[0], out_channels, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Start from a low-confidence edge map so early training is stable.
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, -2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"expected [N, C, H, W], got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, got {x.shape[1]}"
            )

        skips: list[torch.Tensor] = []
        h = x
        for i, enc in enumerate(self.encoders):
            h = enc(h)
            if i < self.depth:
                skips.append(h)
                h = self.pools[i](h)

        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            h = up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(
                    h, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            h = dec(torch.cat([skip, h], dim=1))

        return self.head(h)

    # -- introspection ----------------------------------------------------- #

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def config_dict(self) -> dict:
        return {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "width": self.width,
            "depth": self.depth,
            "block": self.block,
            "bilinear": self.bilinear,
        }


def build_model(cfg) -> BAMUNet:
    """Build a model from a Config, a mapping, or any attribute holder."""
    if cfg is None:
        return BAMUNet()

    if isinstance(cfg, dict):
        get = cfg.get
    elif hasattr(cfg, "model"):
        get = lambda k, d=None: getattr(cfg.model, k, d)  # noqa: E731
    else:
        get = lambda k, d=None: getattr(cfg, k, d)  # noqa: E731

    return BAMUNet(
        in_channels=int(get("in_channels", 4)),
        out_channels=int(get("out_channels", 1)),
        width=int(get("width", 20)),
        depth=int(get("depth", 4)),
        max_width=int(get("max_width", 256)),
        block=str(get("block", "ds")),
        bilinear=bool(get("bilinear", True)),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_checkpoint_model(path, map_location="cpu") -> "BAMUNet":
    """Rebuild a BAMUNet from a training checkpoint.

    The checkpoint carries its own ``model_config``, so the architecture is never
    guessed from a global config file. This is the only supported way to load
    weights in the backend.
    """
    import torch as _torch

    try:
        ckpt = _torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        ckpt = _torch.load(path, map_location=map_location, weights_only=False)

    if not isinstance(ckpt, dict):
        raise ValueError(f"unexpected checkpoint format in {path}")

    model_cfg = ckpt.get("model_config") or ckpt.get("config", {}).get("model", {})
    model = BAMUNet(
        in_channels=int(model_cfg.get("in_channels", 4)),
        out_channels=int(model_cfg.get("out_channels", 1)),
        width=int(model_cfg.get("width", 20)),
        depth=int(model_cfg.get("depth", 4)),
        max_width=int(model_cfg.get("max_width", 256)),
        block=str(model_cfg.get("block", "ds")),
        bilinear=bool(model_cfg.get("bilinear", True)),
    )
    state = ckpt.get("model_state") or ckpt.get("state_dict") or ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def estimate_gflops(
    model: nn.Module, size: int = 256, channels: int = 4, iters: int = 3
) -> float:
    """Rough forward-pass cost in GFLOPs (2 * MACs), measured by hook counting."""
    macs = 0

    def conv_hook(module: nn.Conv2d, inputs: Iterable[torch.Tensor], output):
        nonlocal macs
        out = output[0] if isinstance(output, (tuple, list)) else output
        kernel = module.kernel_size[0] * module.kernel_size[1]
        macs += out.numel() * (module.in_channels // module.groups) * kernel

    handles = [m.register_forward_hook(conv_hook) for m in model.modules() if isinstance(m, nn.Conv2d)]
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            dummy = torch.zeros(1, channels, size, size)
            for _ in range(iters):
                model(dummy)
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)

    return 2.0 * macs / max(iters, 1) / 1e9


__all__ = [
    "BAMUNet",
    "DSConv",
    "DenseConv",
    "DoubleBlock",
    "build_model",
    "count_parameters",
    "load_checkpoint_model",
    "estimate_gflops",
]
