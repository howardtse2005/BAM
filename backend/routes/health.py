"""Health, warmup and model introspection."""

from __future__ import annotations

import platform
import sys

from flask import Blueprint

from ..deps import cfg, fail, index, ok, registry

bp = Blueprint("health", __name__)


@bp.get("/api/health")
def health():
    model = registry().info()
    return ok(
        status="ok" if model["available"] else "degraded",
        version="1.0.0",
        model=model,
        datasets=index().stats(),
        python=platform.python_version(),
        platform=sys.platform,
        config_fingerprint=cfg().fingerprint(),
    )


@bp.get("/api/model_info")
def model_info():
    return ok(model=registry().info())


@bp.post("/api/warmup")
def warmup():
    try:
        elapsed = registry().warmup()
    except Exception as exc:
        return fail(str(exc), 503)
    return ok(ms=round(elapsed, 2), model=registry().info())


@bp.get("/api/config")
def public_config():
    """The subset of config the UI needs. Never exposes filesystem paths."""
    c = cfg()
    return ok(
        config={
            "dataset": c.dataset,
            "crop_size": c.crop_size,
            "crop_min": c.crop_min,
            "crop_ratio": c.crop_ratio,
            "crop_stride": c.crop_stride,
            "max_crops": c.max_crops,
            "short_side": c.short_side,
            "line_width": c.line_width,
            "fill_close": c.fill_close,
            "splits": ["train", "val", "test"],
            "params": {
                "sensitivity": c.sensitivity,
                "threshold": c.threshold,
                "edge_threshold": c.edge_threshold,
                "expand": c.expand,
                "smooth": c.smooth,
                "mode": c.mode,
                "line_width": c.line_width,
                "fill_close": c.fill_close,
                "merge_weighting": c.merge_weighting,
            },
        }
    )


__all__ = ["bp"]
