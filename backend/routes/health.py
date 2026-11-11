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
    cfg_obj = cfg()
    return ok(
        config={
            "crop_size": cfg_obj.infer.crop_size,
            "stride": cfg_obj.infer.stride,
            "max_crops": cfg_obj.infer.max_crops,
            "working_short_side": cfg_obj.data.canonical_short_side,
            "params": {
                "sensitivity": cfg_obj.infer.sensitivity,
                "threshold": cfg_obj.infer.threshold,
                "edge_threshold": cfg_obj.infer.edge_threshold,
                "expand": cfg_obj.infer.expand,
                "smooth": cfg_obj.infer.smooth,
                "mode": cfg_obj.infer.mode,
                "merge_weighting": cfg_obj.infer.merge_weighting,
            },
        }
    )


__all__ = ["bp"]
