"""HTML routes.

The editor is the app: there is no dataset browser in front of it. It opens
empty and the user uploads an image, which is stored under ``upload_dataset``.
"""

from __future__ import annotations

from flask import Blueprint, render_template, request

from ..deps import cfg
from ..schemas import VALID_SPLITS

bp = Blueprint("pages", __name__)


@bp.get("/")
def home():
    return masking()


@bp.get("/masking")
def masking():
    dataset = request.args.get("dataset") or cfg().upload_dataset
    split = request.args.get("split", "train")
    if split not in VALID_SPLITS:
        split = "train"
    image = request.args.get("image", "")
    return render_template(
        "index.html",
        dataset=dataset,
        split=split,
        image=image,
        crop_size=cfg().crop_size,
        crop_min=cfg().crop_min,
        crop_ratio=cfg().crop_ratio,
        crop_stride=cfg().crop_stride,
        max_crops=cfg().max_crops,
        line_width=cfg().line_width,
        fill_close=cfg().fill_close,
        magic_brush_width=cfg().magic_brush_width,
        magic_prompt_ratio=cfg().magic_prompt_ratio,
        magic_prompt_min=cfg().magic_prompt_min,
        magic_prompt_max=cfg().magic_prompt_max,
    )


__all__ = ["bp"]
