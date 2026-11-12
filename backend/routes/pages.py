"""HTML routes."""

from __future__ import annotations

from flask import Blueprint, redirect, render_template, request, url_for

from ..deps import cfg, index
from ..schemas import VALID_SPLITS

bp = Blueprint("pages", __name__)


@bp.get("/")
def home():
    return datasets()


@bp.get("/datasets")
def datasets():
    try:
        items = index().list_datasets()
    except Exception as exc:  # a missing datasets folder should not 500 the app
        items = []
        print(f"[bam] could not list datasets: {exc}")
    return render_template(
        "select_dataset.html",
        datasets=items,
        datasets_root=str(cfg().datasets_root),
        model=cfg().model_path,
    )


@bp.get("/masking")
def masking():
    dataset = request.args.get("dataset", "")
    split = request.args.get("split", "train")
    if split not in VALID_SPLITS:
        split = "train"
    image = request.args.get("image", "")
    if not dataset:
        return redirect(url_for("pages.datasets"))
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
    )


__all__ = ["bp"]
