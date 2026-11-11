"""HTML routes."""

from __future__ import annotations

from flask import Blueprint, redirect, render_template, request, url_for

from ..deps import cfg, index

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
        dataset_root=str(cfg().dataset_root),
        model=cfg().paths.model,
    )


@bp.get("/masking")
def masking():
    dataset = request.args.get("dataset", "")
    image = request.args.get("image", "")
    if not dataset:
        return redirect(url_for("pages.datasets"))
    return render_template(
        "index.html",
        dataset=dataset,
        image=image,
        crop_size=cfg().infer.crop_size,
        stride=cfg().infer.stride,
        max_crops=cfg().infer.max_crops,
        params=cfg().infer,
    )


__all__ = ["bp"]
