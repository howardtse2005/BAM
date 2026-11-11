"""Dataset listing and media serving."""

from __future__ import annotations

from flask import Blueprint, request, send_file

from ..deps import cfg, fail, index, ok
from ..schemas import ValidationError, safe_name
from ..services.storage import StorageError

bp = Blueprint("datasets_api", __name__)

MEDIA_SUBDIRS = {"rgb", "gt", "gt_edge", "labels"}


@bp.get("/api/datasets")
def list_datasets():
    return ok(datasets=index().list_datasets())


@bp.get("/api/datasets/<dataset>/images")
def list_images(dataset: str):
    try:
        safe_name(dataset, "dataset")
        names = index().list_images(dataset)
    except ValidationError as exc:
        return fail(exc, 400)
    except KeyError as exc:
        return fail(exc, 404)

    offset = max(0, int(request.args.get("offset", 0)))
    limit = min(1000, max(1, int(request.args.get("limit", 500))))
    window = names[offset : offset + limit]
    return ok(
        dataset=dataset,
        total=len(names),
        offset=offset,
        limit=limit,
        images=window,
    )


@bp.get("/api/datasets/<dataset>/<image>/meta")
def image_meta(dataset: str, image: str):
    try:
        safe_name(dataset, "dataset")
        safe_name(image, "image")
        rgb = index().load_rgb(dataset, image)
        has_mask = index().mask_path(dataset, image) is not None
    except ValidationError as exc:
        return fail(exc, 400)
    except KeyError as exc:
        return fail(exc, 404)

    return ok(
        dataset=dataset,
        image=image,
        width=int(rgb.shape[1]),
        height=int(rgb.shape[0]),
        has_mask=bool(has_mask),
        canonical_short_side=cfg().data.canonical_short_side,
    )


@bp.get("/media/<dataset>/<sub>/<path:name>")
def media(dataset: str, sub: str, name: str):
    if sub not in MEDIA_SUBDIRS:
        return fail("invalid media type", 404)
    try:
        safe_name(dataset, "dataset")
        safe_name(name, "filename")
        # Resolve through the index so /media/demo/rgb/000000000 also works when
        # the stored file is 000000000.jpg.
        path = index().resolve(dataset, sub, name)
    except (ValidationError, StorageError) as exc:
        return fail(exc, 400)
    except KeyError as exc:
        return fail(exc, 404)

    if path is None or not path.is_file():
        return fail("file not found", 404)
    return send_file(path, conditional=True, max_age=3600)


__all__ = ["bp"]
