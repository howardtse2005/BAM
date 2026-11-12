import base64

import cv2
import numpy as np
import pytest

from training.stroke_synth import mask_to_paths, resample_polyline


def _encode(image: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", image)
    assert ok
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def _decode(data_url: str) -> np.ndarray:
    payload = data_url.split(",", 1)[1]
    return cv2.imdecode(np.frombuffer(base64.b64decode(payload), np.uint8), cv2.IMREAD_GRAYSCALE)


def _loop(gt: np.ndarray, thickness: int = 9, fraction: float = 1.0) -> np.ndarray:
    """A brush loop along the mask boundary; ``fraction`` < 1 leaves it open."""
    line = np.zeros_like(gt)
    for path in mask_to_paths(gt):
        closed = np.vstack([path, path[:1]])
        points = resample_polyline(closed, 8)
        keep = max(2, int(len(points) * fraction))
        cv2.polylines(line, [points[:keep].astype(np.int32)], False, 255, thickness)
    return line


def _segment_payload(app, dataset="toy", split="train", name=None, extra_params=None):
    index = app.extensions["bam_index"]
    name = name or index.list_images(dataset, split)[0]
    gt = index.load_mask(dataset, split, name)
    height, width = gt.shape

    stroke = _loop(gt)
    params = {"mode": "line", "smooth": 2, "expand": 0}
    params.update(extra_params or {})
    return {
        "dataset": dataset,
        "split": split,
        "image": name,
        "canvas": {"width": width, "height": height},
        "stroke_mask": _encode(stroke),
        "params": params,
    }, gt


def test_health(client):
    data = client.get("/api/health").get_json()
    assert data["status"] == "ok"
    assert data["model"]["available"] is True
    assert data["model"]["is_mock"] is True


def test_editor_page_renders(client):
    """The editor opens empty; nothing in the page sizes a crop for it."""
    response = client.get("/")
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "BAM_BOOTSTRAP" in body
    assert "cropStride" not in body


def test_list_datasets_reports_splits(client):
    data = client.get("/api/datasets").get_json()
    entry = next(item for item in data["datasets"] if item["name"] == "toy")
    assert entry["count"] == 6
    assert entry["splits"] == {"train": 4, "val": 1, "test": 1}
    assert entry["thumb"].startswith("/media/toy/train/im/")
    assert entry["has_edge"] is True
    assert entry["has_brush"] is True


def test_list_images_per_split(client):
    train = client.get("/api/datasets/toy/images?split=train").get_json()
    test = client.get("/api/datasets/toy/images?split=test").get_json()
    assert train["total"] == 4
    assert test["total"] == 1
    assert train["split"] == "train"


def test_list_images_rejects_bad_split(client):
    response = client.get("/api/datasets/toy/images?split=banana")
    assert response.status_code == 400


def test_image_meta(client):
    data = client.get("/api/datasets/toy/train/000000000/meta").get_json()
    assert data["width"] > 0
    assert data["has_mask"] is True
    assert data["split"] == "train"


# --------------------------------------------------------------------------- #
# /api/segment: stroke -> predicted line
# --------------------------------------------------------------------------- #


def test_segment_returns_a_line_not_a_region(client, app):
    payload, gt = _segment_payload(app)
    response = client.post("/api/segment", json=payload)
    assert response.status_code == 200

    data = response.get_json()
    assert data["status"] == "success"
    assert data["mode_used"] == "line"
    assert data["mask_base64"] is None
    # One stroke, one square, one forward pass.
    assert data["stats"]["num_crops"] == 1
    assert data["crop_size"] % 32 == 0
    assert app.config["BAM"].crop_min <= data["crop_size"] <= app.config["BAM"].crop_size

    line = _decode(data["line_base64"])
    assert line.shape == gt.shape
    # A line, not a blob: far fewer pixels than the object it traces.
    assert (line > 127).sum() < (gt > 127).sum()
    assert (line > 127).sum() > 0


def test_segment_crops_one_square_around_the_whole_stroke(client, app):
    """The window is the smallest square that holds every painted pixel."""
    from backend.inference.crop_window import snap_up

    payload, _ = _segment_payload(app)
    data = client.post("/api/segment", json=payload).get_json()

    stroke = _decode(payload["stroke_mask"]) > 127
    ys, xs = np.nonzero(stroke)
    window = data["window"]
    x0, y0, size = window["x0"], window["y0"], window["size"]

    assert data["stats"]["window_px"] == size
    assert x0 <= xs.min() and x0 + size > xs.max()
    assert y0 <= ys.min() and y0 + size > ys.max()

    cfg = app.config["BAM"]
    pad = max(2, round(cfg.magic_brush_width / 2))
    extent = max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
    assert size == max(snap_up(extent + 2 * pad), cfg.crop_min)


def test_segment_runs_a_short_stroke_at_native_scale(client, app):
    payload, gt = _segment_payload(app)
    height, width = gt.shape
    stroke = np.zeros_like(gt)
    cv2.circle(stroke, (width // 2, height // 2), 40, 255, 9)
    payload["stroke_mask"] = _encode(stroke)

    data = client.post("/api/segment", json=payload).get_json()
    assert data["scale"] == 1.0
    assert data["crop_size"] == data["window"]["size"]


def test_segment_downscales_a_window_larger_than_the_trained_crop(client, app):
    """A stroke spanning the canvas is still one window, shrunk to fit."""
    payload, gt = _segment_payload(app)
    height, width = gt.shape
    long_stroke = np.zeros_like(gt)
    cv2.line(long_stroke, (2, 2), (width - 3, height - 3), 255, 9)
    payload["stroke_mask"] = _encode(long_stroke)

    data = client.post("/api/segment", json=payload).get_json()
    cfg = app.config["BAM"]
    assert data["stats"]["num_crops"] == 1
    assert data["window"]["size"] > cfg.crop_size
    assert data["crop_size"] == cfg.crop_size
    assert 0.0 < data["scale"] < 1.0

    # Shrinking the window must not lose the prompt: the line still lands on
    # the stroke that produced it.
    line = _decode(data["line_base64"]) > 127
    stroke = _decode(payload["stroke_mask"]) > 127
    assert line.any()
    dist = cv2.distanceTransform(255 - stroke.astype(np.uint8) * 255, cv2.DIST_L2, 5)
    assert float(np.median(dist[line])) <= 3.0


def test_segment_handles_a_stroke_on_the_frame_edge(client, app):
    """A window that hangs off the canvas is padded, not rejected."""
    payload, gt = _segment_payload(app)
    height, width = gt.shape
    edge_stroke = np.zeros_like(gt)
    cv2.line(edge_stroke, (0, 0), (0, height - 1), 255, 9)
    payload["stroke_mask"] = _encode(edge_stroke)

    response = client.post("/api/segment", json=payload)
    assert response.status_code == 200
    assert response.get_json()["window"]["x0"] < 0


def test_segment_line_sits_on_the_object_boundary(client, app):
    payload, gt = _segment_payload(app)
    data = client.post("/api/segment", json=payload).get_json()

    line = _decode(data["line_base64"]) > 127
    boundary = np.zeros_like(gt)
    for path in mask_to_paths(gt):
        cv2.polylines(boundary, [path.astype(np.int32)], True, 255, 1)
    dist = cv2.distanceTransform(255 - boundary, cv2.DIST_L2, 5)

    # The mock backend predicts a band around the stroke, so the line is within
    # a few pixels of the boundary rather than exact. The real model is held to
    # a tighter bound by training/metrics.py.
    assert float(np.median(dist[line])) < 8.0


def test_segment_reports_whether_the_line_closed(client, app):
    closed_payload, _ = _segment_payload(app)
    closed = client.post("/api/segment", json=closed_payload).get_json()
    assert closed["closed"] is True

    open_payload, gt = _segment_payload(app)
    open_payload["stroke_mask"] = _encode(_loop(gt, fraction=0.45))
    opened = client.post("/api/segment", json=open_payload).get_json()
    assert opened["closed"] is False


def test_segment_object_mode_also_fills(client, app):
    from training.metrics import iou

    payload, gt = _segment_payload(app, extra_params={"mode": "object"})
    data = client.post("/api/segment", json=payload).get_json()

    assert data["mode_used"] in ("flood", "closed_gap", "grabcut", "loop")
    assert data["mask_base64"] is not None
    mask = _decode(data["mask_base64"])
    assert mask.shape == gt.shape
    assert iou(mask, gt) > 0.8


def test_segment_reads_the_requested_split(client, app):
    payload, gt = _segment_payload(app, split="val")
    response = client.post("/api/segment", json=payload)
    assert response.status_code == 200
    line = _decode(response.get_json()["line_base64"])
    assert line.shape == gt.shape


def test_segment_requires_stroke(client, app):
    payload, _ = _segment_payload(app)
    payload.pop("stroke_mask")
    response = client.post("/api/segment", json=payload)
    assert response.status_code == 400
    assert "stroke_mask" in response.get_json()["message"]


def test_segment_rejects_bad_split(client, app):
    payload, _ = _segment_payload(app)
    payload["split"] = "banana"
    response = client.post("/api/segment", json=payload)
    assert response.status_code == 400


def test_segment_rejects_an_empty_stroke(client, app):
    payload, gt = _segment_payload(app)
    payload["stroke_mask"] = _encode(np.zeros_like(gt))
    response = client.post("/api/segment", json=payload)
    assert response.status_code == 400
    assert "stroke" in response.get_json()["message"]


def test_segment_unknown_image(client, app):
    payload, _ = _segment_payload(app)
    payload["image"] = "does-not-exist"
    response = client.post("/api/segment", json=payload)
    assert response.status_code == 404


# --------------------------------------------------------------------------- #
# The brush band: the drawn stroke is wider than the prompt the model sees
# --------------------------------------------------------------------------- #


def test_segment_clips_the_line_to_the_brush_band(client, app):
    """Every pixel the magic brush shows is under the stroke the user drew."""
    from backend.inference.edge_to_mask import stroke_band

    payload, _ = _segment_payload(
        app, extra_params={"brush_width": 32, "prompt_width": 20}
    )
    data = client.post("/api/segment", json=payload).get_json()

    assert data["band_width"] == 32
    assert data["prompt_width"] == 20

    stroke = _decode(payload["stroke_mask"])
    line = _decode(data["line_base64"]) > 127
    band = stroke_band(stroke, data["band_width"], data["prompt_width"])
    distance = cv2.distanceTransform((band == 0).astype(np.uint8), cv2.DIST_L2, 3)

    # One px of slack: the line is a ribbon grown from a 1-px skeleton, so its
    # own thickness can reach just past the band edge it was clipped to.
    assert float(distance[line].max()) <= 1.5


def test_segment_band_trims_a_prediction_wider_than_the_stroke(client, app):
    """A band with no tolerance cuts the prediction back to the stroke itself."""
    wide, _ = _segment_payload(
        app, extra_params={"brush_width": 40, "prompt_width": 12}
    )
    tight, _ = _segment_payload(
        app, extra_params={"brush_width": 12, "prompt_width": 12}
    )

    wide_stats = client.post("/api/segment", json=wide).get_json()["stats"]
    tight_stats = client.post("/api/segment", json=tight).get_json()["stats"]

    assert tight_stats["clipped_px"] > 0
    assert wide_stats["clipped_px"] == 0
    assert tight_stats["predicted_px"] > tight_stats["clipped_px"]


def test_segment_clamps_the_prompt_inside_the_band(client, app):
    payload, _ = _segment_payload(
        app, extra_params={"brush_width": 24, "prompt_width": 999}
    )
    data = client.post("/api/segment", json=payload).get_json()
    assert data["band_width"] == 24
    assert data["prompt_width"] == 24


# --------------------------------------------------------------------------- #
# /api/fill: accepted line -> region
# --------------------------------------------------------------------------- #


def _fill_payload(app, line, extra_params=None):
    index = app.extensions["bam_index"]
    name = index.list_images("toy", "train")[0]
    height, width = line.shape
    params = {"smooth": 2, "expand": 0}
    params.update(extra_params or {})
    return {
        "dataset": "toy",
        "split": "train",
        "image": name,
        "canvas": {"width": width, "height": height},
        "line_mask": _encode(line),
        "params": params,
    }


def test_fill_turns_a_closed_line_into_a_region(client, app):
    from training.metrics import iou

    index = app.extensions["bam_index"]
    gt = index.load_mask("toy", "train", index.list_images("toy", "train")[0])

    # An accepted line is the model's 3 px output, not the 9 px brush that
    # produced it: the fill covers the line, so its thickness is the bias.
    response = client.post("/api/fill", json=_fill_payload(app, _loop(gt, thickness=3)))
    assert response.status_code == 200

    data = response.get_json()
    assert data["grew"] is True
    assert data["added_px"] > 0

    mask = _decode(data["mask_base64"])
    assert mask.shape == gt.shape
    assert iou(mask, gt) > 0.85


def test_fill_preserves_holes(client, app):
    index = app.extensions["bam_index"]
    gt = index.load_mask("toy", "train", index.list_images("toy", "train")[0])

    holed = gt.copy()
    cv2.circle(holed, (holed.shape[1] // 2, holed.shape[0] // 2), 12, 0, -1)

    data = client.post(
        "/api/fill", json=_fill_payload(app, _loop(holed, thickness=3))
    ).get_json()
    mask = _decode(data["mask_base64"])
    assert mask[holed.shape[0] // 2, holed.shape[1] // 2] == 0


def test_fill_reports_an_open_line_instead_of_guessing(client, app):
    index = app.extensions["bam_index"]
    gt = index.load_mask("toy", "train", index.list_images("toy", "train")[0])

    data = client.post(
        "/api/fill", json=_fill_payload(app, _loop(gt, thickness=3, fraction=0.45))
    ).get_json()
    assert data["grew"] is False
    assert data["added_px"] == 0


def test_fill_requires_a_line(client, app):
    payload = _fill_payload(app, np.zeros((260, 340), np.uint8))
    payload.pop("line_mask")
    response = client.post("/api/fill", json=payload)
    assert response.status_code == 400
    assert "line_mask" in response.get_json()["message"]


# --------------------------------------------------------------------------- #
# Everything else
# --------------------------------------------------------------------------- #


def test_upload_then_save_mask(client):
    # Use a throwaway dataset so the upload cannot pollute the shared toy split.
    dataset = "uploads"
    image = np.zeros((240, 320, 3), np.uint8)
    cv2.rectangle(image, (60, 50), (260, 190), (40, 160, 90), -1)
    ok, buf = cv2.imencode(".png", image)
    data_url = "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()

    upload = client.post(
        "/api/image/upload",
        json={
            "dataset": dataset,
            "split": "train",
            "name": "uploaded.png",
            "image_base64": data_url,
        },
    )
    assert upload.status_code == 200
    name = upload.get_json()["image"]

    mask = np.zeros((240, 320), np.uint8)
    mask[50:190, 60:260] = 255
    saved = client.post(
        "/api/mask/save",
        json={
            "dataset": dataset,
            "split": "train",
            "image": name,
            "mask_base64": _encode(mask),
        },
    )
    assert saved.status_code == 200

    fetched = client.get(f"/api/mask/{dataset}/train/{name}")
    assert fetched.status_code == 200
    restored = _decode(fetched.get_json()["mask_base64"])
    assert np.array_equal(restored, mask)


def test_media_served(client):
    response = client.get("/media/toy/train/im/000000000.jpg")
    assert response.status_code == 200
    assert response.mimetype == "image/jpeg"


@pytest.mark.parametrize("sub", ["gt_edge", "gt_brush"])
def test_media_serves_the_generated_rasters(client, sub):
    response = client.get(f"/media/toy/train/{sub}/000000000.png")
    assert response.status_code == 200


@pytest.mark.parametrize(
    "url",
    [
        "/media/toy/train/im/..%2F..%2Fsecret.png",
        "/media/toy/train/im/....%2F%2Fsecret.png",
        "/api/mask/toy/train/..%2F..%2Fsecret",
    ],
)
def test_path_traversal_is_rejected(client, url):
    response = client.get(url)
    assert response.status_code in (400, 404)


def test_reference_predict_crops(client, app):
    index = app.extensions["bam_index"]
    name = index.list_images("toy", "train")[0]
    rgb = index.load_rgb("toy", "train", name)
    gt = index.load_mask("toy", "train", name)
    height, width = gt.shape
    size = 128  # the reference client sends its own patches; any legal size works

    crops = []
    for i in range(3):
        y0 = min(max(0, height // 2 - size // 2 + i * 20), height - size)
        x0 = min(max(0, width // 2 - size // 2 + i * 20), width - size)
        patch = rgb[y0 : y0 + size, x0 : x0 + size]
        ok, buf = cv2.imencode(".png", cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
        crops.append(
            {
                "image_base64": "data:image/png;base64,"
                + base64.b64encode(buf.tobytes()).decode(),
                "centerX": x0 + size // 2,
                "centerY": y0 + size // 2,
                "width": size,
                "height": size,
                "canvas_width": width,
                "canvas_height": height,
            }
        )

    response = client.post(
        "/api/magic_brush/predict_crops",
        json={"crops": crops, "sensitivity": 1.0, "threshold": 0.5},
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["num_crops_processed"] == 3
    assert data["degraded"] is True
    assert data["merged_mask_base64"].startswith("data:image/png;base64,")


def test_reference_single_crop(client, app):
    index = app.extensions["bam_index"]
    rgb = index.load_rgb("toy", "train", index.list_images("toy", "train")[0])
    patch = rgb[20:148, 20:148]
    ok, buf = cv2.imencode(".png", cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
    response = client.post(
        "/api/magic_brush/predict_single_crop",
        json={
            "image_base64": "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode(),
            "centerX": 84,
            "centerY": 84,
        },
    )
    assert response.status_code == 200
    assert response.get_json()["prediction_base64"].startswith("data:image/png;base64,")


def test_config_endpoint(client, app):
    data = client.get("/api/config").get_json()["config"]
    cfg = app.config["BAM"]
    assert data["crop_size"] == 256
    assert data["crop_min"] == 64
    assert data["magic_brush_width"] == cfg.magic_brush_width
    assert data["params"]["brush_width"] == cfg.magic_brush_width
    assert data["params"]["prompt_width"] == cfg.prompt_width_for()
    assert data["splits"] == ["train", "val", "test"]


def test_413_on_oversized_payload(client):
    client.application.config["MAX_CONTENT_LENGTH"] = 1024
    response = client.post("/api/segment", json={"blob": "x" * 5000})
    assert response.status_code == 413
