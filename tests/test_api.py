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


def _segment_payload(app, dataset="toy", name=None, extra_params=None):
    index = app.extensions["bam_index"]
    name = name or index.list_images(dataset)[0]
    gt = index.load_mask(dataset, name)
    height, width = gt.shape

    stroke = np.zeros_like(gt)
    boxes = []
    last = None
    for path in mask_to_paths(gt):
        points = resample_polyline(np.vstack([path, path[:1]]), 8)
        cv2.polylines(stroke, [points.astype(np.int32)], False, 255, 9)
        for x, y in points:
            if last is None or np.hypot(x - last[0], y - last[1]) >= 128:
                boxes.append(
                    {"x0": int(round(x - 128)), "y0": int(round(y - 128)), "size": 256}
                )
                last = (x, y)

    params = {"mode": "object", "smooth": 2, "expand": 0}
    params.update(extra_params or {})
    payload = {
        "dataset": dataset,
        "image": name,
        "canvas": {"width": width, "height": height},
        "crops": boxes[:64],
        "stroke_mask": _encode(stroke),
        "params": params,
    }
    return payload, gt


def test_health(client):
    data = client.get("/api/health").get_json()
    assert data["status"] == "ok"
    assert data["model"]["available"] is True
    assert data["model"]["is_mock"] is True


def test_list_datasets(client):
    data = client.get("/api/datasets").get_json()
    names = [item["name"] for item in data["datasets"]]
    assert "toy" in names
    entry = next(item for item in data["datasets"] if item["name"] == "toy")
    assert entry["count"] == 6
    assert entry["thumb"].endswith(".jpg")


def test_list_images(client):
    data = client.get("/api/datasets/toy/images").get_json()
    assert data["total"] == 6
    assert len(data["images"]) == 6


def test_image_meta(client):
    data = client.get("/api/datasets/toy/000000000/meta").get_json()
    assert data["width"] > 0
    assert data["has_mask"] is True


def test_segment_success(client, app):
    from training.metrics import iou

    payload, gt = _segment_payload(app)
    response = client.post("/api/segment", json=payload)
    assert response.status_code == 200

    data = response.get_json()
    assert data["status"] == "success"
    assert data["mode_used"] in ("flood", "closed_gap", "grabcut", "loop")
    assert data["stats"]["num_crops"] > 0
    assert data["stats"]["ms_total"] >= 0

    mask = _decode(data["mask_base64"])
    assert mask.shape == gt.shape
    assert iou(mask, gt) > 0.8


def test_segment_requires_stroke(client, app):
    payload, _ = _segment_payload(app)
    payload.pop("stroke_mask")
    response = client.post("/api/segment", json=payload)
    assert response.status_code == 400
    assert "stroke_mask" in response.get_json()["message"]


def test_segment_rejects_too_many_crops(client, app):
    payload, _ = _segment_payload(app)
    template = payload["crops"][0]
    payload["crops"] = [dict(template) for _ in range(65)]
    response = client.post("/api/segment", json=payload)
    assert response.status_code == 400


def test_segment_unknown_image(client, app):
    payload, _ = _segment_payload(app)
    payload["image"] = "does-not-exist"
    response = client.post("/api/segment", json=payload)
    assert response.status_code == 404


def test_segment_rejects_bad_crop_size(client, app):
    payload, _ = _segment_payload(app)
    payload["crops"][0]["size"] = 128
    response = client.post("/api/segment", json=payload)
    assert response.status_code == 400


def test_segment_line_mode(client, app):
    payload, _ = _segment_payload(app, extra_params={"mode": "line"})
    data = client.post("/api/segment", json=payload).get_json()
    assert data["mode_used"] == "line"


def test_upload_then_save_mask(client):
    image = np.zeros((240, 320, 3), np.uint8)
    cv2.rectangle(image, (60, 50), (260, 190), (40, 160, 90), -1)
    ok, buf = cv2.imencode(".png", image)
    data_url = "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()

    upload = client.post(
        "/api/image/upload",
        json={"dataset": "toy", "name": "uploaded.png", "image_base64": data_url},
    )
    assert upload.status_code == 200
    name = upload.get_json()["image"]

    mask = np.zeros((240, 320), np.uint8)
    mask[50:190, 60:260] = 255
    saved = client.post(
        "/api/mask/save",
        json={"dataset": "toy", "image": name, "mask_base64": _encode(mask)},
    )
    assert saved.status_code == 200

    fetched = client.get(f"/api/mask/toy/{name}")
    assert fetched.status_code == 200
    restored = _decode(fetched.get_json()["mask_base64"])
    assert np.array_equal(restored, mask)


def test_media_served(client):
    response = client.get("/media/toy/rgb/000000000.jpg")
    assert response.status_code == 200
    assert response.mimetype == "image/jpeg"


@pytest.mark.parametrize(
    "url",
    [
        "/media/toy/rgb/..%2F..%2Fsecret.png",
        "/media/toy/rgb/....%2F%2Fsecret.png",
        "/api/mask/toy/..%2F..%2Fsecret",
    ],
)
def test_path_traversal_is_rejected(client, url):
    response = client.get(url)
    assert response.status_code in (400, 404)


def test_legacy_predict_crops(client, app):
    index = app.extensions["bam_index"]
    name = index.list_images("toy")[0]
    rgb = index.load_rgb("toy", name)
    gt = index.load_mask("toy", name)
    height, width = gt.shape

    crops = []
    for i in range(3):
        y0 = min(max(0, height // 2 - 128 + i * 40), height - 256)
        x0 = min(max(0, width // 2 - 128 + i * 40), width - 256)
        patch = rgb[y0 : y0 + 256, x0 : x0 + 256]
        ok, buf = cv2.imencode(".png", cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
        crops.append(
            {
                "image_base64": "data:image/png;base64,"
                + base64.b64encode(buf.tobytes()).decode(),
                "centerX": x0 + 128,
                "centerY": y0 + 128,
                "width": 256,
                "height": 256,
                "canvas_width": width,
                "canvas_height": height,
            }
        )

    response = client.post(
        "/magic_pen/predict_crops", json={"crops": crops, "sensitivity": 1.0, "threshold": 0.5}
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["num_crops_processed"] == 3
    assert data["degraded"] is True
    assert data["merged_mask_base64"].startswith("data:image/png;base64,")


def test_legacy_single_crop(client, app):
    index = app.extensions["bam_index"]
    rgb = index.load_rgb("toy", index.list_images("toy")[0])
    patch = rgb[20:276, 20:276]
    ok, buf = cv2.imencode(".png", cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
    response = client.post(
        "/magic_pen/predict_single_crop",
        json={
            "image_base64": "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode(),
            "centerX": 148,
            "centerY": 148,
        },
    )
    assert response.status_code == 200
    assert response.get_json()["prediction_base64"].startswith("data:image/png;base64,")


def test_config_endpoint(client):
    data = client.get("/api/config").get_json()["config"]
    assert data["crop_size"] == 256
    assert data["stride"] == 128


def test_413_on_oversized_payload(client):
    client.application.config["MAX_CONTENT_LENGTH"] = 1024
    response = client.post("/api/segment", json={"blob": "x" * 5000})
    assert response.status_code == 413
