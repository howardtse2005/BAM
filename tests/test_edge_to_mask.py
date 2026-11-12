import cv2
import numpy as np
import pytest

from backend.inference.edge_to_mask import (
    edge_to_mask,
    fill_holes,
    guided_filter,
    stroke_band,
    thin,
)
from training.metrics import iou
from training.stroke_synth import boundary_from_mask, mask_to_paths, resample_polyline


def _scene(height=320, width=420, hole=True):
    gt = np.zeros((height, width), np.uint8)
    cv2.ellipse(gt, (width // 2, height // 2), (120, 85), 12, 0, 360, 255, -1)
    if hole:
        cv2.circle(gt, (width // 2, height // 2), 26, 0, -1)

    boundary = boundary_from_mask(gt).astype(np.float32) / 255.0
    edge = cv2.GaussianBlur(boundary, (0, 0), 1.2)
    edge = edge / edge.max()
    return gt, edge


def _stroke(gt, thickness=9):
    stroke = np.zeros_like(gt)
    for path in mask_to_paths(gt):
        points = resample_polyline(np.vstack([path, path[:1]]), 8).astype(np.int32)
        cv2.polylines(stroke, [points], False, 255, thickness)
    return stroke


def test_thin_reduces_a_band_to_a_line():
    band = np.zeros((80, 80), np.uint8)
    band[30:50, 10:70] = 255
    skeleton = thin(band)
    assert 0 < (skeleton > 0).sum() < (band > 0).sum() / 5


def test_thin_handles_empty():
    assert thin(np.zeros((10, 10), np.uint8)).max() == 0


def test_fill_holes_closes_inner_ring():
    mask = np.zeros((60, 60), np.uint8)
    cv2.rectangle(mask, (10, 10), (50, 50), 255, -1)
    cv2.circle(mask, (30, 30), 8, 0, -1)
    assert fill_holes(mask)[30, 30] == 255


def test_guided_filter_stays_in_range():
    guide = np.random.default_rng(0).integers(0, 255, (64, 64), dtype=np.uint8)
    src = np.zeros((64, 64), np.float32)
    src[20:44, 20:44] = 1.0
    out = guided_filter(guide, src, radius=4)
    assert out.shape == src.shape
    assert out.min() >= -0.5
    assert out.max() <= 1.5


def test_closed_loop_produces_accurate_mask():
    gt, edge = _scene()
    stroke = _stroke(gt)
    result = edge_to_mask(edge, stroke, rgb=None, smooth=0)

    assert result.mode in ("flood", "closed_gap")
    assert result.confidence > 0.6
    assert iou(result.mask, gt) > 0.90


def test_hole_is_preserved():
    gt, edge = _scene(hole=True)
    stroke = _stroke(gt)
    result = edge_to_mask(edge, stroke, rgb=None, smooth=0)

    centre = result.mask.shape[0] // 2, result.mask.shape[1] // 2
    assert result.mask[centre] == 0
    assert gt[centre] == 0


def test_open_stroke_returns_edge_not_a_blob():
    gt, edge = _scene(hole=False)
    stroke = np.zeros_like(gt)
    path = mask_to_paths(gt)[0]
    half = path[: len(path) // 2]
    points = resample_polyline(half, 8).astype(np.int32)
    cv2.polylines(stroke, [points], False, 255, 9)

    result = edge_to_mask(edge, stroke, rgb=None, smooth=0)
    assert result.mode == "open_loop"
    assert result.low_confidence is True
    # The returned mask is the thin edge, so its area is far below the object.
    assert (result.mask > 0).sum() < 0.1 * gt.size


def test_line_mode_returns_ribbon():
    gt, edge = _scene()
    stroke = _stroke(gt)
    result = edge_to_mask(edge, stroke, rgb=None, smooth=2, mode="line")
    assert result.mode == "line"
    assert (result.mask > 0).sum() > (result.edge > 0).sum()


def test_empty_edge_returns_empty():
    gt, _ = _scene()
    stroke = _stroke(gt)
    result = edge_to_mask(np.zeros_like(gt, np.float32), stroke)
    assert result.mode == "empty"
    assert result.mask.max() == 0


def test_expand_grows_the_mask():
    gt, edge = _scene(hole=False)
    stroke = _stroke(gt)
    base = edge_to_mask(edge, stroke, rgb=None, smooth=0, expand=0)
    grown = edge_to_mask(edge, stroke, rgb=None, smooth=0, expand=4)
    assert (grown.mask > 0).sum() > (base.mask > 0).sum()


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        edge_to_mask(np.zeros((10, 10), np.float32), np.zeros((12, 12), np.uint8))


# --------------------------------------------------------------------------- #
# Brush band: the visible stroke is wider than the prompt the model sees
# --------------------------------------------------------------------------- #


def test_band_grows_the_prompt_by_the_half_width_difference():
    prompt = np.zeros((120, 120), np.uint8)
    prompt[60, 10:110] = 255

    band = stroke_band(prompt, band_width=32, prompt_width=20)
    # (32 - 20) / 2 = 6 px of tolerance on each side of the prompt.
    assert band[60, 60] == 255
    assert band[60 - 5, 60] == 255
    assert band[60 + 5, 60] == 255
    assert band[60 - 7, 60] == 0
    assert band[60 + 7, 60] == 0


def test_band_contains_the_prompt():
    prompt = np.zeros((80, 80), np.uint8)
    cv2.line(prompt, (10, 40), (70, 40), 255, 7)

    band = stroke_band(prompt, band_width=24, prompt_width=7)
    assert not np.any((prompt > 0) & (band == 0))


def test_band_inset_keeps_the_painted_line_inside():
    prompt = np.zeros((120, 120), np.uint8)
    prompt[60, 10:110] = 255

    full = stroke_band(prompt, band_width=32, prompt_width=20)
    inset = stroke_band(prompt, band_width=32, prompt_width=20, inset=1)
    assert (inset > 0).sum() < (full > 0).sum()
    assert not np.any((inset > 0) & (full == 0))


def test_band_never_shrinks_below_the_prompt():
    prompt = np.zeros((40, 40), np.uint8)
    cv2.line(prompt, (5, 20), (35, 20), 255, 9)

    # A prompt wider than the band is a client bug; the band still has to cover
    # the stroke that was actually drawn, never a subset of it.
    band = stroke_band(prompt, band_width=4, prompt_width=20)
    assert np.array_equal(band, prompt)


def test_band_of_an_empty_stroke_is_empty():
    band = stroke_band(np.zeros((32, 32), np.uint8), 32, 20)
    assert band.max() == 0
