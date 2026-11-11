import cv2
import numpy as np

from training.stroke_synth import (
    StrokeParams,
    boundary_band,
    boundary_from_mask,
    encode_stroke,
    ensure_binary,
    fill_holes,
    mask_to_paths,
    polyline_length,
    resample_polyline,
    stroke_distance,
    synthesize_stroke,
)


def _square(size=200, margin=40):
    mask = np.zeros((size, size), np.uint8)
    mask[margin:-margin, margin:-margin] = 255
    return mask


def test_ensure_binary_normalises():
    mask = np.zeros((10, 10), np.uint8)
    mask[2:6, 2:6] = 1
    out = ensure_binary(mask)
    assert out.dtype == np.uint8
    assert set(np.unique(out)).issubset({0, 255})
    assert out.sum() == 16 * 255


def test_boundary_is_one_pixel_ring():
    mask = _square(120, 30)
    boundary = boundary_from_mask(mask)
    ys, xs = np.nonzero(boundary)
    assert ys.size > 0
    assert ys.min() == 30
    assert xs.min() == 30
    # The ring must be thin: its area is far below the mask area.
    assert boundary.sum() / 255 < mask.sum() / 255 / 4


def test_paths_are_ordered_and_closed():
    mask = _square(160, 40)
    paths = mask_to_paths(mask)
    assert len(paths) >= 1
    path = paths[0]
    assert path.shape[1] == 2
    assert polyline_length(path, closed=True) > 300
    # Consecutive points must be neighbours, not a shuffled set.
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    assert steps.max() < 6


def test_resample_spacing():
    line = np.array([[0, 0], [100, 0]], np.float32)
    out = resample_polyline(line, 10)
    steps = np.linalg.norm(np.diff(out, axis=0), axis=1)
    assert np.allclose(steps, 10, atol=1.5)


def test_synthesize_returns_uint8_stroke_near_boundary():
    mask = _square(220, 50)
    stroke = synthesize_stroke(mask, params=StrokeParams(p_no_stroke=0.0), rng=1)
    assert stroke.shape == mask.shape
    assert stroke.dtype == np.uint8
    assert (stroke > 0).sum() > 50

    boundary = boundary_from_mask(mask)
    dist = cv2.distanceTransform(255 - boundary, cv2.DIST_L2, 5)
    near = dist[stroke > 127] <= 20.0
    assert near.mean() > 0.8


def test_synthesize_is_deterministic_for_a_seed():
    mask = _square(160, 40)
    a = synthesize_stroke(mask, params=StrokeParams(p_no_stroke=0.0), rng=42)
    b = synthesize_stroke(mask, params=StrokeParams(p_no_stroke=0.0), rng=42)
    assert np.array_equal(a, b)


def test_no_stroke_option():
    mask = _square(120, 30)
    stroke = synthesize_stroke(mask, params=StrokeParams(p_no_stroke=1.0), rng=0)
    assert stroke.max() == 0


def test_boundary_band_peaks_at_one():
    mask = _square(160, 40)
    band = boundary_band(mask, sigma=1.0)
    assert band.dtype == np.float32
    assert band.shape == mask.shape
    assert 0.99 <= band.max() <= 1.0001
    assert band.min() >= 0.0
    # The band must sit on the boundary, not in the middle of the object.
    assert band[mask.shape[0] // 2, mask.shape[1] // 2] < 0.05


def test_fill_holes_keeps_outer_shape():
    mask = np.zeros((100, 100), np.uint8)
    cv2.rectangle(mask, (20, 20), (80, 80), 255, -1)
    cv2.circle(mask, (50, 50), 10, 0, -1)
    filled = fill_holes(mask)
    assert filled[50, 50] == 255
    assert filled[0, 0] == 0
    assert filled.sum() > mask.sum()


def test_encode_stroke_modes():
    mask = _square(120, 30)
    stroke = synthesize_stroke(mask, params=StrokeParams(p_no_stroke=0.0), rng=5)
    binary = encode_stroke(stroke, "binary")
    ramp = encode_stroke(stroke, "sdf", clip_px=16)
    assert binary.shape == stroke.shape
    assert 0.0 <= binary.min() and binary.max() <= 1.0
    assert ramp.max() > 0.99
    assert ramp.min() < 0.2
    assert stroke_distance(stroke).dtype == np.float32
