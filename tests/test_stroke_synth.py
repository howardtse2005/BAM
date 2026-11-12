import cv2
import numpy as np
import pytest

from training.stroke_synth import (
    BrushParams,
    boundary_band,
    boundary_from_mask,
    brush_distance,
    clean_edge,
    edge_band,
    encode_brush,
    ensure_binary,
    fill_holes,
    mask_to_paths,
    polyline_length,
    resample_polyline,
    synthesize_brush,
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


# --------------------------------------------------------------------------- #
# gt_edge: the clean line the model predicts
# --------------------------------------------------------------------------- #


def test_clean_edge_sits_on_the_boundary():
    mask = _square(200, 40)
    edge = clean_edge(mask)
    assert edge.shape == mask.shape
    assert edge.dtype == np.uint8
    assert (edge > 0).sum() > 0

    boundary = boundary_from_mask(mask)
    dist = cv2.distanceTransform(255 - boundary, cv2.DIST_L2, 5)
    # The line straddles the boundary by its own thickness and nothing more:
    # no offset, no wobble, no coverage gaps.
    assert dist[edge > 127].max() <= 2.5
    assert float(np.median(dist[edge > 127])) <= 1.5


def test_clean_edge_covers_the_whole_perimeter():
    mask = _square(220, 50)
    edge = clean_edge(mask)

    # A full loop: the edge must reach all four sides of the square.
    ys, xs = np.nonzero(edge)
    assert ys.min() <= 51 and ys.max() >= 169
    assert xs.min() <= 51 and xs.max() >= 169

    # And it must be a thin line, not a filled ring.
    assert (edge > 127).sum() < mask.sum() / 255 / 4


def test_clean_edge_is_deterministic_and_ignores_rng():
    mask = _square(160, 40)
    a = clean_edge(mask)
    b = clean_edge(mask)
    assert np.array_equal(a, b)


def test_clean_edge_includes_holes():
    mask = _square(200, 40)
    cv2.circle(mask, (100, 100), 20, 0, -1)
    edge = clean_edge(mask)
    inner = edge[80:120, 80:120]
    assert (inner > 0).sum() > 0


def test_clean_edge_and_brush_describe_the_same_boundary():
    """The brush must span the whole edge; that is what keeps the task aligned.

    If the brush stopped part way round, the 4th input channel and the target
    would be describing different parts of the object and the model would have to
    invent the rest.
    """
    mask = _square(220, 50)
    cv2.circle(mask, (110, 110), 26, 0, -1)
    edge = clean_edge(mask)
    brush = synthesize_brush(mask, params=BrushParams(), rng=3)

    to_brush = cv2.distanceTransform(
        255 - ((brush > 127).astype(np.uint8) * 255), cv2.DIST_L2, 5
    )
    assert float((to_brush[edge > 127] <= 8.0).mean()) > 0.95


def test_brush_strokes_every_contour():
    """Outer boundary and hole boundary both get a stroke, like clean_edge."""
    mask = _square(220, 50)
    cv2.circle(mask, (110, 110), 30, 0, -1)
    brush = synthesize_brush(mask, params=BrushParams(), rng=1)

    # Pixels on the inner ring: just outside the hole.
    inner = brush[76:86, 100:120]
    outer = brush[46:56, 100:120]
    assert (inner > 127).sum() > 0
    assert (outer > 127).sum() > 0


def test_brush_is_rougher_than_the_edge():
    """The brush must stay a rough stroke, not become a copy of the target."""
    mask = _square(220, 50)
    edge = clean_edge(mask)
    brush = synthesize_brush(mask, params=BrushParams(), rng=3)

    edge_px = edge > 127
    brush_px = brush > 127
    # Thicker: a brush, not a hairline.
    assert brush_px.sum() > 1.5 * edge_px.sum()
    # And only a minority of its pixels land exactly on the edge.
    assert float((edge_px & brush_px).sum()) / float(brush_px.sum()) < 0.6


@pytest.mark.parametrize(
    "kwargs",
    [
        {"p_partial": 1.0},
        {"p_no_brush": 1.0},
        {"p_distractor": 1.0},
        {"p_interior": 1.0},
        {"p_dry_brush": 1.0},
    ],
)
def test_deliberate_mistakes_stay_available(kwargs):
    """The knobs that break alignment are off by default but still work."""
    mask = _square(220, 50)
    brush = synthesize_brush(mask, params=BrushParams(**kwargs), rng=2)
    assert brush.shape == mask.shape
    assert brush.dtype == np.uint8


# --------------------------------------------------------------------------- #
# gt_brush: the rough input
# --------------------------------------------------------------------------- #


def test_synthesize_returns_uint8_brush_near_boundary():
    mask = _square(220, 50)
    brush = synthesize_brush(mask, params=BrushParams(p_no_brush=0.0), rng=1)
    assert brush.shape == mask.shape
    assert brush.dtype == np.uint8
    assert (brush > 0).sum() > 50

    boundary = boundary_from_mask(mask)
    dist = cv2.distanceTransform(255 - boundary, cv2.DIST_L2, 5)
    near = dist[brush > 127] <= 20.0
    assert near.mean() > 0.8


def test_synthesize_is_deterministic_for_a_seed():
    mask = _square(160, 40)
    a = synthesize_brush(mask, params=BrushParams(p_no_brush=0.0), rng=42)
    b = synthesize_brush(mask, params=BrushParams(p_no_brush=0.0), rng=42)
    assert np.array_equal(a, b)


def test_no_brush_option():
    mask = _square(120, 30)
    brush = synthesize_brush(mask, params=BrushParams(p_no_brush=1.0), rng=0)
    assert brush.max() == 0


# --------------------------------------------------------------------------- #
# Target and encoding
# --------------------------------------------------------------------------- #


def test_boundary_band_peaks_at_one():
    mask = _square(160, 40)
    band = boundary_band(mask, sigma=1.0)
    assert band.dtype == np.float32
    assert band.shape == mask.shape
    assert 0.99 <= band.max() <= 1.0001
    assert band.min() >= 0.0
    # The band must sit on the boundary, not in the middle of the object.
    assert band[mask.shape[0] // 2, mask.shape[1] // 2] < 0.05


def test_edge_band_matches_boundary_band():
    mask = _square(160, 40)
    assert np.allclose(boundary_band(mask, 1.0), edge_band(clean_edge(mask), 1.0))


def test_edge_band_handles_an_empty_edge():
    band = edge_band(np.zeros((32, 32), np.uint8), sigma=1.0)
    assert band.max() == 0.0


def test_fill_holes_keeps_outer_shape():
    mask = np.zeros((100, 100), np.uint8)
    cv2.rectangle(mask, (20, 20), (80, 80), 255, -1)
    cv2.circle(mask, (50, 50), 10, 0, -1)
    filled = fill_holes(mask)
    assert filled[50, 50] == 255
    assert filled[0, 0] == 0
    assert filled.sum() > mask.sum()


def test_encode_brush_modes():
    mask = _square(120, 30)
    brush = synthesize_brush(mask, params=BrushParams(p_no_brush=0.0), rng=5)
    binary = encode_brush(brush, "binary")
    ramp = encode_brush(brush, "sdf", clip_px=16)
    assert binary.shape == brush.shape
    assert 0.0 <= binary.min() and binary.max() <= 1.0
    assert ramp.max() > 0.99
    assert ramp.min() < 0.2
    assert brush_distance(brush).dtype == np.float32
