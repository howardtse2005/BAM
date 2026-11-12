"""The single square the magic brush crops around a stroke."""

import numpy as np
import pytest

from backend.inference.crop_window import snap_up, stroke_bounds, stroke_window


def _stroke(height: int, width: int, y0: int, x0: int, y1: int, x1: int) -> np.ndarray:
    stroke = np.zeros((height, width), np.uint8)
    stroke[y0:y1, x0:x1] = 255
    return stroke


def test_empty_stroke_has_no_bounds():
    assert stroke_bounds(np.zeros((40, 40), np.uint8)) is None


def test_empty_stroke_raises():
    with pytest.raises(ValueError):
        stroke_window(np.zeros((40, 40), np.uint8))


def test_boolean_strokes_are_accepted():
    stroke = np.zeros((40, 40), bool)
    stroke[10:20, 5:15] = True
    assert stroke_bounds(stroke) == (5, 10, 14, 19)


@pytest.mark.parametrize(
    "y0,x0,y1,x1",
    [
        (100, 120, 160, 420),  # wide
        (40, 200, 240, 260),  # tall
        (90, 90, 130, 130),  # square
        (0, 0, 30, 30),  # touching the frame corner
    ],
)
def test_window_contains_the_whole_stroke(y0, x0, y1, x1):
    stroke = _stroke(300, 500, y0, x0, y1, x1)
    window = stroke_window(stroke, pad=16, min_side=64, max_side=256)

    ys, xs = np.nonzero(stroke)
    assert window.x0 <= xs.min()
    assert window.y0 <= ys.min()
    assert window.x0 + window.size > xs.max()
    assert window.y0 + window.size > ys.max()
    assert window.size % 32 == 0


def test_window_is_the_minimum_square():
    """Nothing beyond the stroke, its band, and the snap to the model grid."""
    stroke = _stroke(300, 500, 100, 120, 160, 420)  # 60 x 300
    window = stroke_window(stroke, pad=16, min_side=64, max_side=256)

    extent = 300 + 2 * 16
    assert window.size == snap_up(extent)
    assert window.size - extent < 32


def test_window_is_centred_on_the_stroke_bounding_box():
    stroke = _stroke(300, 500, 100, 120, 160, 420)
    window = stroke_window(stroke, pad=16, min_side=64, max_side=256)

    centre_x = window.x0 + window.size / 2.0
    centre_y = window.y0 + window.size / 2.0
    assert abs(centre_x - (120 + 420) / 2.0) <= 1.0
    assert abs(centre_y - (100 + 160) / 2.0) <= 1.0


def test_window_is_floored_so_a_dot_still_has_context():
    stroke = np.zeros((300, 300), np.uint8)
    stroke[150, 150] = 255
    window = stroke_window(stroke, pad=2, min_side=64, max_side=256)
    assert window.size == 64
    assert window.model_size == 64
    assert window.scale == 1.0


def test_window_never_upscales():
    """A short stroke runs at its own scale; the prompt keeps its width."""
    stroke = _stroke(300, 300, 140, 140, 160, 160)  # 20 x 20
    window = stroke_window(stroke, pad=4, min_side=64, max_side=256)
    assert window.scale == 1.0
    assert not window.downscaled
    assert window.size >= 20 + 2 * 4


def test_window_downscales_above_the_trained_crop():
    stroke = _stroke(600, 900, 10, 10, 590, 890)  # spans almost everything
    window = stroke_window(stroke, pad=16, min_side=64, max_side=256)
    assert window.size > 256
    assert window.model_size == 256
    assert 0.0 < window.scale < 1.0


def test_window_size_is_capped_by_the_model_grid():
    stroke = _stroke(600, 900, 10, 10, 590, 890)
    window = stroke_window(stroke, pad=16, min_side=64, max_side=256)
    assert window.model_size % 32 == 0
    assert window.model_size <= 256
