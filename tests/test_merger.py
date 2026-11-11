import numpy as np

from backend.inference.merger import CropPlacement, PredictionMerger, apply_morphological_filtering


def _block(size=50, value=1.0):
    prob = np.zeros((size, size), np.float32)
    prob[10 : size - 10, 10 : size - 10] = value
    return prob


def test_single_crop_is_placed_correctly():
    merger = PredictionMerger(100, 100)
    assert merger.add(CropPlacement(0, 0, 50), _block())
    result = merger.finalize(threshold=0.5, apply_morphology=False)

    assert result.mask[20, 20] == 255
    assert result.mask[80, 80] == 0
    assert result.edge[20, 20] == 1.0
    assert result.covered == 2500


def test_overlap_is_averaged():
    merger = PredictionMerger(100, 100)
    merger.add(CropPlacement(0, 0, 50), _block(value=1.0))
    merger.add(CropPlacement(0, 0, 50), _block(value=0.0))
    result = merger.finalize(threshold=0.9, apply_morphology=False)
    assert abs(float(result.edge[20, 20]) - 0.5) < 1e-5


def test_out_of_bounds_crop_is_clamped():
    merger = PredictionMerger(100, 100)
    prob = np.ones((40, 40), np.float32)
    assert merger.add(CropPlacement(-20, -20, 40), prob)
    result = merger.finalize(threshold=0.5, apply_morphology=False)
    assert result.edge[0, 0] == 1.0
    assert result.edge[99, 99] == 0.0
    assert result.covered == 400


def test_fully_outside_crop_is_ignored():
    merger = PredictionMerger(50, 50)
    assert merger.add(CropPlacement(200, 200, 32), np.ones((32, 32), np.float32)) is False
    assert merger.finalize().covered == 0


def test_validity_mask_excludes_padding():
    merger = PredictionMerger(100, 100)
    valid = np.ones((50, 50), np.float32)
    valid[:25, :] = 0.0
    merger.add(CropPlacement(0, 0, 50), _block(), valid)
    result = merger.finalize(threshold=0.5, apply_morphology=False)
    assert result.edge[5, 20] == 0.0
    assert result.edge[35, 20] == 1.0


def test_brush_boost_raises_repeated_pixels():
    prob = _block(value=0.6)

    single = PredictionMerger(100, 100)
    single.add(CropPlacement(0, 0, 50), prob)
    low = single.finalize(threshold=0.65, sensitivity=1.0, apply_morphology=False)

    double = PredictionMerger(100, 100)
    double.add(CropPlacement(0, 0, 50), prob)
    double.add(CropPlacement(0, 0, 50), prob)
    high = double.finalize(threshold=0.65, sensitivity=2.0, apply_morphology=False)

    assert low.mask[20, 20] == 0
    assert high.mask[20, 20] == 255


def test_gaussian_weighting_keeps_uniform_field():
    merger = PredictionMerger(100, 100, weighting="gaussian")
    merger.add(CropPlacement(0, 0, 50), np.ones((50, 50), np.float32))
    result = merger.finalize(threshold=0.5, apply_morphology=False)
    assert abs(float(result.edge[25, 25]) - 1.0) < 1e-5


def test_empty_merger_returns_empty():
    merger = PredictionMerger(32, 32)
    result = merger.finalize()
    assert result.mask.max() == 0
    assert result.covered == 0


def test_morphology_closes_small_gaps():
    mask = np.zeros((40, 40), np.uint8)
    mask[10:30, 10:20] = 255
    mask[10:30, 21:30] = 255
    mask[10:30, 20] = 0
    out = apply_morphological_filtering(mask, kernel_size=3, iterations=2)
    assert out[20, 20] == 255


def test_rejects_bad_weighting():
    try:
        PredictionMerger(10, 10, weighting="magic")
    except ValueError:
        return
    raise AssertionError("expected ValueError")
