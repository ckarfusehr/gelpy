import numpy as np
import pytest

from gelpy.core.line_profiles import (
    apply_slice,
    extract_line_profile,
    normalize_profile,
    strips_overlap,
)


def test_extract_line_profile_is_column_mean():
    image = np.tile(np.arange(10, dtype=float).reshape(-1, 1), (1, 20))
    # constant across columns, so any strip mean equals the row value
    profile = extract_line_profile(image, x_center=10, width=4)
    assert np.allclose(profile, np.arange(10))
    assert profile.shape == (10,)


def test_extract_clamps_to_image_bounds():
    image = np.ones((5, 6))
    profile = extract_line_profile(image, x_center=0, width=8)
    assert profile.shape == (5,)


def test_normalize_area_sums_to_unit_trapezoid():
    profile = np.array([1.0, 2.0, 3.0, 4.0])
    normed = normalize_profile(profile, "area")
    assert np.isclose(np.trapezoid(normed), 1.0)


def test_normalize_min_max_range():
    profile = np.array([2.0, 4.0, 6.0])
    normed = normalize_profile(profile, "min_max")
    assert normed.min() == 0 and normed.max() == 1


def test_normalize_safe_is_noop():
    profile = np.array([1.0, 2.0])
    assert np.array_equal(normalize_profile(profile, "safe"), profile)


def test_normalize_unknown_raises():
    with pytest.raises(ValueError):
        normalize_profile(np.array([1.0]), "bogus")


def test_apply_slice():
    profile = np.arange(10)
    assert np.array_equal(apply_slice(profile, 2, 5), np.array([2, 3, 4]))
    assert np.array_equal(apply_slice(profile, 8, None), np.array([8, 9]))


def test_strips_overlap_detects_adjacent():
    # centers 0 and 5 with width 8 overlap; 5 and 20 do not
    assert strips_overlap([0.0, 5.0, 20.0], width=8) == [(0, 1)]
