"""Image geometry: quarter turns, mirror and crop are exact; the fine rotation adds no artifacts.

Every test runs on a synthetic gel with known bands. A band is measured as its integrated
intensity above the background and its intensity-weighted centroid, inside a disc around the
position the band must have after the transform; that position comes from the coordinate
formulas written out here, independently of the implementation.

Tolerances of the fine rotation (bilinear interpolation, so first order and without overshoot):
a band integral within 0.2 % and a centroid within 0.05 px (measured at most 0.16 % and 0.02 px
for angles up to 30 degrees; the centroid error depends on where the band centre falls between
pixels), and a +angle -angle round trip within 3.5 % of the band peak, pixel by pixel. That last
bound is the interpolation error itself: linear interpolation errs by at most h^2/8 max|f''| per
axis and pass, and for a band of sigma 3 and 6 px the two axes add up to (1/9 + 1/36)/8 = 1.74 %
of its peak at any angle (the trace of the curvature does not turn), twice for the two passes
(measured 3.4 %).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import ValidationError

from gelpy.core.errors import InvalidValueError
from gelpy.core.geometry import (
    apply_geometry,
    check_crop,
    crop,
    fine_rotate,
    oriented_shape,
    quarter_turn,
    rotated_shape,
    source_mask,
)
from gelpy.schemas.image import ImageGeometry

H, W = 180, 140
BANDS = ((40.0, 50.0), (100.0, 52.0), (42.0, 118.0), (98.0, 126.0))  # (x, y) centres
BACKGROUND = 0.05
RADIUS = 14.0  # the disc a band is measured in (sigma 3 px across, 6 px along x)


def synthetic_gel(dtype=np.float64, *, h=H, w=W, bands=BANDS) -> np.ndarray:
    """A gel of four bands (sigma 6 px across the lane, 3 px along it) on a flat background."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    image = np.full((h, w), BACKGROUND)
    for cx, cy in bands:
        image += 0.8 * np.exp(-((x - cx) ** 2 / (2 * 6.0**2) + (y - cy) ** 2 / (2 * 3.0**2)))
    if np.issubdtype(np.dtype(dtype), np.integer):
        top = np.iinfo(dtype).max * 0.9
        return np.rint(image / image.max() * top).astype(dtype)
    return image.astype(dtype)


def measure(image: np.ndarray, at: tuple[float, float], background: float) -> tuple[float, float, float]:
    """(integral above background, centroid x, centroid y) in a disc of RADIUS around ``at``."""
    y, x = np.mgrid[0 : image.shape[0], 0 : image.shape[1]].astype(np.float64)
    disc = (x - at[0]) ** 2 + (y - at[1]) ** 2 <= RADIUS**2
    signal = (image.astype(np.float64) - background) * disc
    total = signal.sum()
    return float(total), float((signal * x).sum() / total), float((signal * y).sum() / total)


# --- the coordinate formulas (the oracle) ---------------------------------------------------
def turned_point(p, shape, turn):
    """Where pixel (x, y) of an image of ``shape`` lands after ``turn`` degrees counter-clockwise."""
    (x, y), (h, w) = p, shape
    for _ in range(turn // 90):
        x, y, h, w = y, w - 1 - x, w, h
    return x, y


def rotated_point(p, shape, angle, out_shape):
    """(x, y) after a fine rotation by ``angle`` degrees counter-clockwise about the centre."""
    (x, y), (h, w), (oh, ow) = p, shape, out_shape
    t = math.radians(angle)
    dx, dy = x - (w - 1) / 2, y - (h - 1) / 2
    return (ow - 1) / 2 + dx * math.cos(t) + dy * math.sin(t), (oh - 1) / 2 - dx * math.sin(t) + dy * math.cos(t)


# --- quarter turns, mirror, crop: exact -----------------------------------------------------
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int32, np.float32, np.float64])
def test_four_quarter_turns_are_the_identity_exactly(dtype):
    image = synthetic_gel(dtype)
    out = image
    for _ in range(4):
        out = quarter_turn(out, 90)
    assert out.dtype == image.dtype
    assert np.array_equal(out, image)
    assert np.array_equal(quarter_turn(quarter_turn(image, 90), 270), image)
    assert np.array_equal(quarter_turn(quarter_turn(image, 180), 180), image)
    assert np.array_equal(quarter_turn(image, -90), quarter_turn(image, 270))


def test_a_positive_turn_is_counter_clockwise_as_displayed():
    image = np.zeros((5, 7))
    image[2, 6] = 1.0  # right of the centre
    turned = quarter_turn(image, 90)
    assert turned.shape == (7, 5)
    assert np.argwhere(turned == 1.0).tolist() == [[0, 2]]  # now above the centre


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
@pytest.mark.parametrize("turn", [0, 90, 180, 270])
@pytest.mark.parametrize("flip", [False, True])
def test_turns_flips_and_crops_are_pure_pixel_copies(dtype, turn, flip):
    image = synthetic_gel(dtype)
    box = (7, 11, 120, 131) if turn in (90, 270) else (5, 9, 131, 170)
    out = apply_geometry(image, ImageGeometry(turn=turn, flip=flip, crop=box))
    expected = np.rot90(image, turn // 90)
    expected = expected[:, ::-1] if flip else expected
    expected = expected[box[1] : box[3], box[0] : box[2]]
    assert out.dtype == image.dtype
    assert out.shape == expected.shape
    assert np.array_equal(out, expected)
    assert out.flags.c_contiguous and not np.shares_memory(out, image)


@pytest.mark.parametrize("turn", [90, 180, 270])
@pytest.mark.parametrize("flip", [False, True])
def test_band_integral_and_position_are_exact_after_turn_flip_and_crop(turn, flip):
    image = synthetic_gel(np.uint16)
    background = float(np.median(image))
    box = (2, 3, 178, 138) if turn in (90, 270) else (2, 6, 136, 175)
    out = apply_geometry(image, ImageGeometry(turn=turn, flip=flip, crop=box))
    width = oriented_shape(H, W, ImageGeometry(turn=turn))[1]

    def moved(point):
        x, y = turned_point(point, (H, W), turn)
        x = width - 1 - x if flip else x
        return x - box[0], y - box[1]

    for band in BANDS:
        before = measure(image, band, background)
        after = measure(out, moved(band), background)
        assert after[0] == before[0]  # the same pixels: exact
        expected = moved(before[1:])
        assert after[1] == pytest.approx(expected[0], abs=1e-9)
        assert after[2] == pytest.approx(expected[1], abs=1e-9)


def test_a_crop_is_a_window_of_the_oriented_image():
    image = synthetic_gel()
    assert np.array_equal(crop(image, (10, 20, 30, 50)), image[20:50, 10:30])
    assert np.array_equal(crop(image, (0, 0, W, H)), image)


@pytest.mark.parametrize(
    ("box", "words"),
    [
        ((0, 0, W + 1, H), f"x1 = {W + 1} is beyond the width {W}"),
        ((0, 0, W, H + 5), f"y1 = {H + 5} is beyond the height {H}"),
        ((10, 0, 10, H), "x1 must be larger than x0"),
    ],
)
def test_an_out_of_range_crop_box_is_refused_with_a_clear_message(box, words):
    with pytest.raises(InvalidValueError) as caught:
        check_crop(box, (H, W))
    message = str(caught.value)
    assert words in message
    assert f"{W} x {H} px" in message
    if box[2] > box[0]:  # a box the contract accepts, but too large for this image
        with pytest.raises(InvalidValueError, match="beyond"):
            apply_geometry(synthetic_gel(), ImageGeometry(crop=box))


@pytest.mark.parametrize("box", [(-1, 0, 10, 10), (0, 0, 0, 10), (5, 5, 10, 4), (0, 0, 10)])
def test_the_geometry_contract_refuses_a_malformed_box(box):
    with pytest.raises(ValidationError):
        ImageGeometry(crop=box)


def test_the_geometry_contract_normalises_turns_and_bounds_the_angle():
    assert ImageGeometry(turn=-90).turn == 270
    assert ImageGeometry(turn=450).turn == 90
    assert ImageGeometry().is_identity()
    assert not ImageGeometry(angle=0.5).is_identity()
    for bad in ({"turn": 45}, {"angle": 45.5}, {"angle": -60}, {"angle": float("nan")}):
        with pytest.raises(ValidationError):
            ImageGeometry(**bad)


# --- the fine rotation: bilinear, no overshoot, dtype kept --------------------------------
def test_a_zero_angle_is_an_exact_copy():
    image = synthetic_gel(np.uint16)
    out = fine_rotate(image, 0.0)
    assert out.dtype == np.uint16 and np.array_equal(out, image) and not np.shares_memory(out, image)


def test_a_positive_angle_is_counter_clockwise_as_displayed():
    image = np.zeros((101, 101))
    image[50, 80] = 1.0
    out = fine_rotate(image, 10.0)
    y, x = np.unravel_index(np.argmax(out), out.shape)
    oy, ox = (out.shape[0] - 101) // 2, (out.shape[1] - 101) // 2
    assert y - oy < 50  # moved up
    assert abs((x - ox) - (50 + 30 * math.cos(math.radians(10)))) <= 1


@pytest.mark.parametrize("angle", [0.7, 2.5, -4.0, 11.0, -45.0])
def test_the_canvas_holds_the_whole_source_with_the_least_symmetric_padding(angle):
    oh, ow = rotated_shape(H, W, angle)
    assert (oh - H) % 2 == 0 and (ow - W) % 2 == 0
    corners = [rotated_point(p, (H, W), angle, (oh, ow)) for p in ((0, 0), (W - 1, 0), (0, H - 1), (W - 1, H - 1))]
    xs, ys = [c[0] for c in corners], [c[1] for c in corners]
    assert min(xs) >= -1e-9 and max(xs) <= ow - 1 + 1e-9
    assert min(ys) >= -1e-9 and max(ys) <= oh - 1 + 1e-9
    assert min(xs) < 1 + 1e-9 and min(ys) < 1 + 1e-9  # one pixel less on each side would cut a corner
    assert fine_rotate(synthetic_gel(), angle).shape == (oh, ow)
    assert oriented_shape(H, W, ImageGeometry(turn=90, angle=angle)) == rotated_shape(W, H, angle)


@pytest.mark.parametrize("angle", [2.5, -4.0, 11.0])
def test_a_fine_rotation_keeps_band_integral_and_position(angle):
    image = synthetic_gel()
    out = fine_rotate(image, angle)
    for band in BANDS:
        moved = rotated_point(band, (H, W), angle, out.shape)
        before = measure(image, band, BACKGROUND)
        after = measure(out, moved, BACKGROUND)
        assert after[0] == pytest.approx(before[0], rel=2e-3)
        assert math.hypot(after[1] - moved[0], after[2] - moved[1]) < 0.05


@pytest.mark.parametrize("angle", [1.5, -3.0, 8.0])
def test_plus_then_minus_the_angle_comes_back_within_the_bound(angle):
    image = synthetic_gel()
    back = fine_rotate(fine_rotate(image, angle), -angle)
    oy, ox = (back.shape[0] - H) // 2, (back.shape[1] - W) // 2
    window = back[oy : oy + H, ox : ox + W]
    peak = image.max() - BACKGROUND
    assert np.abs(window - image).max() < 0.035 * peak
    for band in BANDS:
        assert measure(window, band, BACKGROUND)[0] == pytest.approx(measure(image, band, BACKGROUND)[0], rel=2e-3)


@pytest.mark.parametrize("dtype", [np.uint16, np.float32, np.float64])
def test_a_fine_rotation_never_overshoots_and_fills_the_corners_with_the_minimum(dtype):
    rng = np.random.default_rng(7)
    noise = rng.uniform(1000, 60000, size=(64, 48))
    checker = np.indices((64, 48)).sum(axis=0) % 2 * 59000.0 + 1000.0  # the worst case for ringing
    for source in (noise, checker):
        image = source.astype(dtype)
        out = fine_rotate(image, 7.3)
        assert out.dtype == image.dtype
        assert out.min() >= image.min() and out.max() <= image.max()
        inside = source_mask(64, 48, 7.3)
        assert inside.shape == out.shape and not inside.all()
        assert np.all(out[~inside] == image.min())


def test_integers_are_rounded_to_the_nearest_value_ties_to_even():
    image = synthetic_gel(np.uint16)
    out = fine_rotate(image, 3.3)
    exact = fine_rotate(image.astype(np.float64), 3.3)
    assert out.dtype == np.uint16
    assert np.array_equal(out, np.rint(exact).astype(np.uint16))
    assert np.array_equal(fine_rotate(image.astype(np.float32), 3.3), exact.astype(np.float32))


def test_the_source_mask_marks_what_comes_from_the_source():
    image = synthetic_gel() + 1.0  # every source value above the fill below
    out = fine_rotate(image, -6.0, fill=0.0)
    assert np.array_equal(source_mask(H, W, -6.0), out > 0.0)
    assert source_mask(H, W, 0.0).all()


def test_the_full_geometry_applies_turn_flip_angle_then_crop():
    image = synthetic_gel(np.uint16)
    geometry = ImageGeometry(turn=90, flip=True, angle=2.0, crop=(4, 6, 150, 120))
    stepwise = fine_rotate(np.ascontiguousarray(np.rot90(image)[:, ::-1]), 2.0)[6:120, 4:150]
    out = apply_geometry(image, geometry)
    assert out.dtype == np.uint16 and np.array_equal(out, stepwise)
    assert oriented_shape(H, W, geometry) == fine_rotate(np.rot90(image), 2.0).shape
