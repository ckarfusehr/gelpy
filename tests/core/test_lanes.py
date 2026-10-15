import numpy as np
import pytest

from gelpy.core.errors import LaneSelectionError
from gelpy.core.lanes import (
    column_projection,
    compute_lane_positions,
    detect_lane_centers,
    guess_profile_width,
    resolve_lane_selector,
)
from gelpy.schemas.lanes import LaneDefinition, LaneLayout


def _synthetic_gel(centers, width=1000, height=200, sd=8.0, amp=1.0, bg=0.05):
    """A gel image with a bright vertical band at each given x-center."""
    img = np.full((height, width), bg, dtype=np.float64)
    x = np.arange(width)
    for c in centers:
        img += (amp * np.exp(-((x - c) ** 2) / (2 * sd**2)))[None, :]
    return img


def _synthetic_gel_variable_amplitude(centers, amps, width=1000, height=200, sd=8.0, bg=0.05):
    """Like ``_synthetic_gel`` but each band gets its own amplitude (faint lanes, dust)."""
    img = np.full((height, width), bg, dtype=np.float64)
    x = np.arange(width)
    for c, a in zip(centers, amps):
        img += (a * np.exp(-((x - c) ** 2) / (2 * sd**2)))[None, :]
    return img


def _dusted_gel(centers, width=1000, height=200, dust_x=200.0, dust_sd=2.0, dust_amp=0.3):
    """``_synthetic_gel`` (lanes at sd=8, ~19 px wide) plus one narrow dust spike."""
    img = _synthetic_gel(centers, width=width, height=height)
    x = np.arange(width)
    img += (dust_amp * np.exp(-((x - dust_x) ** 2) / (2 * dust_sd**2)))[None, :]
    return img


def make_layout(labels):
    lanes = [
        LaneDefinition(index=i, label=label, x_center_px=float(i * 10))
        for i, label in enumerate(labels)
    ]
    return LaneLayout(lanes=lanes, profile_width_px=8)


def test_compute_lane_positions():
    pos = compute_lane_positions(0, 100, 3)
    assert pos == [0.0, 50.0, 100.0]


def test_guess_profile_width():
    assert guess_profile_width(20, 4500) == int(4500 / (20 * 2.5))


def test_selector_all():
    layout = make_layout(["a", "b", "c"])
    assert resolve_lane_selector("all", layout) == [0, 1, 2]


def test_selector_index():
    layout = make_layout(["a", "b", "c"])
    assert resolve_lane_selector(2, layout) == [2]


def test_selector_out_of_range_raises_with_available():
    layout = make_layout(["a", "b"])
    with pytest.raises(LaneSelectionError) as exc:
        resolve_lane_selector(5, layout)
    assert exc.value.available == ["a", "b"]


def test_selector_literal_label():
    layout = make_layout(["scaffold", "600 NaCl", "scaffold"])
    assert resolve_lane_selector("scaffold", layout) == [0, 2]


def test_selector_regex():
    layout = make_layout(["1 6 MgCl", "2 10 MgCl", "control"])
    assert resolve_lane_selector("re:.*MgCl.*", layout) == [0, 1]


def test_selector_forced_literal_numeric_label():
    layout = make_layout(["12", "control"])
    # 'label:12' forces a literal label match rather than an index
    assert resolve_lane_selector("label:12", layout) == [0]


def test_selector_mixed_list_dedup_sorted():
    layout = make_layout(["scaffold", "600 NaCl", "scaffold"])
    assert resolve_lane_selector(["scaffold", 1, "re:600.*"], layout) == [0, 1, 2]


def test_selector_no_match_raises():
    layout = make_layout(["a", "b"])
    with pytest.raises(LaneSelectionError):
        resolve_lane_selector("nonexistent", layout)


def test_selector_bool_rejected():
    layout = make_layout(["a", "b"])
    with pytest.raises(LaneSelectionError):
        resolve_lane_selector(True, layout)


# --- lane detection ----------------------------------------------------------
def test_column_projection_collapses_y_axis():
    img = _synthetic_gel([500], width=1000, height=200)
    proj = column_projection(img)
    assert proj.shape == (1000,)
    # The projection peaks at the band's column.
    assert abs(int(np.argmax(proj)) - 500) <= 1


def test_column_projection_row_band():
    img = np.zeros((100, 50))
    img[:20, :] = 1.0  # bright only in the top rows
    assert column_projection(img, y_start=0, y_end=20).mean() == 1.0
    assert column_projection(img, y_start=20, y_end=100).mean() == 0.0


def test_detect_recovers_even_grid():
    true = list(np.linspace(80, 920, 10))
    d = detect_lane_centers(_synthetic_gel(true), 10)
    assert d.n_detected == 10
    assert d.confidence == "ok"
    assert max(abs(a - b) for a, b in zip(d.centers, true)) < 1.0
    assert abs(d.pitch - (920 - 80) / 9) < 1.0
    assert d.profile_width > 0


def test_detect_flags_low_confidence_for_partial_width_crop():
    # The gel fills only the left ~third of a wide crop. This used to be handled by a
    # pure peak-gap pitch estimate with no width/N prior, so it "just worked". The
    # current detector instead uses a soft prior toward width/n_lanes (deliberately, to
    # fix a *worse* real-data failure: a faint-lane gel's pitch collapsing onto its
    # brighter half — see test_detect_recovers_full_width_grid_when_right_lanes_are_faint
    # below) and does not claim to auto-place a gel that fills only a fraction of the
    # crop. That is now explicitly out of scope for auto-detect: it must come back
    # "low" confidence (with a reason) so the caller falls back to manual endpoints,
    # rather than silently returning a wrong grid at "ok" confidence.
    true = list(np.linspace(120, 480, 12))
    d = detect_lane_centers(_synthetic_gel(true, width=1400), 12)
    assert d.confidence == "low"
    assert d.confidence_reason


def test_detect_places_skipped_interior_well():
    # 10 comb positions, but well index 4 is empty (no band): detection must still
    # return 10 evenly-spaced centers with the empty slot correctly placed.
    #
    # Tolerance is pitch-relative and states measured behaviour: 2.0 px on a 93.3 px pitch
    # = 0.021x pitch. Sub-pixel grid phase is neither achieved nor needed -- the profile
    # strip is ~0.7x pitch wide, so a 2% phase error moves the strip by 3% of its own width.
    # See docs/guide/known-limitations.md.
    true = list(np.linspace(80, 920, 10))
    pitch = (true[-1] - true[0]) / 9
    loaded = [c for i, c in enumerate(true) if i != 4]
    d = detect_lane_centers(_synthetic_gel(loaded), 10)
    assert len(d.centers) == 10
    assert d.n_detected == 9
    assert max(abs(a - b) for a, b in zip(d.centers, true)) < 0.05 * pitch


def test_detect_accepts_a_float_smoothing_window():
    # `smoothing_window` is a float in the public signature, the request schema and the CLI
    # (it is a pixel width a caller may scale off a measured pitch), but scipy's
    # uniform_filter1d(size=...) demands an int and raises TypeError on any float, including
    # an integral one. Every user-supplied --smoothing-window therefore used to crash.
    true = list(np.linspace(80, 920, 10))
    d = detect_lane_centers(_synthetic_gel(true), 10, smoothing_window=40.0)
    assert len(d.centers) == 10


def test_detect_places_empty_edge_well_on_the_correct_side():
    # An empty well at the very edge (index 0 unloaded) is ambiguous by peak-fit cost
    # alone: shifting the whole grid right by one pitch (leaving the empty slot
    # off-image on the right) scores identically to correctly placing it on the left,
    # since the fit only measures peak-to-grid-node distance for the *detected* peaks.
    # A lane can't physically sit outside the image, so ties must resolve toward the
    # in-bounds candidate.
    true = list(np.linspace(80, 920, 13))
    loaded = true[1:]
    d = detect_lane_centers(_synthetic_gel(loaded), 13)
    assert d.n_detected == 12
    assert all(0 <= c < 1000 for c in d.centers)
    assert d.confidence == "ok"
    assert max(abs(a - b) for a, b in zip(d.centers, true)) < 2.0


def test_detect_recovers_full_width_grid_when_right_lanes_are_faint():
    # Mirrors the real-data failure that motivated the matched-filter rewrite:
    # bright left-cluster lanes, faint-but-present right-cluster lanes, and
    # a couple of dust bumps scattered among the bright ones. The old median-peak-gap
    # pitch estimator collapsed the whole grid onto the bright left cluster's pitch; the
    # matched-filter + width/N prior must instead recover the true full-width pitch.
    n = 20
    width = 2772
    true = list(np.linspace(113, 2659, n))
    amps = [1.0] * 12 + [0.15] * 8  # right third much fainter, but still real
    img = _synthetic_gel_variable_amplitude(true, amps, width=width, sd=6.0)
    x = np.arange(width)
    for dust_x in (300.0, 700.0):
        img += (0.4 * np.exp(-((x - dust_x) ** 2) / (2 * 2.0**2)))[None, :]

    d = detect_lane_centers(img, n)
    true_pitch = (2659 - 113) / (n - 1)
    # Pitch recovery is the point of this test and is accurate to well under a pixel.
    assert abs(d.pitch - true_pitch) < 2.0
    # Grid *phase* is the weak axis: measured 9.17 px on a 134 px pitch = 0.068x pitch, the
    # largest phase error in the suite. It is the hardest case here (a third of the lanes at
    # 15% amplitude, plus two dust bumps among the bright ones) and is recorded as a known
    # limitation rather than silently loosened -- see docs/guide/known-limitations.md.
    assert max(abs(a - b) for a, b in zip(d.centers, true)) < 0.075 * true_pitch


def test_detect_recovers_grid_with_several_empty_interior_wells():
    # 3 scattered empty interior wells (not one contiguous block) — the specific
    # robustness case the pitch prior's strength was tuned against: a looser prior
    # recovers a clean single-cluster gel but half-pitch-flips or pushes a lane off-crop
    # once several interior wells go dark.
    n = 20
    true = list(np.linspace(113, 2659, n))
    loaded = [c for i, c in enumerate(true) if i not in (5, 10, 14)]
    d = detect_lane_centers(_synthetic_gel(loaded, width=2772, sd=6.0), n)
    assert d.n_detected == n - 3
    assert max(abs(a - b) for a, b in zip(d.centers, true)) < 2.0


def test_detect_flags_low_confidence_when_too_few_lanes_found():
    # Only 3 of 12 comb positions carry any signal: the fit is unreliable and must
    # be flagged so the caller checks the overlay / supplies manual endpoints.
    true = list(np.linspace(120, 480, 12))
    sparse = [true[0], true[5], true[11]]
    d = detect_lane_centers(_synthetic_gel(sparse, width=1400), 12)
    assert d.confidence == "low"


def test_detect_single_lane():
    d = detect_lane_centers(_synthetic_gel([500], width=1000), 1)
    assert len(d.centers) == 1
    assert abs(d.centers[0] - 500) <= 1.0


def test_default_smoothing_already_suppresses_a_narrow_dust_spike():
    # The pitch-scaled smoothing window (0.7 x pitch = 65 px here) is a low-pass filter, so a
    # narrow dust spike never reaches find_peaks in the first place: no flag needed. This is
    # why `min_width` is off by default and why raising it is NOT the first move against dust.
    true = list(np.linspace(80, 920, 10))
    img = _dusted_gel(true)
    d = detect_lane_centers(img, 10)
    assert d.n_detected == 10  # the sd=2 spike is smoothed away, not counted as a lane
    assert d.confidence == "ok"


def test_detect_min_width_rejects_a_spurious_peak_that_survives_smoothing():
    # `min_width` filters on peak width *after* smoothing, so it only has anything to do when
    # a spurious peak survives to find_peaks -- i.e. when the smoothing window has been
    # narrowed deliberately (or, on real data, when the artifact is itself lane-width).
    # A real-data investigation found peak width cannot be a safe universal default: a genuine
    # faint band can be exactly as narrow as a dust spike, so there is no scaled default right
    # for every image. Hence: explicit, off by default, and read off the overlay per image.
    true = list(np.linspace(80, 920, 10))
    img = _dusted_gel(true)

    # With smoothing narrowed to 5 px the sd=2 spike survives and is miscounted as a lane...
    unfiltered = detect_lane_centers(img, 10, smoothing_window=5.0)
    assert unfiltered.n_detected == 11

    # ...and a min_width between the spurious (~5 px) and genuine (~19 px) widths drops it.
    filtered = detect_lane_centers(img, 10, smoothing_window=5.0, min_width=10.0)
    assert filtered.n_detected == 10
    assert filtered.confidence == "ok"
    assert max(abs(a - b) for a, b in zip(filtered.centers, true)) < 1.0


def test_an_invalid_regex_selector_is_a_lane_selection_error():
    from gelpy.core.lanes import resolve_lane_selector

    layout = LaneLayout(
        lanes=[LaneDefinition(index=0, label="a", x_center_px=10.0)], profile_width_px=5
    )
    with pytest.raises(LaneSelectionError) as info:
        resolve_lane_selector("re:(", layout)
    assert info.value.available == ["a"]
