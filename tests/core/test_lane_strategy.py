"""Scaffold guardrails for the pluggable projection-strategy + comb-SNR fallback.

These test the mechanism the adaptive-band designs (D1-D4) build on: the registry, the
comb-SNR objective, and the adopt-only-if-better fallback in ``detect_lane_centers``. They do
NOT test any particular adaptive strategy (each design ships its own). See docs/decisions/adaptive-band/.
"""

from __future__ import annotations

import numpy as np
import pytest

from gelpy.core.lanes import (
    AdaptiveProjection,
    _comb_snr,
    column_projection,
    detect_lane_centers,
    projection_strategies,
)


def _band_gel(*, band_lo=40, band_hi=70, amp=0.6, noise=0.12, seed=0):
    """A gel whose bands live only in rows [band_lo:band_hi] — so the band region projects a
    crisper comb than the full height, giving an adaptive strategy real headroom to win."""
    rng = np.random.default_rng(seed)
    H, W, N = 200, 800, 8
    img = np.full((H, W), 0.05)
    centers = np.linspace(60, 740, N)
    x = np.arange(W)
    for c in centers:
        img[band_lo:band_hi] += (amp * np.exp(-((x - c) ** 2) / (2 * 20**2)))[None, :]
    img += rng.normal(0, noise, img.shape)
    return img, N, list(centers)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Register test strategies on entry, drop them on exit so tests don't leak into each other
    or into the real registry."""
    before = set(projection_strategies.names())
    yield
    for name in set(projection_strategies.names()) - before:
        projection_strategies._entries.pop(name, None)


def test_comb_snr_rewards_crisp_grid_over_flat():
    img, n, centers = _band_gel()
    pitch = float(np.diff(centers).mean())
    crisp = column_projection(img, 40, 70)
    flat = np.ones(img.shape[1])
    assert _comb_snr(crisp, np.array(centers), pitch) > 0.2
    assert _comb_snr(flat, np.array(centers), pitch) == pytest.approx(0.0, abs=1e-9)


def test_comb_snr_degenerate_inputs_return_zero():
    assert _comb_snr(np.ones(100), np.array([10.0, 50.0]), 20.0) == 0.0  # zero span
    assert _comb_snr(np.linspace(0, 1, 100), np.array([]), 20.0) == 0.0  # no centers
    assert _comb_snr(np.linspace(0, 1, 100), np.array([10.0]), 0.0) == 0.0  # non-positive pitch


def test_default_strategy_is_d4_union_and_surfaces_metadata():
    """The default strategy is ``d4_union``, and detection reports which projection it used.

    ``projection_source`` is the *adopted* projection, not the requested strategy: d4_union is
    only taken when its comb-SNR beats the full-height projection by ``_ADAPTIVE_MARGIN``, so
    it can legitimately report "full" on a gel where full height is already as good. Assert
    the contract (one of the two, with the metadata that goes with it) rather than a fixed
    winner, which is what made the previous version of this test stale when d4_union became
    the default and nobody noticed for two months.
    """
    import inspect

    assert inspect.signature(detect_lane_centers).parameters["projection_strategy"].default == (
        "d4_union"
    )
    img, n, _ = _band_gel()
    det = detect_lane_centers(img, n)
    assert det.projection_source in ("d4_union", "full")
    assert det.comb_snr > 0
    if det.projection_source == "full":
        assert det.row_weights is None and det.y_band is None
    else:
        assert det.y_band is not None


def test_worse_strategy_is_rejected_fallback_to_full():
    img, n, _ = _band_gel()

    @projection_strategies.register("_t_flat")
    def _flat(image, n_lanes, *, expected_pitch, y_start, y_end):
        return AdaptiveProjection(np.ones(image.shape[1]), y_band=(10, 20))

    det = detect_lane_centers(img, n, projection_strategy="_t_flat")
    assert det.projection_source == "full"  # flat projection can't beat full height
    assert det.y_band is None


def test_tying_strategy_is_rejected_by_margin():
    img, n, _ = _band_gel()

    @projection_strategies.register("_t_identity")
    def _identity(image, n_lanes, *, expected_pitch, y_start, y_end):
        return AdaptiveProjection(column_projection(image, y_start, y_end))

    det = detect_lane_centers(img, n, projection_strategy="_t_identity")
    assert det.projection_source == "full"  # equal comb-SNR is within _ADAPTIVE_MARGIN


def test_better_strategy_is_adopted_and_surfaces_band():
    img, n, _ = _band_gel()

    @projection_strategies.register("_t_band")
    def _band(image, n_lanes, *, expected_pitch, y_start, y_end):
        return AdaptiveProjection(column_projection(image, 40, 70), y_band=(40, 70))

    det = detect_lane_centers(img, n, projection_strategy="_t_band")
    assert det.projection_source == "_t_band"
    assert det.y_band == (40, 70)
    assert det.comb_snr > detect_lane_centers(img, n).comb_snr


def test_confidence_regression_blocks_adoption():
    """A crisper projection that would flip a correct 'ok' to 'low' must NOT be adopted."""
    img, n, _ = _band_gel()

    @projection_strategies.register("_t_toothin")
    def _toothin(image, n_lanes, *, expected_pitch, y_start, y_end):
        # A single-row slice: high-variance, easily crisper, but drops lanes / destabilizes
        # the fit -> low confidence. The guard must keep full height's "ok".
        return AdaptiveProjection(column_projection(image, 55, 56), y_band=(55, 56))

    full = detect_lane_centers(img, n)
    cand = detect_lane_centers(img, n, projection_strategy="_t_toothin")
    if full.confidence == "ok":
        assert cand.confidence == "ok"  # never traded a correct ok for a low
