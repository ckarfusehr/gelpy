"""Unit tests for the D4 multi-band-union projection strategy.

See docs/decisions/adaptive-band/D4-multiband-union.md. These test D4's own mechanism (row scoring,
clustering, greedy coverage-gated union) — not the shared scaffold (see test_lane_strategy.py).
"""

from __future__ import annotations

import numpy as np
import pytest

from gelpy.core.lanes import (
    _cluster_rows,
    _node_coverage,
    detect_lane_centers,
)


def _two_cluster_gel(*, seed=1, noise=0.08):
    """A gel where lanes 0-2 carry a band high (y=30:50) and lanes 3-9 carry a band low
    (y=140:160) — the Cy3-shaped case: no single y-window covers every lane, only the union
    of the two disjoint clusters does."""
    rng = np.random.default_rng(seed)
    H, W, N = 200, 900, 10
    img = np.full((H, W), 0.05)
    centers = np.linspace(50, 850, N)
    x = np.arange(W)
    for i, c in enumerate(centers):
        band = img[30:50] if i < 3 else img[140:160]
        band += (0.6 * np.exp(-((x - c) ** 2) / (2 * 20**2)))[None, :]
    img += rng.normal(0, noise, img.shape)
    return img, N, list(centers)


def test_cluster_rows_merges_small_gaps_and_splits_large_ones():
    mask = np.zeros(30, dtype=bool)
    mask[5:10] = True  # one run
    mask[12:14] = True  # gap of 2 from the previous run -> merged (<= _D4_GAP_MERGE_PX=3)
    mask[25:28] = True  # far away -> separate band
    ranges = _cluster_rows(mask, gap_merge=3)
    assert ranges == [(5, 14), (25, 28)]


def test_cluster_rows_empty_mask_returns_no_bands():
    assert _cluster_rows(np.zeros(10, dtype=bool), gap_merge=3) == []


def test_node_coverage_degenerate_inputs():
    assert _node_coverage(np.array([]), np.array([10.0, 50.0]), 20.0) == 0.0
    assert _node_coverage(np.array([10.0]), np.array([]), 20.0) == 0.0
    assert _node_coverage(np.array([10.0]), np.array([10.0]), 0.0) == 0.0


def test_node_coverage_counts_only_nearby_peaks():
    centers = np.array([0.0, 100.0, 200.0])
    pitch = 100.0
    # one peak near node 0, none near nodes 1/2
    assert _node_coverage(np.array([2.0]), centers, pitch) == pytest.approx(1 / 3)
    # peaks near all three nodes
    assert _node_coverage(np.array([1.0, 99.0, 201.0]), centers, pitch) == pytest.approx(1.0)


def test_d4_keeps_both_disjoint_clusters_and_covers_every_lane():
    """The core D4 hypothesis: on a gel with two disjoint y-bands (some lanes only visible in
    each), the strategy must keep BOTH clusters — a single-window strategy structurally can't."""
    img, n, centers = _two_cluster_gel()
    det = detect_lane_centers(img, n, projection_strategy="d4_union")
    assert det.projection_source == "d4_union"  # adopted (crisper union beats full height)
    assert isinstance(det.y_band, list) and len(det.y_band) >= 2
    # both the high cluster (~30-50) and low cluster (~140-160) are represented in the union
    los = [lo for lo, hi in det.y_band]
    his = [hi for lo, hi in det.y_band]
    assert min(los) < 60  # a kept band overlaps the high cluster
    assert max(his) > 130  # a kept band overlaps the low cluster
    # every lane is actually covered (the point of the coverage-gated greedy keep)
    assert det.n_detected >= n - 1  # allow for at most one missed peak before grid-fit recovers it
    max_err = max(abs(a - b) for a, b in zip(det.centers, centers))
    assert max_err / (np.diff(centers).mean()) < 0.25  # passes the harness tolerance


def test_d4_falls_back_when_nothing_clears_the_threshold():
    """A flat image (no row is more comb-like than any other) must not fabricate a band."""
    img = np.full((100, 400), 0.5) + np.random.default_rng(0).normal(0, 0.01, (100, 400))
    det = detect_lane_centers(img, 5, projection_strategy="d4_union")
    assert det.projection_source == "full"


def test_d4_harness_composite_at_least_matches_baseline():
    from tests.synthetic.harness import aggregate, baseline_detect, evaluate

    mine = lambda img, n: detect_lane_centers(img, n, projection_strategy="d4_union")
    results = evaluate({"baseline": baseline_detect, "d4": mine})
    base_agg = aggregate(results["baseline"])
    d4_agg = aggregate(results["d4"])
    assert d4_agg.composite >= base_agg.composite
    assert d4_agg.false_ok_rate == 0.0
