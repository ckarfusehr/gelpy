"""Smoke test: the fitter runs on real extracted lanes without crashing and returns sane output."""

from __future__ import annotations

import pytest

from tests.synthetic.fit_real import DEFAULT_TIF, fit_real

pytestmark = pytest.mark.skipif(not DEFAULT_TIF.exists(), reason="example TIFF absent")


def test_fit_real_smoke():
    results = fit_real(n_lanes=20)
    assert len(results) == 20
    for r in results:
        assert r.status in {"ok", "empty", "failed"}
        if r.status == "ok":
            assert r.coverage is not None
            # areas are on the true-signal basis; fractions of ok bands sum near coverage
            total = sum(b.area_fraction for b in r.bands)
            assert abs(total - r.coverage) < 1e-6
