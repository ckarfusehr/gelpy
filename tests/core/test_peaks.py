"""Peak shapes + driver + grow_select strategy (the new engine)."""

import numpy as np
from scipy.integrate import simpson

from gelpy.core.peaks import peak_registry
from gelpy.core.peaks.driver import Component, estimate_noise, fit_components
from gelpy.core.peaks.strategies import fit_strategy_registry
from gelpy.schemas.fit_config import FitConfig


def gaussian(x, amp, mean, sd):
    return amp * np.exp(-((x - mean) ** 2) / (2 * sd**2))


def test_registry_has_all_shapes():
    assert set(peak_registry.names()) == {"gaussian", "emg", "bi_gaussian", "pseudo_voigt"}


def test_closed_form_area_matches_numeric():
    x = np.linspace(-500, 500, 500001)
    for name, params in [
        ("gaussian", (2.0, 10.0, 8.0)),
        ("emg", (2.0, 10.0, 8.0, 25.0)),
        ("bi_gaussian", (2.0, 10.0, 5.0, 20.0)),
    ]:
        m = peak_registry.get(name)()
        assert np.isclose(m.area(*params), simpson(m.single_peak(x, *params), x=x), rtol=2e-3)


def test_emg_signed_tau_mirrors_and_area_invariant():
    e = peak_registry.get("emg")()
    x = np.linspace(-100, 200, 3001)
    right = e.single_peak(x, 2.0, 50.0, 8.0, 30.0)
    left = e.single_peak(2 * 50.0 - x, 2.0, 50.0, 8.0, -30.0)
    assert np.allclose(right, left, atol=1e-9)
    assert np.isclose(e.area(2, 50, 8, 30), e.area(2, 50, 8, -30))
    assert e.peak_maximum_x(2, 50, 8, 30) > 50  # right tail -> mode right of center
    assert e.peak_maximum_x(2, 50, 8, -30) < 50


def test_emg_no_overflow_far_tail():
    e = peak_registry.get("emg")()
    vals = e.single_peak(np.array([-400.0, 0.0, 400.0]), 1.0, 10.0, 2.0, 0.5)
    assert np.all(np.isfinite(vals))


def test_driver_recovers_two_peaks_and_covariance():
    x = np.arange(250, dtype=float)
    signal = gaussian(x, 1.0, 100.0, 8.0) + gaussian(x, 0.6, 140.0, 6.0)
    signal = signal + np.random.default_rng(0).normal(0, 0.01, x.size)
    g = peak_registry.get("gaussian")()
    comps = [Component(g, (0.8, 97.0, 10.0), 97.0, 10.0), Component(g, (0.5, 143.0, 5.0), 143.0, 5.0)]
    r = fit_components(signal, comps, width_prior_strength=0.5)
    assert np.isclose(r.components[0].area(), 1.0 * 8 * np.sqrt(2 * np.pi), rtol=0.05)
    assert r.covariance is not None
    assert estimate_noise(signal) < 0.05


def test_grow_select_two_gaussians():
    x = np.arange(300, dtype=float)
    profile = 0.2 + gaussian(x, 1.0, 90.0, 7.0) + gaussian(x, 0.8, 210.0, 7.0)
    r = fit_strategy_registry.get("grow_select")(profile, FitConfig(), lane_index=0, label="two")
    assert r.status == "ok"
    assert len(r.bands) == 2
    assert abs(sum(b.area_fraction for b in r.bands) - (r.coverage or 0)) < 1e-6


def test_grow_select_flat_is_empty():
    profile = np.full(200, 0.3) + np.random.default_rng(1).normal(0, 0.001, 200)
    r = fit_strategy_registry.get("grow_select")(profile, FitConfig(), lane_index=0, label="flat")
    assert r.status == "empty"
    assert r.confidence == "low"
