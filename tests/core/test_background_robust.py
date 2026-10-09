"""Robust IRLS background fit: unit tests for the helpers + surface-recovery on
synthetic gels with a known background and planted signal (the correctness proof)."""

import numpy as np
import pytest

from gelpy.core.background import background_registry
from gelpy.core.background.robust_plane import (
    asymmetric_weights,
    polynomial_design,
    probabilistic_weights,
    robust_sigma,
    weighted_lstsq,
)
from gelpy.schemas.background import RobustPlaneParams


# --- synthetic gel with a known background surface ---------------------------
def _surface(h, w, coef, order):
    """Evaluate a polynomial surface on the normalized [-1,1] grid (same basis as the fit)."""
    gx, gy = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    return (polynomial_design(gx, gy, order) @ np.asarray(coef, float)).reshape(h, w)


def _bands(h, w, centers, amp=300.0, sd=6.0, rows=slice(None)):
    """Add bright vertical Gaussian bands (positive signal) over `rows`."""
    img = np.zeros((h, w))
    x = np.arange(w)
    band = np.zeros(w)
    for c in np.atleast_1d(centers):
        band += amp * np.exp(-((x - c) ** 2) / (2 * sd**2))
    img[rows, :] += band[None, :]
    return img


def _gel(h, w, coef, order, centers, noise=2.0, seed=0, **band_kw):
    truth = _surface(h, w, coef, order)
    rng = np.random.default_rng(seed)
    img = truth + rng.normal(0, noise, (h, w)) + _bands(h, w, centers, **band_kw)
    return img, truth


def _fit(img, **params):
    model = background_registry.get("robust_plane")()
    return model.remove(img, RobustPlaneParams(**params))


# --- helper units ------------------------------------------------------------
def test_polynomial_design_shapes():
    gx, gy = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-1, 1, 4))
    assert polynomial_design(gx, gy, 0).shape == (20, 1)
    assert polynomial_design(gx, gy, 1).shape == (20, 3)
    assert polynomial_design(gx, gy, 2).shape == (20, 6)


def test_weighted_lstsq_recovers_known_coefficients():
    gx, gy = np.meshgrid(np.linspace(-1, 1, 40), np.linspace(-1, 1, 30))
    design = polynomial_design(gx, gy, 2)
    coef = np.array([3.0, -2.0, 5.0, 1.5, -0.5, 0.8])
    z = design @ coef
    est = weighted_lstsq(design, z, np.ones(z.size))
    assert np.allclose(est, coef, atol=1e-6)


def test_robust_sigma_matches_true_noise():
    rng = np.random.default_rng(1)
    r = rng.normal(0, 4.0, 20000)
    assert robust_sigma(r) == pytest.approx(4.0, rel=0.05)


def test_asymmetric_weights_sides():
    r = np.array([-5.0, -0.1, 0.1, 5.0])
    w = asymmetric_weights(r, 0.02)
    assert np.allclose(w, [1.0, 1.0, 0.02, 0.02])


def test_probabilistic_weights_monotone_and_bounds():
    r = np.array([-10.0, -1.0, 0.0, 1.0, 3.0, 10.0])
    w = probabilistic_weights(r, sigma=1.0, signal_prior=0.3)
    assert np.all(w[r <= 0] == 1.0)  # background side fully trusted
    assert w[-1] < 0.05  # far above surface -> not background
    pos = w[r > 0]
    assert np.all(np.diff(pos) < 0)  # strictly descending with residual


# --- surface recovery on synthetic gels --------------------------------------
def test_recovers_tilted_plane_ignoring_bands():
    img, truth = _gel(400, 800, [100, 40, 25], 1, np.linspace(60, 740, 12), noise=2.0)
    r = _fit(img, weight_mode="probabilistic")
    err = r.background_plane - truth
    assert np.sqrt(np.mean(err**2)) < 0.5  # << the noise floor of 2.0


def test_quadratic_needs_order_2():
    coef = [100, 20, 15, -60, 0, -40]  # strong curvature
    img, truth = _gel(400, 800, coef, 2, np.linspace(60, 740, 10), noise=2.0)
    plane = _fit(img, order=1).background_plane
    quad = _fit(img, order=2).background_plane
    rms_plane = np.sqrt(np.mean((plane - truth) ** 2))
    rms_quad = np.sqrt(np.mean((quad - truth) ** 2))
    assert rms_quad < 0.5  # order 2 recovers the curved surface
    assert rms_plane > 5 * rms_quad  # a plane cannot, by a wide margin


def test_band_touching_top_and_bottom_edge():
    # The case that breaks fixed top/bottom stripes: signal fills the stripe rows.
    img, truth = _gel(300, 600, [50, 10, 8], 1, np.linspace(50, 550, 8), noise=1.5)
    img += _bands(300, 600, np.linspace(50, 550, 8), amp=250, rows=slice(0, 20))
    img += _bands(300, 600, np.linspace(50, 550, 8), amp=250, rows=slice(280, 300))
    err = _fit(img).background_plane - truth
    assert np.sqrt(np.mean(err**2)) < 1.0


def test_robust_to_high_signal_fraction():
    # Many wide bands -> >35% of pixels are signal (median/quantile heuristics ride up
    # here), but with real inter-lane gutters so background is still observable.
    img, truth = _gel(300, 900, [80, 30, 20], 1, np.linspace(70, 830, 11), noise=2.0, sd=9.0)
    frac = np.mean((img - truth) > 6 * 2.0)
    assert frac > 0.35  # genuinely signal-heavy
    err = _fit(img, weight_mode="probabilistic").background_plane - truth
    assert np.sqrt(np.mean(err**2)) < 1.5


def test_smears_are_excluded_from_the_fit():
    # A diffuse positive smear ramp above the bands must not lift the surface.
    img, truth = _gel(400, 800, [60, 5, 5], 1, np.linspace(80, 720, 10), noise=1.5)
    y = np.linspace(0, 1, 400)[:, None]
    img = img + 40.0 * y * (np.arange(800)[None, :] % 80 < 40)  # patchy positive smear
    err = _fit(img).background_plane - truth
    # surface should track the true background, not the +smear
    assert np.median(err) < 2.0


def test_no_clip_keeps_zero_centered_baseline():
    img, truth = _gel(300, 600, [40, 5, 5], 1, [150, 300, 450], noise=2.0)
    signed = _fit(img, clip_negative=False).corrected_image
    clipped = _fit(img, clip_negative=True).corrected_image
    # Background-only columns (between bands): signed residual ~0-mean; clip biases positive.
    gutter = np.r_[10:120, 500:590]
    assert abs(np.mean(signed[:, gutter])) < 0.5
    assert np.mean(clipped[:, gutter]) > np.mean(signed[:, gutter])  # rectification pedestal


def test_deterministic():
    img, _ = _gel(200, 400, [30, 10, 5], 1, [100, 200, 300], noise=2.0)
    a = _fit(img).background_plane
    b = _fit(img).background_plane
    assert np.array_equal(a, b)  # no RNG -> byte-identical (content-hash cache needs this)
