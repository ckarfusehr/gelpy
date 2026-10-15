"""Robust IRLS background fit: unit tests for the helpers + surface-recovery on
synthetic gels with a known background and planted signal (the correctness proof)."""

import tracemalloc

import numpy as np
import pytest

from gelpy.core.background import background_registry
from gelpy.core.background.robust_plane import (
    _smooth_weights,
    asymmetric_weights,
    evaluate_surface,
    fit_surface,
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


def _reference_fit_surface(image, params):
    """The previous implementation, verbatim apart from names: it builds full-resolution grids
    and the full design matrix even when subsampling. Kept only as the yardstick."""
    h, w = image.shape
    gx_full, gy_full = np.meshgrid(np.linspace(-1.0, 1.0, w), np.linspace(-1.0, 1.0, h))
    design_full = polynomial_design(gx_full, gy_full, params.order)
    s = params.subsample
    if s > 1:
        gx_fit, gy_fit = gx_full[::s, ::s], gy_full[::s, ::s]
        fit_shape = gx_fit.shape
        design_fit = polynomial_design(gx_fit, gy_fit, params.order)
        z_fit = image[::s, ::s].ravel().astype(np.float64)
    else:
        fit_shape = (h, w)
        design_fit = design_full
        z_fit = image.ravel().astype(np.float64)
    contiguity_px = params.contiguity / s if params.contiguity > 0 else 0.0
    weights = np.ones_like(z_fit)
    prev_theta = theta = None
    for _ in range(params.max_iter):
        theta = weighted_lstsq(design_fit, z_fit, weights)
        if prev_theta is not None:
            rel = float(np.max(np.abs(theta - prev_theta)) / (np.max(np.abs(theta)) + 1e-12))
            if rel < params.tol:
                break
        prev_theta = theta
        residual = z_fit - design_fit @ theta
        sigma = robust_sigma(residual)
        if params.weight_mode == "asymmetric":
            weights = asymmetric_weights(residual, params.asymmetry_p)
        else:
            weights = probabilistic_weights(residual, sigma, params.signal_prior)
        if contiguity_px > 0:
            weights = _smooth_weights(weights, fit_shape, contiguity_px)
    surface = (design_full @ theta).reshape(h, w)
    full_residual = image - surface
    full_sigma = robust_sigma(full_residual.ravel())
    if params.weight_mode == "asymmetric":
        full_w = asymmetric_weights(full_residual.ravel(), params.asymmetry_p)
    else:
        full_w = probabilistic_weights(full_residual.ravel(), full_sigma, params.signal_prior)
    return surface, full_w.reshape(h, w)


COEFS = [10.0, 2.0, -1.0, 0.5, 0.2, -0.3]


@pytest.mark.parametrize("order", [0, 1, 2])
def test_evaluate_surface_matches_the_design_matrix(order):
    """Broadcasting two 1-D coordinate vectors gives the terms of ``polynomial_design``."""
    theta = np.asarray(COEFS[: (1, 3, 6)[order]])
    gx, gy = np.linspace(-1.0, 1.0, 37), np.linspace(-1.0, 1.0, 23)
    gxx, gyy = np.meshgrid(gx, gy)
    expected = (polynomial_design(gxx, gyy, order) @ theta).reshape(23, 37)
    surface = evaluate_surface(theta, gx, gy, order)
    assert surface.shape == (23, 37)
    np.testing.assert_allclose(surface, expected, rtol=1e-12, atol=1e-12)


def test_evaluate_surface_makes_one_full_size_temporary_at_order_2():
    """The surface itself and one scratch array for the xy term: two image-size arrays, not three."""
    theta = np.asarray(COEFS)
    gx, gy = np.linspace(-1.0, 1.0, 1200), np.linspace(-1.0, 1.0, 1000)
    tracemalloc.start()
    try:
        evaluate_surface(theta, gx, gy, 2)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 2.5 * 1000 * 1200 * 8, peak


@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize("size", [(300, 400), (301, 403)], ids=["300x400", "301x403"])
def test_a_subsampled_fit_gives_the_same_surface_as_before(order, size):
    """The surface covers every row and column, whether or not the stride of 4 lands on the last
    one: 301 rows end on a sampled row, 403 columns leave two columns after the last sampled."""
    img, _ = _gel(*size, COEFS[: (1, 3, 6)[order]], order, [80, 200, 320])
    params = RobustPlaneParams(order=order, subsample=4)
    surface, weights, _ = fit_surface(img, params)
    ref_surface, ref_weights = _reference_fit_surface(img, params)
    np.testing.assert_allclose(surface, ref_surface, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(weights, ref_weights, rtol=1e-6, atol=1e-9)


def test_the_default_fit_is_bit_identical_to_before():
    img, _ = _gel(200, 260, COEFS, 2, [60, 130, 200])
    params = RobustPlaneParams(order=2)  # subsample=1
    surface, _, _ = fit_surface(img, params)
    assert np.array_equal(surface, _reference_fit_surface(img, params)[0])


def test_a_subsampled_fit_never_builds_full_resolution_grids():
    img, _ = _gel(1000, 1200, COEFS, 2, [200, 600, 1000])
    params = RobustPlaneParams(order=2, subsample=4)

    def peak(fit):
        tracemalloc.start()
        try:
            fit(img, params)
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    before, after = peak(_reference_fit_surface), peak(fit_surface)
    # Measured 0.54; the two grids kept alive alone give 0.66, which 0.7 would let through.
    assert after < 0.6 * before, (after, before)
