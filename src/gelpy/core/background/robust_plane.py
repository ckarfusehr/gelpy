"""Robust background surface fit by iterative reweighted least squares (IRLS).

Replaces ``plane_ransac``'s two hard-coded background stripes. No user input is needed:
sample signal (bands, smears) only ever *adds* intensity, so the true background lives on
the low (sub-surface) side of the fit residual. The reweighting keys on that side and lets
a low-order polynomial surface settle onto the background, ignoring signal wherever it is.

Two weight modes share one M-step (weighted-least-squares polynomial fit):

* ``asymmetric`` — Eilers ALS: a fixed small weight for above-surface pixels. Simple.
* ``probabilistic`` (default) — a two-component posterior P(background | residual) with the
  noise scale ``sigma`` estimated from the negative residual side each iteration, so it
  self-scales to the noise floor and stays unbiased even when signal covers a large image
  fraction (the real gels run 27-42 % signal pixels).

Fully deterministic (no RNG), which the content-hash cache requires.
"""

from __future__ import annotations

import numpy as np

from ...schemas.background import RobustPlaneParams
from .base import BackgroundFitResult, BackgroundModel, background_registry

_TINY = 1e-12
_INV_SQRT_2PI = 0.3989422804014327


def polynomial_design(gx: np.ndarray, gy: np.ndarray, order: int) -> np.ndarray:
    """Column-stacked polynomial terms in normalized coords ``gx, gy`` (raveled).

    order 0 -> [1]; 1 -> [1, x, y]; 2 -> [1, x, y, x^2, xy, y^2].
    """
    x, y = gx.ravel(), gy.ravel()
    cols = [np.ones_like(x)]
    if order >= 1:
        cols += [x, y]
    if order >= 2:
        cols += [x * x, x * y, y * y]
    return np.stack(cols, axis=1)


def weighted_lstsq(design: np.ndarray, z: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Solve the weighted normal equations ``(AᵀWA) θ = AᵀW z`` for the coefficients.

    Normal equations (a k×k solve, k ∈ {1,3,6}) are far cheaper than an SVD over the
    millions of pixels. A tiny diagonal ridge guards against a degenerate weight field
    (e.g. all weight collapsed onto one row) making the system singular.
    """
    wa = design * weights[:, None]
    ata = design.T @ wa
    atz = design.T @ (weights * z)
    ridge = 1e-9 * np.trace(ata) / ata.shape[0] + _TINY
    ata[np.diag_indices_from(ata)] += ridge
    return np.linalg.solve(ata, atz)


def robust_sigma(residual: np.ndarray) -> float:
    """Noise scale from the negative residual side (assumed pure background noise).

    Signal cannot be negative, so ``residual < 0`` is background noise seen from below. For
    zero-mean Gaussian noise, ``1.4826·median(|r₋|) = σ``. Falls back to the global MAD if
    too few negatives (e.g. a surface still biased low early on).
    """
    neg = residual[residual < 0]
    if neg.size >= 32:
        return float(1.4826 * np.median(np.abs(neg))) + _TINY
    med = np.median(residual)
    return float(1.4826 * np.median(np.abs(residual - med))) + _TINY


def asymmetric_weights(residual: np.ndarray, p: float) -> np.ndarray:
    """Eilers ALS weights: ``p`` above the surface (signal side), ``1`` below."""
    return np.where(residual > 0.0, p, 1.0)


def probabilistic_weights(residual: np.ndarray, sigma: float, signal_prior: float) -> np.ndarray:
    """Posterior P(pixel is background | residual) for a 2-component mixture.

    Background ~ N(0, σ²); signal ~ one-sided exponential on the positive side (mean σ).
    Below the surface (r ≤ 0) the signal density is 0, so weight is exactly 1 — the
    background side is fully trusted. Above, the weight descends smoothly toward 0 as the
    residual climbs past a few σ.
    """
    z = residual / sigma
    w = np.ones_like(residual)
    pos = z > 0.0
    zp = z[pos]
    bg = _INV_SQRT_2PI * np.exp(-0.5 * zp * zp)
    sig = np.exp(-zp)  # exponential(rate=1) density on z>0
    pi = signal_prior
    w[pos] = (1.0 - pi) * bg / ((1.0 - pi) * bg + pi * sig + _TINY)
    return w


def _smooth_weights(weights: np.ndarray, shape: tuple[int, int], sigma_px: float) -> np.ndarray:
    """Blur the weight field so trust is spatially coherent (contiguity prior)."""
    from scipy.ndimage import gaussian_filter

    smoothed = gaussian_filter(weights.reshape(shape), sigma=sigma_px, mode="nearest")
    return smoothed.ravel()


def fit_surface(image: np.ndarray, params: RobustPlaneParams):
    """IRLS fit of a background surface. Returns ``(surface, weights, diagnostics)``.

    ``surface`` and ``weights`` are full-image arrays; ``weights`` is the final per-pixel
    background trust (recomputed on the full grid for the diagnostic even when the fit ran
    on a subsampled grid).
    """
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
    prev_theta = None
    theta = None
    n_iter = 0
    for n_iter in range(1, params.max_iter + 1):
        theta = weighted_lstsq(design_fit, z_fit, weights)
        # Convergence on the fitted surface itself (the quantity we care about) — the soft
        # probabilistic weights keep jittering after the surface has settled, so a
        # weight-based criterion would never trip.
        if prev_theta is not None:
            rel = float(np.max(np.abs(theta - prev_theta)) / (np.max(np.abs(theta)) + _TINY))
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
    full_w = full_w.reshape(h, w)

    diagnostics = {f"c{i}": float(v) for i, v in enumerate(theta)}
    diagnostics.update(
        order=float(params.order),
        sigma=float(full_sigma),
        n_iter=float(n_iter),
        inlier_fraction=float(np.mean(full_w)),
    )
    return surface, full_w, diagnostics


@background_registry.register(
    "robust_plane",
    config_model=RobustPlaneParams,
    description="Iterative-reweighted least-squares fit of a low-order background surface; "
    "no stripes needed, robust to bands/smears via the signal-is-positive asymmetry.",
)
class RobustPlane(BackgroundModel):
    name = "robust_plane"

    def remove(self, image: np.ndarray, params: RobustPlaneParams) -> BackgroundFitResult:
        image = image.astype(np.float64, copy=False)
        surface, weights, diagnostics = fit_surface(image, params)
        corrected = image - surface
        if params.clip_negative:
            corrected = np.clip(corrected, 0.0, None)
        return BackgroundFitResult(
            corrected_image=corrected,
            background_plane=surface,
            params=diagnostics,
            inlier_fraction=diagnostics["inlier_fraction"],
            weights=weights,
        )
