"""1-D baseline estimation under a lane profile (fit-time, per-lane).

Deliberately *conservative*: smears are signal, so the baseline models only true
low-frequency background and must not bend to follow a broad band.

The default is a low-order **polynomial** (``method="poly"``, ``poly_order=2``), not arPLS.
Measured absorption of a genuine broad smear (sigma=55 px) sitting on a flat residual
background, as a fraction of the smear's true area:

===================  ==========
baseline             absorbed
===================  ==========
``none``                  0.0%
``poly`` order 0          2.1%
``poly`` order 1          2.2%
``poly`` order 2          5.5%
``poly`` order 3         13.1%
``arpls`` lam=1e5        34.4%
``asls`` lam=1e5         58.5%
===================  ==========

More baseline freedom means more signal eaten, monotonically, which is why the stiffest
option that can still track real drift is the default. arPLS/AsLS (asymmetrically reweighted
penalized least squares, Baek et al. 2015 — a Whittaker smoother whose weights shrink for
points above the baseline) remain available for gels with genuinely wiggly, non-polynomial
backgrounds, at the cost shown above. The near-origin **well** region is optionally
zero-weighted so that large loading-pocket signal is never absorbed as background.

Whether *any* fit-time baseline is the right default after a good 2-D background subtraction
is an open question, with this measurement recorded in
``docs/decisions/2026-07-15-baseline-redundancy-and-well-boundary.md``.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from ...schemas.baseline import BaselineConfig


def well_region(length: int, well_frac: float) -> np.ndarray:
    """Boolean mask of the near-origin well region (first ``well_frac`` of the window)."""
    mask = np.zeros(length, dtype=bool)
    n = int(round(well_frac * length))
    if n > 0:
        mask[:n] = True
    return mask


def estimate_baseline(y: np.ndarray, config: BaselineConfig) -> np.ndarray:
    """Estimate the baseline of ``y`` per ``config``; returns an array the length of ``y``."""
    y = np.asarray(y, dtype=np.float64)
    length = y.size
    if config.method == "none" or length < 5:
        return np.zeros_like(y)

    protect = well_region(length, config.well_frac) if config.protect_well else None
    if config.method == "poly":
        return _poly_baseline(y, config.poly_order, protect=protect)
    if config.method == "asls":
        return _penalized(y, config.lam, protect=protect, updater=_asls_weights, p=config.p)
    # default: arpls
    return _penalized(y, config.lam, protect=protect, updater=_arpls_weights)


# --- Whittaker penalized-least-squares core ----------------------------------
def _whittaker(y: np.ndarray, w: np.ndarray, lam: float) -> np.ndarray:
    """Solve (W + lam D2^T D2) z = W y for the smoothed baseline z (2nd-difference penalty)."""
    length = y.size
    d2 = sp.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(length - 2, length))
    penalty = lam * (d2.T @ d2)
    lhs = sp.diags(w) + penalty
    return spsolve(lhs.tocsc(), w * y)


def _arpls_weights(y: np.ndarray, z: np.ndarray, _p: float) -> np.ndarray:
    """arPLS logistic reweighting: points below the baseline keep weight ~1; points above
    (i.e. peaks) are down-weighted by how far above the negative-residual noise they sit."""
    d = y - z
    neg = d[d < 0]
    if neg.size == 0:
        return np.ones_like(y)
    m = float(neg.mean())
    s = float(neg.std()) + 1e-12
    return 1.0 / (1.0 + np.exp(np.clip(2.0 * (d - (2.0 * s - m)) / s, -50.0, 50.0)))


def _asls_weights(y: np.ndarray, z: np.ndarray, p: float) -> np.ndarray:
    """AsLS asymmetric weights: p above the baseline, (1-p) below."""
    return np.where(y > z, p, 1.0 - p)


def _penalized(
    y: np.ndarray,
    lam: float,
    *,
    protect: np.ndarray | None,
    updater,
    p: float = 0.01,
    niter: int = 50,
    tol: float = 1e-3,
) -> np.ndarray:
    w = np.ones_like(y)
    if protect is not None:
        w[protect] = 0.0
    z = np.zeros_like(y)
    for _ in range(niter):
        z = _whittaker(y, w, lam)
        w_new = updater(y, z, p)
        if protect is not None:
            w_new[protect] = 0.0
        denom = float(np.linalg.norm(w)) + 1e-12
        if float(np.linalg.norm(w_new - w)) / denom < tol:
            w = w_new
            break
        w = w_new
    return z


def _poly_baseline(
    y: np.ndarray, order: int, *, protect: np.ndarray | None, niter: int = 40, tol: float = 1e-3
) -> np.ndarray:
    """Robustly reweighted **polynomial** baseline (arPLS weights, polynomial basis).

    arPLS-style asymmetric reweighting pins the baseline to the correct *level* (peaks are
    down-weighted smoothly, not hard-excluded), while the low-order polynomial basis cannot
    dip under a peak or bend to a localized smear. Hard exclusion + a high-order basis (the
    naive approach) instead ratchets down to the lower noise envelope and dips across excluded
    peak gaps, badly inflating the recovered signal area on faint lanes.
    """
    length = y.size
    x = np.linspace(-1.0, 1.0, length)
    w = np.ones(length)
    if protect is not None:
        w[protect] = 0.0
    z = np.zeros_like(y)
    for _ in range(niter):
        if float(w.sum()) < order + 1:
            break
        coeffs = np.polyfit(x, y, order, w=np.sqrt(w))
        z = np.polyval(coeffs, x)
        d = y - z
        neg = d[d < 0]
        if neg.size == 0:
            break
        m = float(neg.mean())
        s = float(neg.std()) + 1e-12
        w_new = 1.0 / (1.0 + np.exp(np.clip(2.0 * (d - (2.0 * s - m)) / s, -50.0, 50.0)))
        if protect is not None:
            w_new[protect] = 0.0
        if float(np.linalg.norm(w_new - w)) / (float(np.linalg.norm(w)) + 1e-12) < tol:
            w = w_new
            break
        w = w_new
    return z
