"""Shared robust-MAP fitting driver.

Fits a set of (possibly heterogeneously shaped) peak components to a baseline-subtracted
1-D signal, jointly, with:

- **robust loss** (``soft_l1`` by default) so dust spikes / outliers don't drag real bands,
- **soft priors** appended as extra residual rows (MAP = least-squares + penalties): a
  log-normal width prior anchoring each component to its detected scale, and a shrinkage
  prior pulling EMG tails toward 0 (symmetric unless the data earns a tail),
- **bounds** from each parameter's *role* (amplitude >= 0, center near its seed, width in
  ``[MIN_WIDTH, max_width_frac*L]``, tail bounded both ways, eta in [0, 1]),
- **covariance** from the Gauss-Newton Jacobian at the solution, for per-peak uncertainty.

Residuals are normalized by an estimated noise sigma so priors, robust ``f_scale``, and the
covariance are all on a data-scale-independent footing (important because area-normalized
profiles are tiny).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log

import numpy as np
import scipy.optimize as opt

from .base import PeakModel

MIN_WIDTH = 0.5  # px; lower bound for any width/scale parameter
_ROBUST_F_SCALE = 3.0  # robustify residuals beyond ~3 sigma (units of estimated noise)
_MAX_NFEV_PER_PARAM = 200


@dataclass
class Component:
    """One peak component: a shape model, its (seed or fitted) params, and its seed anchors."""

    model: PeakModel
    params: tuple[float, ...]
    seed_center: float
    seed_scale: float

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return self.model.single_peak(x, *self.params)

    def area(self) -> float:
        return self.model.area(*self.params)

    def maximum_x(self) -> float:
        return self.model.peak_maximum_x(*self.params)


@dataclass
class FitResult:
    components: list[Component]
    covariance: np.ndarray | None
    slices: list[tuple[int, int]]  # (start, stop) into the flat param/cov vector, per component
    sse: float  # data-only sum of squared residuals (raw units)
    noise_sigma: float
    residual: np.ndarray  # data residual (model - signal), raw units
    at_bound: list[bool] = field(default_factory=list)  # per component
    success: bool = True
    n_params: int = 0


# --- numeric helpers ----------------------------------------------------------
_NOISE_FLOOR_FRAC = 8e-3  # noise >= this fraction of the signal's robust peak-to-peak range


def estimate_noise(signal: np.ndarray) -> float:
    """Robust noise sigma from the MAD of the 2nd difference (removes smooth trend).

    Floored to a small fraction of the signal's peak-to-peak range: real digitized profiles
    can have near-zero pixel-noise, which would drive the ``min_snr`` seeding gate to zero and
    make the fitter chase every sub-pixel wiggle (over-fitting + very slow). The floor only
    engages when the MAD estimate is unrealistically tiny; on noisy synthetic/real data the
    MAD dominates and the floor is inert.
    """
    s = np.asarray(signal, dtype=np.float64)
    if s.size < 5:
        return max(float(np.std(s)), 1e-12)
    d2 = s[2:] - 2.0 * s[1:-1] + s[:-2]
    mad = float(np.median(np.abs(d2 - np.median(d2))))
    sigma = 1.4826 * mad / np.sqrt(6.0)  # Var(2nd diff) = 6 sigma^2
    ptp = float(np.percentile(s, 99) - np.percentile(s, 1))
    return max(sigma, _NOISE_FLOOR_FRAC * ptp, 1e-12)


def aicc(sse: float, n: int, k: int) -> float:
    """Small-sample corrected AIC for a Gaussian-residual least-squares fit."""
    if n <= 0:
        return float("inf")
    sse = max(sse, 1e-300)
    aic = n * log(sse / n) + 2.0 * k
    denom = n - k - 1
    if denom <= 0:
        return float("inf")  # too many params to be identifiable
    return aic + 2.0 * k * (k + 1) / denom


def bic(sse: float, n: int, k: int) -> float:
    if n <= 0:
        return float("inf")
    sse = max(sse, 1e-300)
    return n * log(sse / n) + k * log(n)


def evaluate_composite(components: list[Component], x: np.ndarray) -> np.ndarray:
    """Sum all components' contributions at ``x``."""
    total = np.zeros_like(x, dtype=np.float64)
    for comp in components:
        total = total + comp.evaluate(x)
    return total


# --- bounds & priors (role-driven, model-agnostic) ---------------------------
def _component_bounds(
    model: PeakModel, seed_center: float, seed_scale: float, length: int, max_width_frac: float
) -> tuple[list[float], list[float]]:
    lo: list[float] = []
    hi: list[float] = []
    max_width = max(max_width_frac * length, MIN_WIDTH * 2)
    center_slack = max(2.0 * seed_scale, 4.0)
    for role in model.param_roles():
        if role == "amplitude":
            lo.append(0.0)
            hi.append(np.inf)
        elif role == "center":
            lo.append(max(0.0, seed_center - center_slack))
            hi.append(min(float(length - 1), seed_center + center_slack))
        elif role == "width":
            lo.append(MIN_WIDTH)
            hi.append(max_width)
        elif role == "tail":
            lo.append(-max_width)
            hi.append(max_width)
        elif role == "unit":
            lo.append(0.0)
            hi.append(1.0)
        else:  # pragma: no cover - unknown role
            lo.append(-np.inf)
            hi.append(np.inf)
    return lo, hi


def _prior_rows(
    model: PeakModel,
    params: np.ndarray,
    seed_scale: float,
    width_strength: float,
    tail_strength: float,
) -> list[float]:
    rows: list[float] = []
    roles = model.param_roles()
    anchor = max(seed_scale, MIN_WIDTH)
    for value, role in zip(params, roles):
        if role == "width" and width_strength > 0:
            rows.append(width_strength * log(max(float(value), 1e-6) / anchor))
        elif role == "tail" and tail_strength > 0:
            rows.append(tail_strength * float(value) / anchor)
    return rows


def _clip_into(p0: list[float], lo: list[float], hi: list[float]) -> np.ndarray:
    arr = np.array(p0, dtype=np.float64)
    return np.clip(arr, np.array(lo) + 1e-9, np.array(hi) - 1e-9)


# --- the fit ------------------------------------------------------------------
def fit_components(
    signal: np.ndarray,
    components: list[Component],
    *,
    x: np.ndarray | None = None,
    robust_loss: str = "soft_l1",
    width_prior_strength: float = 0.5,
    tail_shrinkage: float = 0.3,
    max_width_frac: float = 0.9,
    noise_sigma: float | None = None,
) -> FitResult:
    """Jointly fit ``components`` to ``signal`` (already baseline-subtracted)."""
    signal = np.asarray(signal, dtype=np.float64)
    length = signal.size
    if x is None:
        x = np.arange(length, dtype=np.float64)
    if noise_sigma is None:
        noise_sigma = estimate_noise(signal)

    if not components:
        return FitResult(
            components=[], covariance=None, slices=[], sse=float(np.sum(signal**2)),
            noise_sigma=noise_sigma, residual=-signal, at_bound=[], success=True, n_params=0,
        )

    # Flatten params, bounds, and record per-component slices.
    p0: list[float] = []
    lo: list[float] = []
    hi: list[float] = []
    slices: list[tuple[int, int]] = []
    for comp in components:
        start = len(p0)
        p0.extend(comp.params)
        clo, chi = _component_bounds(
            comp.model, comp.seed_center, comp.seed_scale, length, max_width_frac
        )
        lo.extend(clo)
        hi.extend(chi)
        slices.append((start, len(p0)))
    p0_arr = _clip_into(p0, lo, hi)
    n_params = len(p0_arr)

    def residuals(theta: np.ndarray) -> np.ndarray:
        model_sum = np.zeros_like(signal)
        prior: list[float] = []
        for comp, (a, b) in zip(components, slices):
            pslice = theta[a:b]
            model_sum = model_sum + comp.model.single_peak(x, *pslice_tuple(pslice))
            prior.extend(
                _prior_rows(comp.model, pslice, comp.seed_scale, width_prior_strength, tail_shrinkage)
            )
        data = (model_sum - signal) / noise_sigma
        if prior:
            return np.concatenate([data, np.asarray(prior, dtype=np.float64)])
        return data

    try:
        result = opt.least_squares(
            residuals,
            p0_arr,
            bounds=(np.array(lo), np.array(hi)),
            loss=robust_loss,
            f_scale=_ROBUST_F_SCALE,
            max_nfev=_MAX_NFEV_PER_PARAM * n_params,
        )
        theta = result.x
        success = bool(result.success) or result.status > 0
        cov = _covariance(result.jac)
    except (RuntimeError, ValueError) as exc:  # pragma: no cover - defensive
        theta = p0_arr
        success = False
        cov = None
        _ = exc

    # Build fitted components + diagnostics.
    fitted: list[Component] = []
    at_bound: list[bool] = []
    for comp, (a, b) in zip(components, slices):
        params = tuple(float(v) for v in theta[a:b])
        fitted.append(Component(comp.model, params, comp.seed_center, comp.seed_scale))
        at_bound.append(_hit_width_bound(comp.model, theta[a:b], lo[a:b], hi[a:b]))

    model_sum = evaluate_composite(fitted, x)
    residual = model_sum - signal
    sse = float(np.sum(residual**2))
    return FitResult(
        components=fitted, covariance=cov, slices=slices, sse=sse, noise_sigma=noise_sigma,
        residual=residual, at_bound=at_bound, success=success, n_params=n_params,
    )


def pslice_tuple(pslice: np.ndarray) -> tuple[float, ...]:
    return tuple(float(v) for v in pslice)


def _covariance(jac: np.ndarray) -> np.ndarray | None:
    """Posterior covariance ~ inv(J^T J) at the solution (priors included as extra rows)."""
    try:
        jtj = jac.T @ jac
        return np.linalg.inv(jtj)
    except np.linalg.LinAlgError:
        try:
            return np.linalg.pinv(jac.T @ jac)
        except np.linalg.LinAlgError:  # pragma: no cover
            return None


def _hit_width_bound(model: PeakModel, theta, lo, hi, rtol: float = 1e-3) -> bool:
    """A width-role parameter pinned at its UPPER bound: a runaway width (a 'peak' spanning the
    trace to absorb drift) — the real failure mode. A center at the profile edge (expected for
    edge peaks) or a width at the lower bound (a legitimately narrow dust spike) is not flagged.
    """
    for idx in model.role_indices("width"):
        value, high = theta[idx], hi[idx]
        if np.isfinite(high) and abs(value - high) <= rtol * (abs(high) + 1.0):
            return True
    return False
