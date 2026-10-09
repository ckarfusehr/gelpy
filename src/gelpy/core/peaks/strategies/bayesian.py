"""Tier 3 - bayesian: seeded posterior sampling around the MAP for uncertainty + robustness.

Escalation tier for lanes the quality signal flags. It takes the grow_select MAP decomposition
(structure + shapes + parameters + Laplace covariance), then runs a **seeded** random-walk
Metropolis sampler over the joint parameter posterior — Gaussian likelihood
(``-0.5*||(model-signal)/noise||^2``) plus the same soft priors the driver uses (log-normal
width, tail shrinkage, non-negativity via bounds). Posterior-mean parameters become the
reported fit and the posterior covariance flows through the standard assembly, so per-band
``area_uncertainty`` becomes a genuine credible interval rather than a local linearization.

Self-contained (numpy only) and seeded from ``FitConfig.seed`` -> deterministic and cacheable
like every other stage. N is taken from grow_select's AICc model selection (already a
principled choice); the sampler quantifies parameter/area uncertainty given that structure.
"""

from __future__ import annotations

import numpy as np

from ....schemas.fit_config import FitConfig
from ....schemas.peaks import LaneFitResult
from ..driver import (
    Component,
    FitResult,
    _component_bounds,
    _prior_rows,
    evaluate_composite,
)
from ._common import assemble_result
from .grow_select import _greedy_grow, prepare, refine
from .registry import fit_strategy_registry

_N_STEPS = 6000
_BURN_IN = 1500
_THIN = 5


def _log_posterior(theta, comps, slices, signal, x, noise, lo, hi, config):
    if np.any(theta < lo) or np.any(theta > hi):
        return -np.inf
    model = np.zeros_like(signal)
    prior_sq = 0.0
    for comp, (a, b) in zip(comps, slices):
        pslice = theta[a:b]
        model = model + comp.model.single_peak(x, *(float(v) for v in pslice))
        rows = _prior_rows(
            comp.model, pslice, comp.seed_scale,
            config.priors.width_prior_strength, config.priors.tail_shrinkage,
        )
        prior_sq += float(np.sum(np.square(rows))) if rows else 0.0
    resid = (model - signal) / noise
    return -0.5 * float(np.sum(resid**2)) - 0.5 * prior_sq


def _proposal_cholesky(cov, ndim, param_scale, rng):
    """Scaled Cholesky factor for the random-walk proposal (MAP covariance if usable)."""
    step = (2.4 / np.sqrt(ndim)) ** 2  # optimal RW-Metropolis scaling
    if cov is not None and np.all(np.isfinite(cov)) and cov.shape == (ndim, ndim):
        try:
            return np.linalg.cholesky(step * cov + 1e-12 * np.eye(ndim))
        except np.linalg.LinAlgError:
            pass
    return np.diag(np.sqrt(step) * (0.05 * np.abs(param_scale) + 1e-6))


def _sample(map_fit: FitResult, signal, x, noise, config) -> np.ndarray | None:
    comps = map_fit.components
    slices = map_fit.slices
    theta = np.array([v for c in comps for v in c.params], dtype=np.float64)
    ndim = theta.size
    if ndim == 0:
        return None
    lo_list: list[float] = []
    hi_list: list[float] = []
    for comp in comps:
        clo, chi = _component_bounds(
            comp.model, comp.seed_center, comp.seed_scale, signal.size, config.priors.max_width_frac
        )
        lo_list.extend(clo)
        hi_list.extend(chi)
    lo, hi = np.array(lo_list), np.array(hi_list)
    theta = np.clip(theta, lo + 1e-9, hi - 1e-9)

    rng = np.random.default_rng(config.seed)
    chol = _proposal_cholesky(map_fit.covariance, ndim, theta, rng)

    logp = _log_posterior(theta, comps, slices, signal, x, noise, lo, hi, config)
    samples: list[np.ndarray] = []
    for i in range(_N_STEPS):
        prop = theta + chol @ rng.standard_normal(ndim)
        logp_prop = _log_posterior(prop, comps, slices, signal, x, noise, lo, hi, config)
        if np.log(rng.random()) < logp_prop - logp:
            theta, logp = prop, logp_prop
        if i >= _BURN_IN and (i - _BURN_IN) % _THIN == 0:
            samples.append(theta.copy())
    return np.array(samples) if samples else None


@fit_strategy_registry.register("bayesian", description="Seeded posterior sampling around the MAP (Tier 3).")
def fit(profile: np.ndarray, config: FitConfig, *, lane_index: int, label: str) -> LaneFitResult:
    profile, baseline, signal, x, noise, models, kwargs = prepare(profile, config)
    seed_model = models.get("gaussian") or next(iter(models.values()))
    best = _greedy_grow(signal, x, config, seed_model, noise, kwargs)
    map_fit = refine(signal, x, best.components, config, models, noise, kwargs)

    samples = _sample(map_fit, signal, x, noise, config) if map_fit.components else None
    if samples is None:
        return assemble_result(
            profile, baseline, map_fit, config, lane_index=lane_index, label=label, method="bayesian"
        )

    mean = samples.mean(axis=0)
    post_cov = np.cov(samples, rowvar=False)
    if post_cov.ndim == 0:  # single parameter
        post_cov = post_cov.reshape(1, 1)
    comps = [
        Component(c.model, tuple(float(v) for v in mean[a:b]), c.seed_center, c.seed_scale)
        for c, (a, b) in zip(map_fit.components, map_fit.slices)
    ]
    residual = evaluate_composite(comps, x) - signal
    posterior_fit = FitResult(
        components=comps,
        covariance=post_cov,
        slices=map_fit.slices,
        sse=float(np.sum(residual**2)),
        noise_sigma=noise,
        residual=residual,
        at_bound=map_fit.at_bound,
        success=True,
        n_params=mean.size,
    )
    return assemble_result(
        profile, baseline, posterior_fit, config, lane_index=lane_index, label=label, method="bayesian"
    )
