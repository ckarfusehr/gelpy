"""Shared post-processing for every fit strategy: baseline -> true-area accounting.

Turns a set of fitted components + a baseline into a ``LaneFitResult``: absolute and
fractional areas on the *true* signal basis (``A_sig = integral(profile - baseline)``), a
residual fraction, per-band uncertainty / class / identifiability, the hybrid-smear
region-integral fallback, and the quality + confidence vector. Tiers differ only in how
they arrive at the components; assembly is identical.
"""

from __future__ import annotations

import numpy as np

from ....schemas.baseline import BaselineConfig, BaselineFit
from ....schemas.fit_config import FitConfig
from ....schemas.peaks import BandFit, LaneFitResult
from ..baseline import estimate_baseline
from ..classify import classify_components
from ..driver import Component, FitResult, evaluate_composite, fit_components
from ..quality import assess_confidence, compute_quality

_SMEAR_R2 = 0.75  # below this local R^2, a smear component falls back to region integration
_OVERLAP = 0.5  # pairwise overlap above which two components are flagged non-identifiable
_EMPTY_SNR = 3.0  # A_sig below this * noise * sqrt(n) => the lane is treated as empty


def compute_baseline(profile: np.ndarray, config: BaselineConfig) -> np.ndarray:
    return estimate_baseline(profile, config)


def _area_uncertainty(comp: Component, cov, sl) -> float | None:
    if cov is None:
        return None
    a, b = sl
    block = np.asarray(cov)[a:b, a:b]
    p = np.array(comp.params, dtype=np.float64)
    base = comp.model.area(*p)
    grad = np.zeros_like(p)
    for i in range(p.size):
        dp = 1e-6 * max(abs(p[i]), 1.0)
        pp = p.copy()
        pp[i] += dp
        grad[i] = (comp.model.area(*pp) - base) / dp
    var = float(grad @ block @ grad)
    return float(np.sqrt(var)) if var > 0 else 0.0


def _identifiability(components: list[Component]) -> list[bool]:
    n = len(components)
    flags = [False] * n
    centers = [c.model.peak_maximum_x(*c.params) for c in components]
    widths = [max(c.model.effective_width(*c.params), 1e-6) for c in components]
    for i in range(n):
        for j in range(i + 1, n):
            d2 = (centers[i] - centers[j]) ** 2
            overlap = np.exp(-d2 / (2.0 * (widths[i] ** 2 + widths[j] ** 2)))
            if overlap > _OVERLAP:
                flags[i] = flags[j] = True
    return flags


def _shape_r2(comp: Component, local_signal: np.ndarray, window: slice, x: np.ndarray) -> float | None:
    seg = local_signal[window]
    model_seg = comp.evaluate(x[window])
    ss_tot = float(np.sum((seg - seg.mean()) ** 2))
    if ss_tot <= 0:
        return None
    ss_res = float(np.sum((seg - model_seg) ** 2))
    return 1.0 - ss_res / ss_tot


def assemble_result(
    profile: np.ndarray,
    baseline: np.ndarray,
    fit: FitResult,
    config: FitConfig,
    *,
    lane_index: int,
    label: str,
    method: str,
) -> LaneFitResult:
    profile = np.asarray(profile, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    signal = profile - baseline
    n = signal.size
    x = np.arange(n, dtype=np.float64)
    a_sig = float(np.trapezoid(signal))
    baseline_fit = BaselineFit(
        method=config.baseline.method, lam=config.baseline.lam, mean_level=float(baseline.mean())
    )

    empty_threshold = _EMPTY_SNR * fit.noise_sigma * np.sqrt(n)
    if not fit.components or a_sig <= empty_threshold:
        return LaneFitResult(
            lane_index=lane_index, label=label, status="empty", method=method,
            baseline=baseline_fit, signal_area=a_sig, coverage=0.0, residual_fraction=1.0,
            quality=compute_quality(signal, signal, 0, fit.noise_sigma, 0),
            confidence="low", confidence_reasons=["no significant signal detected"],
        )

    classes = classify_components(fit.components, n, config.baseline.well_frac)
    ident = _identifiability(fit.components)
    denom = a_sig if a_sig > 1e-12 else 1.0

    bands: list[BandFit] = []
    for i, comp in enumerate(fit.components):
        others = evaluate_composite([c for k, c in enumerate(fit.components) if k != i], x)
        local = signal - others
        ctr = comp.model.peak_maximum_x(*comp.params)
        width = comp.model.effective_width(*comp.params)
        lo = max(0, int(ctr - 3 * width))
        hi = min(n, int(ctr + 3 * width) + 1)
        window = slice(lo, hi)
        r2 = _shape_r2(comp, local, window, x)

        params = comp.model.build_params(comp.params)
        region = None
        area = comp.area()
        area_unc = _area_uncertainty(comp, fit.covariance, fit.slices[i])
        # Hybrid smear: a broad component that fits its region poorly is integrated directly.
        if classes[i] == "smear" and (r2 is None or r2 < _SMEAR_R2):
            excess = np.clip(local[window], 0.0, None)
            area = float(np.trapezoid(excess))
            region = (lo, hi)
            params = None
            area_unc = None
            r2 = None
            ctr = float(x[window][int(np.argmax(excess))]) if excess.size else ctr

        comp_curve = comp.evaluate(x)
        csum = float(np.sum(comp_curve))
        centroid = float(np.sum(x * comp_curve) / csum) if csum > 0 else ctr
        bands.append(
            BandFit(
                band_number=i,
                component_class=classes[i],
                area=area,
                area_fraction=area / denom,
                area_uncertainty=area_unc,
                maximum_position=float(ctr),
                centroid=centroid,
                params=params,
                region=region,
                shape_r2=r2,
                at_bound=fit.at_bound[i] if i < len(fit.at_bound) else False,
                low_identifiability=ident[i],
            )
        )

    bands.sort(key=lambda b: b.maximum_position)
    for k, b in enumerate(bands):
        b.band_number = k
    total_area = float(sum(b.area for b in bands))
    coverage = total_area / denom
    quality = compute_quality(signal, fit.residual, fit.n_params, fit.noise_sigma, len(bands))
    confidence, reasons = assess_confidence(quality, coverage, 1.0 - coverage, bands)
    return LaneFitResult(
        lane_index=lane_index, label=label, status="ok", method=method, bands=bands,
        baseline=baseline_fit, signal_area=a_sig, coverage=coverage,
        residual_fraction=1.0 - coverage, quality=quality,
        confidence=confidence, confidence_reasons=reasons,
    )


def refit_and_assemble(
    profile: np.ndarray,
    baseline: np.ndarray,
    components: list[Component],
    config: FitConfig,
    *,
    lane_index: int,
    label: str,
    method: str,
) -> LaneFitResult:
    """Joint-refit ``components`` against the baseline-subtracted signal, then assemble."""
    signal = profile - baseline
    fit = fit_components(
        signal, components,
        robust_loss=config.robust_loss,
        width_prior_strength=config.priors.width_prior_strength,
        tail_shrinkage=config.priors.tail_shrinkage,
        max_width_frac=config.priors.max_width_frac,
    )
    return assemble_result(
        profile, baseline, fit, config, lane_index=lane_index, label=label, method=method
    )
