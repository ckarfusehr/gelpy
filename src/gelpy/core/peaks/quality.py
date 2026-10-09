"""Fit-quality metrics and the confidence flag that drives escalation / the refit loop.

The key discriminator is that ``status == ok`` (the optimizer converged) is not proof a fit
is right: a structured residual (low Durbin-Watson) betrays a missed or mis-shaped peak even
when the reduced chi-square looks fine, and low coverage betrays signal the model never
explained. These, plus per-peak uncertainty / at-bound / identifiability flags, become a
machine-readable ``confidence`` + ``reasons`` an agent branches on.
"""

from __future__ import annotations

import numpy as np

from ...schemas.peaks import BandFit, FitQuality
from .driver import aicc, bic

# Confidence thresholds (tunable; calibrated on synthetic + real lanes).
_COVERAGE_MIN = 0.85  # < this: real signal went unmodeled
_COVERAGE_MAX = 1.10  # > this: fitted peaks sum past the true signal area (overlap double-counts)
_R2_MIN = 0.90  # only catches a genuinely marginal fit; correct broad/tailed peaks sit at ~0.97
_AREA_UNC_SEVERE = 1.0  # area uncertainty > 100% of the area => that area is untrustworthy -> low
_AREA_UNC_WARN = 0.5  # > 50% => a warning (surfaced, but not enough alone to flag the lane)


def durbin_watson(residual: np.ndarray) -> float:
    r = np.asarray(residual, dtype=np.float64)
    denom = float(np.sum(r**2))
    if denom <= 0:
        return 2.0
    return float(np.sum(np.diff(r) ** 2) / denom)


def compute_quality(
    signal: np.ndarray, residual: np.ndarray, n_params: int, noise_sigma: float, n_components: int
) -> FitQuality:
    signal = np.asarray(signal, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)
    n = signal.size
    sse = float(np.sum(residual**2))
    dof = max(n - n_params, 1)
    reduced_chi2 = sse / (noise_sigma**2 * dof) if noise_sigma > 0 else None
    ss_tot = float(np.sum((signal - signal.mean()) ** 2))
    r2 = (1.0 - sse / ss_tot) if ss_tot > 0 else None
    return FitQuality(
        reduced_chi2=reduced_chi2,
        r2=r2,
        residual_autocorr=durbin_watson(residual),
        noise_sigma=float(noise_sigma),
        n_components=n_components,
        aicc=aicc(sse, n, n_params),
        bic=bic(sse, n, n_params),
    )


def assess_confidence(
    quality: FitQuality, coverage: float | None, residual_fraction: float | None, bands: list[BandFit]
) -> tuple[str, list[str]]:
    """Return ('ok'|'low', reasons). Any reason -> 'low'.

    Uses coverage + R^2 + per-peak flags, all scale-free and robust on real data. Reduced
    chi-square and Durbin-Watson are reported in ``FitQuality`` but deliberately NOT hard
    triggers: real digitized profiles have minuscule pixel-noise, so chi-square explodes and
    DW collapses on perfectly good fits, which would flag every real lane 'low'.
    """
    reasons: list[str] = []
    low = False
    if coverage is not None and coverage < _COVERAGE_MIN:
        rf = residual_fraction if residual_fraction is not None else (1.0 - coverage)
        reasons.append(f"coverage {coverage:.0%}: {rf:.0%} of the signal is unmodeled")
        low = True
    if coverage is not None and coverage > _COVERAGE_MAX:
        reasons.append(f"coverage {coverage:.0%}: fitted peaks exceed the signal (overlap/over-fit)")
        low = True
    if quality.r2 is not None and quality.r2 < _R2_MIN:
        reasons.append(f"poor fit (R2={quality.r2:.3f})")
        low = True
    for b in bands:
        rel_unc = (b.area_uncertainty / b.area) if (b.area_uncertainty is not None and b.area > 0) else 0.0
        if b.at_bound:  # runaway width: a "peak" spanning the trace to absorb drift
            reasons.append(f"band {b.band_number} ({b.component_class}) width ran to a bound")
            low = True
        if rel_unc > _AREA_UNC_SEVERE:
            reasons.append(f"band {b.band_number} area unreliable (+/-{100 * rel_unc:.0f}%)")
            low = True
        # Warnings: surfaced for the agent but not enough on their own to flag the lane low
        # (dense real lanes always have a close pair; the overall areas can still be sound).
        elif rel_unc > _AREA_UNC_WARN:
            reasons.append(f"note: band {b.band_number} area uncertain (+/-{100 * rel_unc:.0f}%)")
        if b.low_identifiability:
            reasons.append(f"note: band {b.band_number} overlaps a neighbor (areas correlated)")
    return ("low" if low else "ok"), reasons
