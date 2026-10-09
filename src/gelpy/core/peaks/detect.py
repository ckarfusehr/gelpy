"""Peak seeding for the grow-and-select engine.

One entry point: :func:`strongest_bump`, the workhorse the greedy loop calls each round on
the current residual. It returns the tallest coherent positive bump (argmax of a lightly
smoothed residual), scale-free, with its width estimated locally (FWHM). Because it works on
the *residual*, overlapping peaks and shoulders separate out over iterations: fit the tall
one, and its neighbor becomes the next argmax.

An up-front second-derivative shoulder seeder (``ridge_candidates`` /
``second_derivative_centers``) was implemented alongside this and removed in the 1.0 release:
no strategy ever called it, and residual-argmax growth already resolves shoulders. The design
rationale is preserved in docs/decisions/; recover the code from git history if a strategy
ever wants an up-front candidate list.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
_SEED_SMOOTH = 1.5  # px; light smoothing so a single noisy sample can't be a seed


def estimate_scale_fwhm(signal: np.ndarray, position: int, height: float) -> float:
    """Half-max full width of the bump at ``position`` -> Gaussian-sigma-equivalent scale."""
    length = signal.size
    half = height / 2.0
    left = position
    while left > 0 and signal[left] > half:
        left -= 1
    right = position
    while right < length - 1 and signal[right] > half:
        right += 1
    fwhm = max(right - left, 1)
    sigma = fwhm * _FWHM_TO_SIGMA
    return float(min(max(sigma, 1.0), 0.5 * length))


def strongest_bump(
    residual: np.ndarray, noise_sigma: float, smooth: float = _SEED_SMOOTH
) -> tuple[int, float, float, float] | None:
    """Tallest positive bump in ``residual`` as ``(position, scale, height, snr)`` (or None)."""
    resid = np.asarray(residual, dtype=np.float64)
    smoothed = gaussian_filter1d(resid, smooth)
    positive = np.clip(smoothed, 0.0, None)
    if not np.any(positive > 0):
        return None
    position = int(np.argmax(positive))
    height = float(positive[position])
    scale = estimate_scale_fwhm(positive, position, height)
    snr = height / noise_sigma if noise_sigma > 0 else np.inf
    return position, scale, height, float(snr)
