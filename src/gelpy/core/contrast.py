"""Contrast adjustment (display only)."""

from __future__ import annotations

import numpy as np
from skimage import exposure


def apply_gamma(image: np.ndarray, gamma: float, gain: float) -> np.ndarray:
    """Non-linear gamma correction (``skimage.exposure.adjust_gamma``)."""
    return exposure.adjust_gamma(image, gamma=gamma, gain=gain)


def rescale_linear(
    image: np.ndarray, low: float, high: float, mode: str = "percentile"
) -> np.ndarray:
    """Linear intensity rescale.

    ``mode="percentile"`` interprets ``low``/``high`` as fractions in [0, 1] and maps
    them to the corresponding image percentiles (legacy float behavior).
    ``mode="absolute"`` treats them as raw intensity bounds (legacy int behavior).
    """
    if mode == "percentile":
        q_low, q_high = np.percentile(image, [low * 100, high * 100])
        in_range = (q_low, q_high)
    elif mode == "absolute":
        in_range = (low, high)
    else:
        raise ValueError(f"Unknown rescale mode {mode!r}; expected 'percentile' or 'absolute'.")
    return exposure.rescale_intensity(image, in_range=in_range)
