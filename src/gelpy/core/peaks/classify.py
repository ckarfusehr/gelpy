"""Label each fitted component: sample / well / smear / dust / unknown.

Labels are for interpretation and yield reporting only — every component stays in the
signal total (the user's inclusive rule: never drop signal). Classification is by position
(the well sits at the migration origin) and effective width. Width thresholds are primarily
*absolute* (dust is a few px, a smear is tens of px — physical scales of a gel band), because
median-relative thresholds fail exactly when the dust or smear dominates the lane. A relative
smear test is kept as an OR for lanes whose real bands are unusually wide.
"""

from __future__ import annotations

import numpy as np

from .base import SQRT_2PI
from .driver import Component

# Thresholds (tunable; calibrated on the harness). Absolute values are in px.
_DUST_ABS = 3.5  # a component this narrow (sigma-equivalent) is dust-like
_SMEAR_ABS = 18.0  # a component this wide is a smear
_SMEAR_REL = 3.0  # ... or this many times the median band width (for unusually wide bands)


def _equivalent_sigma(comp: Component) -> float:
    """Shape-agnostic width: the sigma of a Gaussian with this component's area and height.

    Robust to which shape the fitter chose (a broad smear fit as a pseudo-Voigt reports a
    small ``sigma`` param but still has a large area/height ratio)."""
    mx = comp.model.peak_maximum_x(*comp.params)
    height = float(comp.evaluate(np.array([mx]))[0])
    if height <= 0:
        return 0.0
    return comp.area() / (height * SQRT_2PI)


def classify_components(
    components: list[Component], length: int, well_frac: float = 0.08
) -> list[str]:
    if not components:
        return []
    widths = np.array([_equivalent_sigma(c) for c in components], dtype=np.float64)
    centers = np.array([c.model.peak_maximum_x(*c.params) for c in components], dtype=np.float64)
    median_w = float(np.median(widths))
    well_edge = well_frac * length

    classes: list[str] = []
    for w, ctr in zip(widths, centers):
        if ctr <= well_edge:
            classes.append("well")
        # Smear if absolutely wide OR wide relative to this lane's typical band (OR, not max:
        # when the smear dominates the median, the relative test alone would never fire).
        elif w >= _SMEAR_ABS or w >= _SMEAR_REL * median_w:
            classes.append("smear")
        elif w <= _DUST_ABS:
            classes.append("dust")
        else:
            classes.append("sample")
    return classes
