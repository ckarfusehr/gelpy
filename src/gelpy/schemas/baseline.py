"""Per-lane baseline configuration and the recorded baseline fit.

This is the 1-D, fit-time baseline (residual drift/curvature under the peaks), distinct
from the 2-D background-removal stage. It is deliberately *conservative*: smears are
signal, so the baseline models only true low-frequency background, never a broad band.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .common import CanonicalModel

BaselineMethod = Literal["arpls", "asls", "poly", "none"]


class BaselineConfig(CanonicalModel):
    """How to estimate the 1-D baseline before/with peak fitting.

    Default is a low-order iterative **polynomial**: it tracks smooth drift while bending
    only weakly to a localized smear, so smears mostly stay in the signal. Measured
    absorption of a genuine broad smear on a flat residual background: ``none`` 0.0%,
    ``poly`` order 0 2.1%, order 1 2.2%, order 2 (default) 5.5%, order 3 13.1%, ``arpls``
    34.4%, ``asls`` 58.5%. arPLS/AsLS are available for gels with genuinely wiggly
    (non-polynomial) backgrounds, at that cost.
    """

    method: BaselineMethod = Field(default="poly", description="Baseline estimator.")
    lam: float = Field(
        default=1e5,
        gt=0,
        description="Whittaker smoothness (arpls/asls). Larger = stiffer, flatter baseline "
        "that cannot bend to follow a smear.",
    )
    p: float = Field(
        default=0.01,
        gt=0,
        lt=1,
        description="Asymmetry weight for points above the baseline (asls only).",
    )
    poly_order: int = Field(
        default=2,
        ge=0,
        description="Polynomial order for method='poly' (the default). 2 tracks "
        "constant/linear/mild-quadratic drift while absorbing ~5.5% of a broad smear; order 3 "
        "absorbs ~13%, order 0 (constant) ~2%. Use 0 when the window starts past a hard "
        "exclusion boundary (e.g. a well-cavity dip), where no anchoring feature is left to "
        "hold a higher-order baseline down.",
    )
    protect_well: bool = Field(
        default=True,
        description="Down-weight the near-origin well region so the baseline does not absorb it.",
    )
    well_frac: float = Field(
        default=0.08,
        ge=0,
        lt=1,
        description="Fraction of the window (from the origin) treated as the well region.",
    )


class BaselineFit(CanonicalModel):
    """Recorded baseline outcome (for reporting; values are recomputed deterministically)."""

    method: BaselineMethod
    lam: float | None = None
    poly_coeffs: list[float] | None = None
    mean_level: float | None = Field(
        default=None, description="Mean baseline level over the window (diagnostic)."
    )
