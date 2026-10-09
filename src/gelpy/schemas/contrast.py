"""Contrast-adjustment configuration (display only — never affects analysis)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .common import CanonicalModel


class GammaCorrection(CanonicalModel):
    """Non-linear (gamma) contrast adjustment, applied for rendered images only."""

    gamma: float = Field(default=0.1, gt=0, description="Gamma exponent.")
    gain: float = Field(default=1.0, gt=0, description="Multiplicative gain after gamma.")


class IntensityRange(CanonicalModel):
    """Linear rescale bounds.

    ``mode="percentile"`` interprets ``low``/``high`` as fractions in [0, 1]
    (the legacy float behavior); ``mode="absolute"`` treats them as raw intensity
    values (the legacy int behavior).
    """

    low: float
    high: float
    mode: Literal["percentile", "absolute"] = "percentile"

    @model_validator(mode="after")
    def _check_bounds(self) -> "IntensityRange":
        if self.high <= self.low:
            raise ValueError("IntensityRange.high must be greater than low")
        if self.mode == "percentile" and not (0.0 <= self.low and self.high <= 1.0):
            raise ValueError("percentile bounds must lie in [0, 1]")
        return self


class ContrastConfig(CanonicalModel):
    """How the gel image is displayed. Purely cosmetic: analysis runs on the raw
    (optionally background-corrected) image, not the contrast-adjusted one."""

    gamma: GammaCorrection = Field(default_factory=GammaCorrection)
    # Default linear range: 0th percentile (true image minimum — never crush the dark
    # end) to the 99.99th percentile (at most 0.01% of pixels saturate to white).
    intensity_range: IntensityRange | None = Field(
        default_factory=lambda: IntensityRange(low=0.0, high=0.9999, mode="percentile")
    )
