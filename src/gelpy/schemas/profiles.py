"""Line-profile post-processing configuration.

The cached line profile is the full, unsliced column-mean intensity (keyed only by
image/background + lane position + width). Slicing and normalization are cheap
transforms applied *after* the cached profile is loaded, so they live here and feed
the fit hash rather than the profile hash.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .common import CanonicalModel

NormalizationType = Literal["area", "min_max", "safe"]


class LineProfileConfig(CanonicalModel):
    """How to turn a cached raw profile into a fit-ready profile."""

    normalization: NormalizationType = Field(
        default="area", description="Normalization applied before fitting."
    )
    slice_start: int = Field(default=0, ge=0, description="Inclusive start index of the slice.")
    slice_end: int | None = Field(
        default=None, description="Exclusive end index; None means to the end of the profile."
    )

    @model_validator(mode="after")
    def _check_slice(self) -> "LineProfileConfig":
        if self.slice_end is not None and self.slice_end <= self.slice_start:
            raise ValueError("slice_end must be greater than slice_start")
        return self
