"""Peak-shape parameters, per-band fits, fit-quality, and per-lane fit results.

No numpy/scipy here — this stays importable standalone. Every shape is parameterized
so that its **area has a closed form** and shape-selection is continuous: an EMG with
``tau -> 0`` is exactly a Gaussian of the same ``amplitude``/``center``/``sigma``, so the
tail-shrinkage prior degrades gracefully to a symmetric peak.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import Field

from .baseline import BaselineFit
from .common import CanonicalModel

ShapeKind = Literal["gaussian", "emg", "bi_gaussian", "pseudo_voigt"]
ComponentClass = Literal["sample", "well", "smear", "dust", "unknown"]


# --- per-shape fitted parameters (discriminated by ``kind``) ------------------
class GaussianPeakParams(CanonicalModel):
    """Symmetric Gaussian. area = amplitude * sigma * sqrt(2*pi)."""

    kind: Literal["gaussian"] = "gaussian"
    amplitude: float = Field(description="Peak height.")
    center: float = Field(description="Center position (px); also the maximum.")
    sigma: float = Field(gt=0, description="Standard deviation (px).")


class EmgPeakParams(CanonicalModel):
    """Exponentially-modified Gaussian (Gaussian convolved with a one-sided exponential).

    Parameterized so ``area = amplitude * sigma * sqrt(2*pi)`` for all ``tau`` and the
    ``tau -> 0`` limit is a Gaussian of height ``amplitude``. ``tau > 0`` tails toward
    larger x (migration direction); the maximum sits left of ``center``.
    """

    kind: Literal["emg"] = "emg"
    amplitude: float = Field(description="Gaussian-component height scale.")
    center: float = Field(description="Gaussian-component mean (px); NOT the maximum.")
    sigma: float = Field(gt=0, description="Gaussian-component standard deviation (px).")
    tau: float = Field(
        description="Exponential tail time-constant (px); sign = direction (>0 tails toward "
        "larger x, <0 toward smaller x), 0 = pure Gaussian."
    )


class BiGaussianPeakParams(CanonicalModel):
    """Split (bi-)Gaussian with independent left/right widths — cheap asymmetry.

    area = amplitude * sqrt(pi/2) * (sigma_left + sigma_right).
    """

    kind: Literal["bi_gaussian"] = "bi_gaussian"
    amplitude: float = Field(description="Peak height.")
    center: float = Field(description="Center position (px); also the maximum.")
    sigma_left: float = Field(gt=0, description="Std dev on the low-x side (px).")
    sigma_right: float = Field(gt=0, description="Std dev on the high-x side (px).")


class PseudoVoigtPeakParams(CanonicalModel):
    """Linear Gaussian/Lorentzian blend for sharp, heavy-winged peaks (e.g. the well).

    gamma (Lorentzian HWHM) is FWHM-matched to sigma: gamma = sigma * sqrt(2*ln2).
    area = amplitude * ((1-eta) * sigma * sqrt(2*pi) + eta * pi * gamma).
    """

    kind: Literal["pseudo_voigt"] = "pseudo_voigt"
    amplitude: float = Field(description="Peak height.")
    center: float = Field(description="Center position (px); also the maximum.")
    sigma: float = Field(gt=0, description="Gaussian width (px).")
    eta: float = Field(ge=0, le=1, description="Lorentzian fraction (0 = Gaussian, 1 = Lorentzian).")


PeakParams = Annotated[
    Union[GaussianPeakParams, EmgPeakParams, BiGaussianPeakParams, PseudoVoigtPeakParams],
    Field(discriminator="kind"),
]


# --- results ------------------------------------------------------------------
class BandFit(CanonicalModel):
    """One fitted component within a lane. Area is on the *true* signal basis."""

    band_number: int = Field(ge=0)
    component_class: ComponentClass = Field(
        default="sample", description="Semantic label; never excludes the band from the total."
    )
    area: float = Field(description="Absolute integrated area (baseline-subtracted units).")
    area_fraction: float = Field(description="area / total baseline-subtracted signal area A_sig.")
    area_uncertainty: float | None = Field(
        default=None, description="1-sigma on area from the fit covariance (None if unavailable)."
    )
    maximum_position: float = Field(
        description="x-position (px) of the component maximum, in the coordinates of the "
        "profile that was fitted -- i.e. RELATIVE TO slice_start, not to the source image. "
        "With --slice 50 600 a band at image row 89 is reported at 39. The per-lane fit "
        "figure uses the same axis, so the two agree, but positions are only comparable "
        "across runs that used the same slice window: add slice_start to compare against an "
        "unsliced run."
    )
    centroid: float | None = Field(default=None, description="Area-weighted centroid x (px).")
    params: PeakParams | None = Field(
        default=None, description="Fitted shape params; None for a region-integrated smear."
    )
    region: tuple[int, int] | None = Field(
        default=None, description="(lo, hi) px support for a region-integrated component."
    )
    shape_r2: float | None = Field(default=None, description="Local R^2 of this component's fit.")
    at_bound: bool = Field(default=False, description="A parameter hit its bound (suspect fit).")
    low_identifiability: bool = Field(
        default=False, description="Overlaps a neighbor too much to be separately determined."
    )


class FitQuality(CanonicalModel):
    """Scalar goodness-of-fit signals for one lane (drives escalation + the refit loop)."""

    reduced_chi2: float | None = None
    r2: float | None = None
    residual_autocorr: float | None = Field(
        default=None, description="Durbin-Watson (~2 = white residual; <<2 = structured)."
    )
    noise_sigma: float | None = None
    n_components: int = 0
    aicc: float | None = None
    bic: float | None = None


class LaneFitResult(CanonicalModel):
    """Outcome of fitting one lane. A failed/empty lane is preserved, never dropped."""

    lane_index: int = Field(ge=0)
    label: str
    status: Literal["ok", "empty", "failed"] = "ok"
    error: str | None = None
    method: str = "grow_select"
    bands: list[BandFit] = Field(default_factory=list)
    baseline: BaselineFit | None = None
    signal_area: float | None = Field(default=None, description="A_sig = integral of (y - baseline).")
    coverage: float | None = Field(default=None, description="sum(area_k) / A_sig.")
    residual_fraction: float | None = Field(default=None, description="1 - coverage.")
    quality: FitQuality | None = None
    confidence: Literal["ok", "low"] = "ok"
    confidence_reasons: list[str] = Field(default_factory=list)

    def to_rows(self) -> list[dict[str, Any]]:
        """Flatten to result rows (one per band). Failed/empty lanes emit one lane-level row."""
        base: dict[str, Any] = {
            "lane_index": self.lane_index,
            "label": self.label,
            "status": self.status,
            "confidence": self.confidence,
            "method": self.method,
            "residual_fraction": self.residual_fraction,
        }
        if self.status != "ok" or not self.bands:
            return [{**base, "error": self.error}]
        rows: list[dict[str, Any]] = []
        for band in self.bands:
            row: dict[str, Any] = {
                **base,
                "band_number": band.band_number,
                "component_class": band.component_class,
                "area": band.area,
                "area_fraction": band.area_fraction,
                "area_uncertainty": band.area_uncertainty,
                "maximum_position": band.maximum_position,
            }
            if band.params is not None:
                row["shape"] = band.params.kind
                for key, value in band.params.model_dump().items():
                    if key != "kind":
                        row[f"param_{key}"] = value
            else:
                row["shape"] = "region"
            rows.append(row)
        return rows
