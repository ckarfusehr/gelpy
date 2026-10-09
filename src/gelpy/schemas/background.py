"""Background-removal stage configuration."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from .common import CanonicalModel


class PlaneRansacParams(CanonicalModel):
    """Parameters for the ``plane_ransac`` background model.

    ``stripes`` is a list of ``(position, height)`` pairs marking horizontal,
    background-only regions of the image used to fit the plane. ``None`` selects a
    heuristic default (a thin stripe near the top and near the bottom).
    """

    stripes: list[tuple[int, int]] | None = Field(
        default=None,
        description="(start_row, height) background stripes; None = heuristic default.",
    )


class RobustPlaneParams(CanonicalModel):
    """Parameters for the ``robust_plane`` background model.

    An iterative-reweighted least-squares fit of a low-order polynomial surface to the
    *background-only* pixels. It needs no user-placed stripes: the physical fact that
    sample signal (bands, smears) only ever *adds* intensity means background lives on
    the low (sub-surface) side of the residual, and the reweighting keys on that side.

    Zero-config is the intended path — every field has a sensible default.
    """

    order: int = Field(
        default=1,
        ge=0,
        le=2,
        description="Polynomial surface order: 0=offset, 1=plane (default), 2=quadratic "
        "(captures vignetting/curvature; opt-in — can bend into diffuse haze).",
    )
    weight_mode: Literal["probabilistic", "asymmetric"] = Field(
        default="probabilistic",
        description="'probabilistic' (default): two-component mixture, σ self-scaled to the "
        "noise floor — robust when signal covers a large image fraction. 'asymmetric': "
        "fixed Eilers ALS weight (simpler, no σ estimation).",
    )
    asymmetry_p: float = Field(
        default=0.02,
        gt=0.0,
        lt=1.0,
        description="Weight given to above-surface (signal) pixels in 'asymmetric' mode.",
    )
    signal_prior: float = Field(
        default=0.3,
        ge=0.0,
        lt=1.0,
        description="Prior fraction of signal pixels for the 'probabilistic' mixture.",
    )
    max_iter: int = Field(default=15, ge=1, le=100, description="Max IRLS iterations.")
    tol: float = Field(
        default=1e-3, gt=0.0, description="Convergence tol on max weight change between iterations."
    )
    subsample: int = Field(
        default=1,
        ge=1,
        description="Fit the surface on every n-th pixel (lossless for a smooth surface); "
        "evaluated on the full grid. Speed escape hatch for very large images.",
    )
    contiguity: float = Field(
        default=0.0,
        ge=0.0,
        description="Spatial-smoothing sigma (px) applied to the weight field each iteration; "
        "0 disables. Makes weights coherent so isolated hot pixels aren't trusted as background.",
    )
    clip_negative: bool = Field(
        default=False,
        description="If True, clamp negatives to 0 (legacy behaviour). Default False keeps signed "
        "residuals so the baseline stays centered at 0 — clamping rectifies noise into a "
        "positive pedestal that biases downstream peak integrals.",
    )


class BackgroundStageConfig(CanonicalModel):
    """Selects and configures a background model.

    ``model`` is validated against the background registry (``core.background``);
    ``params`` is validated against that model's parameter schema at apply time.
    """

    model: str = Field(default="robust_plane", description="Registered background-model name.")
    params: dict[str, Any] = Field(
        default_factory=dict, description="Model-specific parameters (schema per model)."
    )
