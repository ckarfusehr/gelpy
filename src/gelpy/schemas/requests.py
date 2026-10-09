"""Request models — the single argument of every ``runfolder.commands`` function.

Every command is ``f(request: SomeRequest) -> SomeResult`` so the CLI stays thin and
a future MCP server is a near-zero-rework wrapper (``mcp.tool()(commands.refit_lane)``).
These request models are therefore the top-level tool arguments an agent sees, so
their field descriptions matter: keep them populated.

The fit workflow uses two distinct commands rather than one overloaded one:

- ``FitRunRequest`` — a full fit config applied to an explicit lane selection
  (the "broad pass").
- ``RefitLaneRequest`` — one lane plus all-optional overrides merged onto that
  lane's last-used settings (the "single-lane tweak").
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from .common import CanonicalModel
from .baseline import BaselineConfig
from .fit_config import FitConfig, PriorConfig
from .lanes import LANE_SELECTOR_DESCRIPTION, LANE_TERM_DESCRIPTION, LaneSelector, LaneSelectorTerm
from .profiles import LineProfileConfig, NormalizationType


class SetImageRequest(CanonicalModel):
    """Re-interpret the source image of an existing run (polarity/channel/page)."""

    run_dir: str = Field(description="Existing run folder.")
    channel: str | None = Field(default=None, description="New channel strategy; None keeps current.")
    invert: bool | None = Field(default=None, description="Force polarity; None keeps current unless --auto.")
    page: int | None = Field(default=None, ge=0, description="New page index; None keeps current.")
    auto: bool = Field(default=False, description="Re-run auto-detection for polarity.")


class InitRunRequest(CanonicalModel):
    """Create a run folder with a default lane layout (refine later with set-lanes)."""

    run_dir: str = Field(description="Run folder to create (holds recipe.json, cache, results).")
    image_path: str = Field(description="Path to the source gel image; referenced by path+hash.")
    n_lanes: int = Field(default=10, gt=0, description="Number of lanes for the default layout.")
    width: int | None = Field(
        default=None, gt=0, description="Profile strip width (px); guessed from n_lanes if omitted."
    )
    labels: list[str] | None = Field(
        default=None, description="Lane labels (length must equal n_lanes); defaults to '1'..'N'."
    )
    label_rotation: float = Field(default=90.0, description="Label rotation (deg) in plots.")
    channel: str = Field(
        default="auto",
        description="Channel->grayscale strategy: auto|mean|luminance|r|g|b|dominant|max.",
    )
    invert: bool | None = Field(
        default=None,
        description="Force polarity: True=dark-on-light, False=bright-on-dark, None=auto-detect.",
    )
    page: int = Field(default=0, ge=0, description="Page index for a multi-page source.")


class SetLanesRequest(CanonicalModel):
    """Set lane positions / labels / width. Changing positions or width auto-invalidates
    downstream caches by hash — no explicit cache clearing needed."""

    run_dir: str = Field(description="Existing run folder.")
    linspace: tuple[float, float, int] | None = Field(
        default=None, description="(left, right, n) evenly spaced lane centers in px."
    )
    x_centers: list[float] | None = Field(
        default=None, description="Explicit lane center x-positions in px (alternative to linspace)."
    )
    endpoints: tuple[float, float] | None = Field(
        default=None,
        description="(left, right) outer lane-center x-positions in px; the N lane centers are "
        "linspaced between them. N comes from `n_lanes` or the current lane count. This is the "
        "manual fallback when `lanes detect` is uncertain — read the two outer centers off the "
        "detection overlay.",
    )
    n_lanes: int | None = Field(
        default=None, gt=0, description="Lane count for `endpoints` (defaults to the current count)."
    )
    labels: list[str] | None = Field(
        default=None, description="Lane labels; must match the number of positions."
    )
    width: int | None = Field(default=None, gt=0, description="Profile strip width (px).")
    label_rotation: float | None = Field(default=None, description="Label rotation (deg) in plots.")


class SetContrastRequest(CanonicalModel):
    """Adjust display contrast (cosmetic; never affects analysis)."""

    run_dir: str = Field(description="Existing run folder.")
    gamma: float | None = Field(default=None, gt=0, description="Gamma-correction exponent.")
    gain: float | None = Field(default=None, description="Gamma-correction gain.")
    intensity_low: float | None = Field(default=None, description="Lower intensity clip bound.")
    intensity_high: float | None = Field(default=None, description="Upper intensity clip bound.")
    intensity_mode: str | None = Field(
        default=None, description="How intensity bounds are interpreted: 'percentile' or 'absolute'."
    )


class RemoveBackgroundRequest(CanonicalModel):
    """Configure background removal (applied lazily by the pipeline, not eagerly)."""

    run_dir: str = Field(description="Existing run folder.")
    model: str = Field(default="robust_plane", description="Background model name (see `models list`).")
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Model-specific parameters, validated against the model's schema at apply time.",
    )
    stripes: list[tuple[int, int]] | None = Field(
        default=None,
        description="plane_ransac-only shim: (position, height) background stripes. Prefer `params`.",
    )


class ExtractProfilesRequest(CanonicalModel):
    """Extract (and cache) the raw line profiles for a lane selection."""

    run_dir: str = Field(description="Existing run folder.")
    lanes: LaneSelector = Field(default="all", description=LANE_SELECTOR_DESCRIPTION)


class GetResultsRequest(CanonicalModel):
    """Assemble results from cache without recomputing anything."""

    run_dir: str = Field(description="Existing run folder.")


class StatusRequest(CanonicalModel):
    """Report the current state of a run folder (configured / fit / stale lanes)."""

    run_dir: str = Field(description="Existing run folder.")


class DetectLanesRequest(CanonicalModel):
    """Detect lane centers from the image by fitting an even N-lane grid to the y-projection.

    ``n_lanes`` is ground truth (from the plate sheet / user) and is never inferred; detection
    only anchors that many evenly-spaced centers to the actual bright lane columns, so the layout
    is correct regardless of how the gel was cropped. Always inspect the verification overlay
    before relying on the result; if ``confidence`` is ``"low"``, adjust the knobs below or fall
    back to ``set_lanes`` with explicit ``endpoints``.
    """

    run_dir: str = Field(description="Existing run folder.")
    n_lanes: int | None = Field(
        default=None, gt=0, description="True lane count (defaults to the current recipe count)."
    )
    y_start: int | None = Field(
        default=None, ge=0, description="Top row of the projection band (default: image top)."
    )
    y_end: int | None = Field(
        default=None,
        ge=0,
        description="Bottom row of the projection band (default: image bottom). Restricting to the "
        "well region near the top often separates faint lanes more cleanly.",
    )
    smoothing_window: float | None = Field(
        default=None,
        ge=0,
        description="Moving-window average width (px) applied to the projection before peak "
        "detection (default: 0.7 x the expected pitch, width/n_lanes). Wells are plateaus, "
        "often brighter on one edge than the other — averaging over a considerable width "
        "prevents that from biasing the detected center.",
    )
    min_prominence: float | None = Field(
        default=None,
        description="Peak prominence floor (fraction of the normalized projection range; default "
        "0.05). Raise it to drop spurious peaks from dust/artifacts.",
    )
    min_distance: float | None = Field(
        default=None, gt=0, description="Minimum px between detected peaks (default: smoothing-scaled)."
    )
    min_width: float | None = Field(
        default=None,
        gt=0,
        description="Minimum peak width in px (default: 1.0, i.e. off). On real gels a genuine "
        "(faint) band can be as narrow as a dust/artifact spike, so there's no universally safe "
        "scaled default — compare peak widths in the overlay first, then raise this above the "
        "spurious ones but below the genuine ones.",
    )
    use_corrected: bool = Field(
        default=False,
        description="Detect on the background-corrected image instead of the raw one. Default is "
        "raw: fluorescence gels already have a dark background with bright lanes, and "
        "plane-subtraction (which clips negatives) tends to add projection noise. Enable this for "
        "gels with strong uneven illumination where the raw background swamps the lanes.",
    )
    projection_strategy: str = Field(
        default="d4_union",
        description="How the 2-D image is collapsed to the 1-D projection the grid is fit to. "
        "'d4_union' (default) clusters informative y-rows and unions the bands that add lane "
        "coverage, falling back to 'full' (the full-height column mean) whenever it doesn't beat "
        "that baseline, so it is never worse than 'full'.",
    )
    apply: bool = Field(
        default=False, description="Persist the detected centers to the recipe (like `set_lanes`)."
    )
    labels: list[str] | None = Field(
        default=None, description="Lane labels to apply (only when `apply`); defaults to existing."
    )
    label_rotation: float | None = Field(default=None, description="Label rotation (deg) in plots.")


class FitRunRequest(CanonicalModel):
    """Run a full fit pass over a lane selection."""

    run_dir: str = Field(description="Existing run folder.")
    lanes: LaneSelector = Field(default="all", description=LANE_SELECTOR_DESCRIPTION)
    profile: LineProfileConfig = Field(
        default_factory=LineProfileConfig, description="Slice + normalization applied before fitting."
    )
    fit: FitConfig = Field(
        default_factory=FitConfig, description="Fit strategy, baseline, allowed shapes, and priors."
    )
    auto_escalate: bool = Field(
        default=False,
        description="Climb the strategy ladder (grow_select -> sparse_seed -> bayesian) on any lane "
        "whose fit confidence is 'low', up to the max tier.",
    )
    plot: bool = Field(default=False, description="Also render per-lane fit overlay PNGs.")


class FitConfigOverrides(CanonicalModel):
    """All-optional overrides merged onto a lane's last-used ``FitConfig``."""

    method: str | None = Field(default=None, description="Fit strategy name.")
    baseline: BaselineConfig | None = Field(default=None, description="Replace the baseline config.")
    shapes: list[str] | None = Field(default=None, description="Replace the allowed shape menu.")
    priors: PriorConfig | None = Field(default=None, description="Replace the priors/stopping config.")
    robust_loss: str | None = Field(default=None, description="Robust loss for least_squares.")
    seed: int | None = Field(default=None, ge=0, description="RNG seed.")


class LineProfileConfigOverrides(CanonicalModel):
    """All-optional overrides merged onto a lane's last-used ``LineProfileConfig``."""

    normalization: NormalizationType | None = Field(
        default=None, description="Profile normalization: 'area', 'min_max', or 'safe'."
    )
    slice_start: int | None = Field(default=None, ge=0, description="Profile slice start index.")
    slice_end: int | None = Field(default=None, description="Profile slice end index.")


class RefitLaneRequest(CanonicalModel):
    """Refit a single lane, changing only the fields explicitly provided."""

    run_dir: str = Field(description="Existing run folder.")
    lane: LaneSelectorTerm = Field(
        description="The single lane to refit. " + LANE_TERM_DESCRIPTION
    )
    fit_overrides: FitConfigOverrides = Field(
        default_factory=FitConfigOverrides,
        description="Fit-config fields to change; unset fields keep the lane's last-used values.",
    )
    profile_overrides: LineProfileConfigOverrides = Field(
        default_factory=LineProfileConfigOverrides,
        description="Profile-config fields to change; unset fields keep the lane's last-used values.",
    )
    auto_escalate: bool = Field(
        default=False, description="Climb the strategy ladder if this lane's confidence stays 'low'."
    )
    plot: bool = Field(default=False, description="Also render the lane's fit overlay PNG.")
