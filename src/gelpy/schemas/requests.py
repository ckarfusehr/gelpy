"""Request models — the single argument of every ``runfolder.commands`` function.

Every command is ``f(request: SomeRequest) -> SomeResult`` so the CLI stays thin, and
``cli/tools.py`` pairs each request model with the handler of its CLI step, so ``gelpy tool
call`` and the MCP server take these models as they are. These request models are therefore
the top-level tool arguments an agent sees, so their field descriptions matter: keep them
populated.

The fit workflow uses two distinct commands rather than one overloaded one:

- ``FitRunRequest`` — a full fit config applied to an explicit lane selection
  (the "broad pass").
- ``RefitLaneRequest`` — one lane plus all-optional overrides merged onto that
  lane's last-used settings (the "single-lane tweak").
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field

from .common import CanonicalModel
from .baseline import BaselineConfig
from .fit_config import FitConfig, PriorConfig
from .lanes import (
    LANE_SELECTOR_DESCRIPTION,
    LANE_TERM_DESCRIPTION,
    MAX_LANES,
    LaneSelector,
    LaneSelectorTerm,
)
from .profiles import LineProfileConfig, NormalizationType


class RequestModel(CanonicalModel):
    """Base of every request model: a number in a request must be finite.

    A request is typed by a person or an agent, and JSON has no NaN or infinity: stored in
    recipe.json or fit_config.json, such a number reads back as null and the run folder no longer
    loads. Results are not strict: they are computed, and one NaN must not stop one from being
    reported.
    """

    model_config = ConfigDict(allow_inf_nan=False)


class SetImageRequest(RequestModel):
    """Re-interpret the source image of an existing run (polarity/channel/page)."""

    run_dir: str = Field(description="Existing run folder.")
    channel: str | None = Field(default=None, description="New channel strategy; None keeps current.")
    invert: bool | None = Field(default=None, description="Force polarity; None keeps current unless --auto.")
    page: int | None = Field(default=None, ge=0, description="New page index; None keeps current.")
    auto: bool = Field(default=False, description="Re-run auto-detection for polarity.")


#: The four gel views `image show` renders.
ImageKind = Literal["raw", "linear", "non_linear", "corrected"]


class RenderImageRequest(RequestModel):
    """Render the gel image of a run folder to a PNG under render/."""

    run_dir: str = Field(description="Existing run folder.")
    kind: ImageKind = Field(
        default="corrected",
        description="Which image under which display contrast: 'raw' (the canonical array, no "
        "display contrast), 'linear' / 'non_linear' (the source under the recipe's linear range "
        "or gamma), 'corrected' (the background-corrected array under gamma).",
    )
    lanes: bool = Field(
        default=False,
        description="Draw the semi-transparent lane strips over the image: a QC view of where "
        "gelpy measures, never the sample view.",
    )


class InitRunRequest(RequestModel):
    """Create a run folder with a default lane layout.

    Refine the layout later with `lanes set` (tool `lanes_set`).
    """

    run_dir: str = Field(description="Run folder to create (holds recipe.json, cache, results).")
    image_path: str = Field(description="Path to the source gel image; referenced by path+hash.")
    n_lanes: int = Field(
        default=10, gt=0, le=MAX_LANES, description="Number of lanes for the default layout (1 to 1000)."
    )
    width: int | None = Field(
        default=None, gt=0, description="Profile strip width (px); guessed from n_lanes if omitted."
    )
    labels: list[str] | None = Field(
        default=None,
        max_length=MAX_LANES,
        description="Lane labels (length must equal n_lanes, at most 1000); defaults to '1'..'N'.",
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
    force: bool = Field(
        default=False,
        description="Replace the run if the folder already holds one: its recipe and fit "
        "configuration are replaced, its results, report and renders removed; the cache and "
        "the history log are kept. Without it an existing run is refused (run_exists).",
    )


class SetLanesRequest(RequestModel):
    """Set lane positions / labels / width. Changing positions or width auto-invalidates
    downstream caches by hash — no explicit cache clearing needed."""

    run_dir: str = Field(description="Existing run folder.")
    linspace: tuple[float, float, Annotated[int, Field(le=MAX_LANES)]] | None = Field(
        default=None, description="(left, right, n) evenly spaced lane centers in px; n is at most 1000."
    )
    x_centers: list[float] | None = Field(
        default=None,
        max_length=MAX_LANES,
        description="Explicit lane center x-positions in px, at most 1000 (alternative to linspace).",
    )
    endpoints: tuple[float, float] | None = Field(
        default=None,
        description="(left, right) outer lane-center x-positions in px; the N lane centers are "
        "linspaced between them. N comes from `n_lanes` or the current lane count. This is the "
        "manual fallback when `lanes detect` is uncertain — read the two outer centers off the "
        "detection overlay.",
    )
    n_lanes: int | None = Field(
        default=None,
        gt=0,
        le=MAX_LANES,
        description="Lane count for `endpoints`, at most 1000 (defaults to the current count).",
    )
    labels: list[str] | None = Field(
        default=None,
        max_length=MAX_LANES,
        description="Lane labels, at most 1000; must match the number of positions.",
    )
    width: int | None = Field(default=None, gt=0, description="Profile strip width (px).")
    label_rotation: float | None = Field(default=None, description="Label rotation (deg) in plots.")


class SetContrastRequest(RequestModel):
    """Adjust display contrast (cosmetic; never affects analysis)."""

    run_dir: str = Field(description="Existing run folder.")
    gamma: float | None = Field(default=None, gt=0, description="Gamma-correction exponent.")
    gain: float | None = Field(default=None, description="Gamma-correction gain.")
    intensity_low: float | None = Field(default=None, description="Lower intensity clip bound.")
    intensity_high: float | None = Field(default=None, description="Upper intensity clip bound.")
    intensity_mode: str | None = Field(
        default=None, description="How intensity bounds are interpreted: 'percentile' or 'absolute'."
    )


class RemoveBackgroundRequest(RequestModel):
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
    plot: bool = Field(
        default=False,
        description="Also render the background diagnostic figure "
        "(render/background_diagnostic.png).",
    )


class ExtractProfilesRequest(RequestModel):
    """Extract (and cache) the raw line profiles for a lane selection."""

    run_dir: str = Field(description="Existing run folder.")
    lanes: LaneSelector = Field(default="all", description=LANE_SELECTOR_DESCRIPTION)
    plot: bool = Field(default=False, description="Also render the selected profiles overlaid.")
    profile: LineProfileConfig = Field(
        default_factory=LineProfileConfig,
        description="Slice and normalization of the plot only; the cached profiles are always "
        "the full, unnormalized ones.",
    )
    allow_raw: bool = Field(
        default=False,
        description="Measure the raw image although no background removal is configured. Off "
        "by default: profiles and fits of a raw image carry each lane's background floor into "
        "its normalization, so lanes are not comparable (background_required).",
    )


class GetResultsRequest(RequestModel):
    """Assemble results from cache without recomputing anything."""

    run_dir: str = Field(description="Existing run folder.")


class StatusRequest(RequestModel):
    """Report the current state of a run folder (configured / fit / stale lanes)."""

    run_dir: str = Field(description="Existing run folder.")


class DetectLanesRequest(RequestModel):
    """Detect lane centers from the image by fitting an even N-lane grid to the y-projection.

    ``n_lanes`` is ground truth (from the plate sheet / user) and is never inferred; detection
    only anchors that many evenly-spaced centers to the actual bright lane columns, so the layout
    is correct regardless of how the gel was cropped. Always inspect the verification overlay
    before relying on the result; if ``confidence`` is ``"low"``, adjust the knobs below or fall
    back to ``lanes set`` (tool ``lanes_set``) with explicit ``endpoints``.
    """

    run_dir: str = Field(description="Existing run folder.")
    n_lanes: int | None = Field(
        default=None,
        gt=0,
        le=MAX_LANES,
        description="True lane count, at most 1000 (defaults to the current recipe count).",
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
        default=False, description="Persist the detected centers to the recipe (like `lanes_set`)."
    )
    labels: list[str] | None = Field(
        default=None,
        max_length=MAX_LANES,
        description="Lane labels to apply, at most 1000 (only when `apply`); defaults to existing.",
    )
    label_rotation: float | None = Field(default=None, description="Label rotation (deg) in plots.")
    plot: bool = Field(
        default=True,
        description="Also render the verification overlay (render/lane_detect.png).",
    )


class FitRunRequest(RequestModel):
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
    allow_raw: bool = Field(
        default=False,
        description="Measure the raw image although no background removal is configured. Off "
        "by default: profiles and fits of a raw image carry each lane's background floor into "
        "its normalization, so lanes are not comparable (background_required).",
    )


class FitConfigOverrides(RequestModel):
    """All-optional overrides merged onto a lane's last-used ``FitConfig``."""

    method: str | None = Field(default=None, description="Fit strategy name.")
    baseline: BaselineConfig | None = Field(default=None, description="Replace the baseline config.")
    shapes: list[str] | None = Field(default=None, description="Replace the allowed shape menu.")
    priors: PriorConfig | None = Field(default=None, description="Replace the priors/stopping config.")
    robust_loss: str | None = Field(default=None, description="Robust loss for least_squares.")
    seed: int | None = Field(default=None, ge=0, description="RNG seed.")


class LineProfileConfigOverrides(RequestModel):
    """All-optional overrides merged onto a lane's last-used ``LineProfileConfig``."""

    normalization: NormalizationType | None = Field(
        default=None, description="Profile normalization: 'area', 'min_max', or 'safe'."
    )
    slice_start: int | None = Field(default=None, ge=0, description="Profile slice start index.")
    slice_end: int | None = Field(default=None, description="Profile slice end index.")


class RefitLaneRequest(RequestModel):
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
    allow_raw: bool = Field(
        default=False,
        description="Measure the raw image although no background removal is configured. Off "
        "by default: profiles and fits of a raw image carry each lane's background floor into "
        "its normalization, so lanes are not comparable (background_required).",
    )


class HtmlReportRequest(RequestModel):
    """Write the static HTML report of a run folder."""

    run_dir: str = Field(description="Existing run folder.")
    css: str | None = Field(
        default=None,
        description="Path of a UTF-8 stylesheet file (a regular file of at most 1 MiB), copied "
        "unparsed and byte for byte as theme.css and linked after the default stylesheet, so it can "
        "override any --gelpy-* variable or rule. Its references are yours: a relative url() "
        "resolves against the report folder.",
    )
    out_dir: str | None = Field(
        default=None,
        description="Folder to write the report into (default: <run_dir>/report). It must be new, "
        "empty, or hold the output of an earlier report html (it carries .gelpy-report.json): any "
        "other non-empty folder is refused with invalid_value. The run folder, the folders inside "
        "it other than report/, the folders around it and any folder that holds a recipe.json or "
        "a manifest.json are refused whatever they carry, and report/ may not be a link, as the "
        "default or as out_dir. A rebuild replaces only the files that earlier build wrote, and "
        "never writes over an existing file its marker does not list, so the tool is not "
        "destructive. One report per folder: the run lock does not cover this folder.",
    )
