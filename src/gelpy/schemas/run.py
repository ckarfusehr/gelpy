"""Run manifest (cache ledger) and command summary models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from .common import CanonicalModel
from .image import ImageRef
from .report import ReportStatus


class StageRecord(CanonicalModel):
    """One entry in the cache ledger: what was computed, its hash, and where it lives."""

    stage: str = Field(description="Stage name (image, background, profile, fit).")
    key: str = Field(description="Fully-qualified stage key, e.g. 'fit:lane=12'.")
    hash: str = Field(description="Content hash that this cache entry is keyed by.")
    cache_path: str = Field(description="Run-relative path to the cached artifact.")
    inputs: dict[str, Any] = Field(
        default_factory=dict, description="Resolved inputs that produced the hash (for debugging)."
    )
    gelpy_version: str = Field(
        default="",
        description="gelpy version that computed this stage ('' before 1.1). A fit with no "
        "entry in fit_config.json has no config to recompute its hash from; this tells whether "
        "it is stale after an upgrade.",
    )


class RunManifest(CanonicalModel):
    """Disposable cache ledger, fully regenerable from ``recipe.json`` + source image.

    ``stages`` is keyed by fully-qualified stage key (e.g. ``"image"``,
    ``"background"``, ``"profile:lane=12"``, ``"fit:lane=12"``).
    """

    recipe_hash: str = Field(description="Content hash of the recipe that produced this manifest.")
    gelpy_version: str = Field(
        default="", description="gelpy version that last wrote this manifest ('' before 1.1)."
    )
    image: ImageRef | None = Field(default=None, description="Resolved source-image identity.")
    stages: dict[str, StageRecord] = Field(default_factory=dict)


class RunUpdateSummary(CanonicalModel):
    """Terse result of a command: what recomputed, what was reused, where files landed.

    Large arrays never pass through this model as text — files stay as files and are
    referenced by run-relative path.
    """

    recomputed: list[str] = Field(default_factory=list)
    reused: list[str] = Field(default_factory=list)
    lanes: dict[str, Any] = Field(default_factory=dict)
    files: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(
        default_factory=list, description="Non-fatal, ignorable messages for the agent/user."
    )
    image: dict[str, Any] = Field(
        default_factory=dict, description="Structured image interpretation decision (polarity/channel)."
    )


class LaneStatus(CanonicalModel):
    """Per-lane state within a run folder (from cache; nothing is recomputed)."""

    index: int = Field(ge=0, description="Lane index.")
    label: str = Field(description="Lane label.")
    has_fit: bool = Field(
        default=False,
        description="A fit for this lane exists in the cache: its file is there and, for a "
        "current fit, validates (a damaged file counts as missing).",
    )
    fit_current: bool = Field(
        default=False,
        description="The fit the recipe, the stored configuration and the running gelpy version "
        "call for is in the cache and its file validates: it can be used as it stands. An edit "
        "that is undone makes its fit current again at once.",
    )
    status: Literal["ok", "empty", "failed"] | None = Field(
        default=None,
        description="Outcome of the current cached fit; null when the lane has none (the file of "
        "a stale fit is not read). Mirrors LaneFitResult.status, "
        "including 'empty' (converged, but no significant signal in the window) — a "
        "narrower set here would make `status` fail on a valid run.",
    )
    n_bands: int = Field(
        default=0,
        ge=0,
        description="Number of fitted bands in the current cached fit; 0 when the lane has none.",
    )


class RunStatus(CanonicalModel):
    """A cold-orientation snapshot of a run folder: what's configured, fit, or stale.

    Assembled from the recipe, manifest, and fit-config store without recomputing —
    the single call an agent (or human) uses to learn a run's state.
    """

    run_dir: str = Field(description="The run folder this status describes.")
    source_image: str = Field(
        description="The source image path as the recipe records it: relative to the run folder "
        "where possible."
    )
    image_width: int = Field(gt=0, description="Source image width (px).")
    image_height: int = Field(gt=0, description="Source image height (px).")
    background: str | None = Field(
        default=None, description="Configured background model name, or null if none."
    )
    polarity: Literal["bright_on_dark", "dark_on_light"] | None = Field(
        default=None, description="Recorded source-image polarity."
    )
    polarity_confidence: Literal["ok", "low"] | None = Field(
        default=None, description="Polarity-detection confidence."
    )
    warnings: list[str] = Field(
        default_factory=list, description="Unresolved non-fatal image warnings."
    )
    n_lanes: int = Field(default=0, ge=0, description="Number of lanes in the recipe.")
    n_fit: int = Field(default=0, ge=0, description="Lanes with a current (non-stale) fit.")
    n_stale: int = Field(
        default=0,
        ge=0,
        description="Lanes whose recorded fit cannot be used as it stands: the recipe, the stored "
        "configuration or the running gelpy version changed since it was computed and the fit they "
        "now call for is not in the cache, or its cache file is missing or damaged. After an edit "
        "is undone the lane is current again at once: the cache keeps every fit it computed.",
    )
    n_failed: int = Field(default=0, ge=0, description="Lanes whose current fit failed.")
    stale_stages: list[str] = Field(
        default_factory=list,
        description="Stages the next command that needs them would recompute: recorded in the "
        "ledger, for a lane the recipe still has, with no cache artifact for the current recipe "
        "and running gelpy version (a damaged fit file counts as missing). Examples: "
        "'background' after a background change, 'profile:lane=3' after a lane moved. A stage "
        "whose artifact is cached is not listed, so after an edit is undone nothing is. A deleted "
        "artifact stays listed until a command that needs it computes it again: a fit that is "
        "cached pulls on nothing upstream, so `fit run` leaves a missing image, background or "
        "profile listed. Lane stages are sorted by kind, then lane number.",
    )
    lanes: list[LaneStatus] = Field(default_factory=list, description="Per-lane state.")
    report: ReportStatus | None = Field(
        default=None,
        description="The HTML report in the run's own report/ folder and whether it is current "
        "(it shows the run as it is now), or null when there is none: no folder, or one that "
        "`gelpy report html` did not write.",
    )
    gelpy_version: str = Field(default="", description="The gelpy version answering this call.")
