"""Run manifest (cache ledger) and command summary models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from .common import CanonicalModel
from .image import ImageRef


class StageRecord(CanonicalModel):
    """One entry in the cache ledger: what was computed, its hash, and where it lives."""

    stage: str = Field(description="Stage name (image, background, profile, fit).")
    key: str = Field(description="Fully-qualified stage key, e.g. 'fit:lane=12'.")
    hash: str = Field(description="Content hash that this cache entry is keyed by.")
    cache_path: str = Field(description="Run-relative path to the cached artifact.")
    inputs: dict[str, Any] = Field(
        default_factory=dict, description="Resolved inputs that produced the hash (for debugging)."
    )


class RunManifest(CanonicalModel):
    """Disposable cache ledger, fully regenerable from ``recipe.json`` + source image.

    ``stages`` is keyed by fully-qualified stage key (e.g. ``"image"``,
    ``"background"``, ``"profile:lane=12"``, ``"fit:lane=12"``).
    """

    recipe_hash: str = Field(description="Content hash of the recipe that produced this manifest.")
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
    has_fit: bool = Field(default=False, description="A fit for this lane exists in the cache.")
    fit_current: bool = Field(
        default=False, description="The cached fit still matches the current recipe (not stale)."
    )
    status: Literal["ok", "empty", "failed"] | None = Field(
        default=None,
        description="Outcome of the cached fit, if any. Mirrors LaneFitResult.status, "
        "including 'empty' (converged, but no significant signal in the window) — a "
        "narrower set here would make `status` fail on a valid run.",
    )
    n_bands: int = Field(default=0, ge=0, description="Number of fitted bands in the cached fit.")


class RunStatus(CanonicalModel):
    """A cold-orientation snapshot of a run folder: what's configured, fit, or stale.

    Assembled from the recipe, manifest, and fit-config store without recomputing —
    the single call an agent (or human) uses to learn a run's state.
    """

    run_dir: str = Field(description="The run folder this status describes.")
    source_image: str = Field(description="Path to the source image referenced by the recipe.")
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
        default=0, ge=0, description="Lanes with a fit that no longer matches the recipe."
    )
    n_failed: int = Field(default=0, ge=0, description="Lanes whose current fit failed.")
    lanes: list[LaneStatus] = Field(default_factory=list, description="Per-lane state.")
