"""The incremental-recompute DAG walk.

Each ``ensure_*`` stage checks *its own* cache before touching its upstream
dependency, and the upstream is only fetched (and only then decoded/computed) on a
miss. So a fit cache-hit never re-decodes the image or re-runs RANSAC — it just
reads one small JSON file. Recompute/reuse is tracked per stage key for a terse
summary.
"""

from __future__ import annotations

import numpy as np

from ..core.background import background_registry
from ..core.errors import SourceImageChangedError
from ..core.image_io import apply_image_meta
from ..core.line_profiles import apply_slice, extract_line_profile, normalize_profile
from ..schemas.fit_config import FitConfig
from ..schemas.lanes import LaneDefinition
from ..schemas.peaks import LaneFitResult
from ..schemas.profiles import LineProfileConfig
from ..schemas.recipe import Recipe
from ..schemas.run import RunManifest
from . import stages
from .cache import get_or_compute_json, get_or_compute_npy, get_or_compute_npz, write_json
from .layout import RunPaths
from .manifest import record_stage
from .recipe_io import build_image_ref, source_unchanged
from .stages import StageName, stage_key


class Pipeline:
    """Walks image -> background -> profile[lane] -> fit[lane], caching each stage."""

    def __init__(self, paths: RunPaths, recipe: Recipe, manifest: RunManifest):
        self.paths = paths
        self.recipe = recipe
        self.manifest = manifest
        self._image: np.ndarray | None = None
        self._background_image: np.ndarray | None = None
        self._recomputed: set[str] = set()
        self._reused: set[str] = set()

    # --- provenance tracking -------------------------------------------------
    def _mark(self, key: str, *, recomputed: bool) -> None:
        if recomputed:
            self._recomputed.add(key)
            self._reused.discard(key)
        elif key not in self._recomputed:
            self._reused.add(key)

    def _note_upstream_reused(self, lane: LaneDefinition) -> None:
        """Mark image/background/profile as reused when a downstream cache hit means
        we never physically loaded them (they exist on disk and stay valid)."""
        if self.paths.image_cache(stages.image_hash(self.recipe)).exists():
            self._mark(stage_key(StageName.IMAGE), recomputed=False)
        if self.recipe.background is not None:
            if self.paths.background_cache(stages.background_hash(self.recipe)).exists():
                self._mark(stage_key(StageName.BACKGROUND), recomputed=False)
        prof_path = self.paths.profile_cache(
            lane.index, stages.profile_hash(self.recipe, lane)
        )
        if prof_path.exists():
            self._mark(stage_key(StageName.PROFILE, lane.index), recomputed=False)

    @property
    def recomputed(self) -> list[str]:
        return sorted(self._recomputed)

    @property
    def reused(self) -> list[str]:
        return sorted(self._reused)

    # --- stages --------------------------------------------------------------
    def ensure_image(self) -> np.ndarray:
        if self._image is not None:
            return self._image
        h = stages.image_hash(self.recipe)
        path = self.paths.image_cache(h)
        # The recipe's recorded sha256 is the source of truth for the image identity
        # and feeds every downstream hash. If the file on disk has genuinely changed
        # bytes, caching the new pixels under the old recipe hash would make the whole
        # chain incoherent (and risk a false hit if the file later reverts) — so we
        # surface it instead. The (path,size,mtime) fingerprint is a cheap pre-check
        # that avoids re-hashing an unchanged file.
        if not source_unchanged(self.recipe.source_image):
            fresh = build_image_ref(self.recipe.source_image.path)
            if fresh.sha256 != self.recipe.source_image.sha256:
                raise SourceImageChangedError(
                    f"Source image {self.recipe.source_image.path!r} has changed on disk "
                    f"(recipe sha256 {self.recipe.source_image.sha256[:12]}…, "
                    f"current {fresh.sha256[:12]}…). Re-run `gelpy init` against a fresh "
                    f"run folder, or point the recipe at the intended file."
                )
        existed = path.exists()
        arr = get_or_compute_npy(
            path, lambda: apply_image_meta(self.recipe.source_image.path, self.recipe.image)
        )
        self._mark(stage_key(StageName.IMAGE), recomputed=not existed)
        record_stage(self.manifest, stage_key(StageName.IMAGE), StageName.IMAGE.value, h, str(path))
        self._image = arr
        return arr

    def ensure_background_image(self) -> np.ndarray:
        """Return the analysis image: background-corrected if a stage is configured,
        otherwise the decoded image (passthrough — no background cache entry)."""
        if self._background_image is not None:
            return self._background_image
        if self.recipe.background is None:
            self._background_image = self.ensure_image()
            return self._background_image

        h = stages.background_hash(self.recipe)
        path = self.paths.background_cache(h)
        existed = path.exists()

        def _compute() -> dict[str, np.ndarray]:
            image = self.ensure_image()
            model = background_registry.get(self.recipe.background.model)()
            config_model = background_registry.config_model(self.recipe.background.model)
            params = config_model.model_validate(self.recipe.background.params)
            result = model.remove(image, params)
            return {"corrected": result.corrected_image}

        data = get_or_compute_npz(path, _compute)
        self._mark(stage_key(StageName.BACKGROUND), recomputed=not existed)
        record_stage(
            self.manifest, stage_key(StageName.BACKGROUND), StageName.BACKGROUND.value, h, str(path)
        )
        self._background_image = data["corrected"]
        return self._background_image

    def ensure_profile(self, lane: LaneDefinition) -> np.ndarray:
        """Full, unsliced column-mean profile for a lane (cached)."""
        h = stages.profile_hash(self.recipe, lane)
        path = self.paths.profile_cache(lane.index, h)
        existed = path.exists()

        def _compute() -> list[float]:
            image = self.ensure_background_image()
            profile = extract_line_profile(
                image, lane.x_center_px, self.recipe.lanes.profile_width_px
            )
            return [float(v) for v in profile]

        data = get_or_compute_json(path, _compute)
        self._mark(stage_key(StageName.PROFILE, lane.index), recomputed=not existed)
        record_stage(
            self.manifest,
            stage_key(StageName.PROFILE, lane.index),
            StageName.PROFILE.value,
            h,
            str(path),
        )
        return np.asarray(data, dtype=np.float64)

    def ensure_fit(
        self,
        lane: LaneDefinition,
        profile_config: LineProfileConfig,
        fit_config: FitConfig,
    ) -> LaneFitResult:
        """Fitted result for a lane (cached). Cache-hit reads one JSON file only."""
        h = stages.fit_hash(self.recipe, lane, profile_config, fit_config)
        path = self.paths.fit_cache(lane.index, h)
        key = stage_key(StageName.FIT, lane.index)

        if path.exists():
            try:
                result = LaneFitResult.model_validate_json(path.read_text())
            except Exception:
                path.unlink(missing_ok=True)  # corrupt cache -> fall through to recompute
            else:
                self._mark(key, recomputed=False)
                self._note_upstream_reused(lane)
                record_stage(self.manifest, key, StageName.FIT.value, h, str(path))
                return result

        full = self.ensure_profile(lane)
        prepared = normalize_profile(
            apply_slice(full, profile_config.slice_start, profile_config.slice_end),
            profile_config.normalization,
        )
        # Local import keeps core.peaks (numpy/scipy) lazy and registers the strategy tiers.
        from ..core.peaks.strategies import fit_strategy_registry

        strategy = fit_strategy_registry.get(fit_config.method)
        result = strategy(prepared, fit_config, lane_index=lane.index, label=lane.label)
        write_json(path, result)
        self._mark(key, recomputed=True)
        # Record that any upstream stages we did NOT recompute (because a nearer
        # cache hit short-circuited the lazy thunk) were reused, so the summary is
        # complete even though they were never physically loaded.
        self._note_upstream_reused(lane)
        record_stage(self.manifest, key, StageName.FIT.value, h, str(path))
        return result
