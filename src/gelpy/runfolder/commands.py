"""Programmatic command API: ``f(request) -> result``.

Every function here is pure request-in / result-out with no knowledge of argv or
stdout. That keeps the CLI thin and makes a future MCP server a near-zero-rework
wrapper (``mcp.tool()(commands.refit_lane)``). Domain errors are raised as
``GelpyError`` subclasses; the presentation layer turns them into ``ErrorResponse``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..core.background import background_registry
from ..core.errors import LaneSelectionError
from ..core.lanes import (
    compute_lane_positions,
    detect_lane_centers,
    guess_profile_width,
    resolve_lane_selector,
)
from ..core.line_profiles import strips_overlap
from ..schemas.background import BackgroundStageConfig
from ..schemas.contrast import ContrastConfig, GammaCorrection, IntensityRange
from ..schemas.fit_config import FitConfig, FitConfigStore, PerLaneFitConfig
from ..schemas.image import ImageRef
from ..schemas.lanes import (
    LaneDefinition,
    LaneDetectionResult,
    LaneLayout,
    LaneSelectorTerm,
)
from ..schemas.peaks import LaneFitResult
from ..schemas.profiles import LineProfileConfig
from ..schemas.recipe import Recipe
from ..schemas.requests import (
    DetectLanesRequest,
    ExtractProfilesRequest,
    FitConfigOverrides,
    FitRunRequest,
    GetResultsRequest,
    InitRunRequest,
    LineProfileConfigOverrides,
    RefitLaneRequest,
    RemoveBackgroundRequest,
    SetContrastRequest,
    SetImageRequest,
    SetLanesRequest,
    StatusRequest,
)
from ..schemas.results import AnalysisResult
from ..schemas.run import LaneStatus, RunManifest, RunStatus, RunUpdateSummary
from .fit_config_io import load_fit_config, save_fit_config
from .hashing import content_hash
from .layout import RunPaths
from .manifest import load_manifest, save_manifest
from .pipeline import Pipeline
from .recipe_io import build_image_ref, build_image_ref_and_meta, load_recipe, save_recipe
from .results import _fit_is_current, collect_results, lane_summary, write_results
from .stages import StageName, stage_key


# --- internal helpers --------------------------------------------------------
def _load(run_dir: str) -> tuple[RunPaths, Recipe, RunManifest]:
    paths = RunPaths(run_dir)
    if not paths.exists():
        raise FileNotFoundError(f"No run folder at {run_dir!r} (missing recipe.json)")
    return paths, load_recipe(paths), load_manifest(paths)


def _persist(paths: RunPaths, recipe: Recipe, manifest: RunManifest) -> None:
    manifest.recipe_hash = content_hash(recipe)
    save_recipe(paths, recipe)
    save_manifest(paths, manifest)


def _append_history(paths: RunPaths, command: str, payload: dict[str, Any]) -> None:
    paths.log.mkdir(parents=True, exist_ok=True)
    entry = {"t": time.time(), "command": command, "payload": payload}
    with open(paths.history, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def _rel(paths: RunPaths, p) -> str:
    try:
        return str(p.relative_to(paths.root))
    except ValueError:
        return str(p)


def image_warnings(meta) -> list[str]:
    """Non-fatal, ignorable image messages for the agent/user."""
    warnings: list[str] = []
    if meta.polarity_confidence == "low":
        warnings.append(
            f"image polarity is low-confidence ({meta.polarity_reason}); treating as "
            f"{meta.polarity}. Override with `image set --invert` / `--no-invert` if wrong."
        )
    if meta.source_pages > 1:
        warnings.append(
            f"source image has {meta.source_pages} pages; using page {meta.page}. "
            f"Select another with --page (init or `image set`)."
        )
    return warnings


# --- commands ----------------------------------------------------------------
def init_run(request: InitRunRequest) -> RunUpdateSummary:
    """Create a run folder with a default lane layout (refine later with set_lanes)."""
    paths = RunPaths(request.run_dir)
    paths.ensure_dirs()
    ref, image_meta = build_image_ref_and_meta(
        request.image_path,
        channel=request.channel,
        invert="auto" if request.invert is None else request.invert,
        page=request.page,
    )

    n_lanes = request.n_lanes
    positions = compute_lane_positions(0.05 * ref.width, 0.95 * ref.width, n_lanes)
    width = request.width or guess_profile_width(n_lanes, ref.width)
    labels = request.labels or [str(i + 1) for i in range(n_lanes)]
    if len(labels) != n_lanes:
        raise ValueError(f"expected {n_lanes} labels, got {len(labels)}")

    layout = _build_layout(positions, labels, width, request.label_rotation)
    recipe = Recipe(source_image=ref, lanes=layout, image=image_meta)
    manifest = RunManifest(recipe_hash=content_hash(recipe), image=ref)
    _persist(paths, recipe, manifest)
    save_fit_config(paths, FitConfigStore())  # empty per-lane store; the file always exists
    _append_history(paths, "init", {"image": request.image_path, "n_lanes": n_lanes})
    return RunUpdateSummary(
        files={"recipe": _rel(paths, paths.recipe)},
        warnings=image_warnings(image_meta),
        image=image_meta.model_dump(mode="json"),
    )


def set_image(request: SetImageRequest) -> RunUpdateSummary:
    """Re-interpret the source image; changing channel/invert/page invalidates caches by hash."""
    paths, recipe, manifest = _load(request.run_dir)
    current = recipe.image
    channel = request.channel or current.channel_strategy
    page = current.page if request.page is None else request.page
    if request.auto:
        invert: bool | str = "auto"
    elif request.invert is not None:
        invert = request.invert
    else:
        invert = current.inverted  # keep current polarity decision
    _, image_meta = build_image_ref_and_meta(
        recipe.source_image.path, channel=channel, invert=invert, page=page
    )
    recipe.image = image_meta
    _persist(paths, recipe, manifest)
    _append_history(paths, "set_image", {"channel": channel, "invert": invert, "page": page})
    return RunUpdateSummary(
        files={"recipe": _rel(paths, paths.recipe)},
        warnings=image_warnings(image_meta),
        image=image_meta.model_dump(mode="json"),
    )


def set_lanes(request: SetLanesRequest) -> RunUpdateSummary:
    """Set lane positions / labels / width. Changing positions or width auto-invalidates
    downstream caches by hash — no explicit cache clearing needed."""
    paths, recipe, manifest = _load(request.run_dir)
    linspace = request.linspace
    x_centers = request.x_centers
    labels = request.labels
    width = request.width
    label_rotation = request.label_rotation

    if linspace is not None:
        left, right, n = linspace
        positions = compute_lane_positions(left, right, int(n))
    elif request.endpoints is not None:
        left, right = request.endpoints
        n = request.n_lanes or len(recipe.lanes.lanes)
        if n < 1:
            raise ValueError("endpoints needs n_lanes (no existing lanes to infer the count from)")
        positions = compute_lane_positions(left, right, int(n))
    elif x_centers is not None:
        positions = list(x_centers)
    else:
        positions = [lane.x_center_px for lane in recipe.lanes.lanes]

    if labels is None:
        labels = recipe.lanes.labels()
        if len(labels) != len(positions):
            labels = [str(i + 1) for i in range(len(positions))]
    if len(labels) != len(positions):
        raise ValueError(f"labels ({len(labels)}) must match positions ({len(positions)})")

    width = width or recipe.lanes.profile_width_px
    rotation = label_rotation if label_rotation is not None else recipe.lanes.label_rotation
    recipe.lanes = _build_layout(positions, labels, width, rotation)

    summary = RunUpdateSummary(files={"recipe": _rel(paths, paths.recipe)})
    overlaps = strips_overlap(positions, width)
    if overlaps:
        summary.lanes["warning"] = {
            "overlapping_strips": overlaps,
            "message": "adjacent lane profile strips overlap at this width",
        }
    _persist(paths, recipe, manifest)
    _append_history(paths, "set_lanes", {"n": len(positions), "width": width})
    return summary


def _normalize_y_band(y_band: object | None) -> list[list[int]] | None:
    """Coerce a strategy's ``y_band`` (None, a single ``(lo, hi)``, or a list of them) to the
    ``list[[lo, hi]]`` the result schema carries."""
    if y_band is None:
        return None
    pairs = y_band if isinstance(y_band, (list, tuple)) and y_band and isinstance(
        y_band[0], (list, tuple)
    ) else [y_band]
    return [[int(lo), int(hi)] for lo, hi in pairs]


def detect_lanes(request: DetectLanesRequest) -> LaneDetectionResult:
    """Detect lane centers from the image by fitting an even N-lane grid to its y-projection.

    N is ground truth (never inferred). The result carries a ``confidence`` flag; callers
    should render the verification overlay (``cli/_render.render_lanes``) and, on ``"low"``
    confidence, fall back to ``set_lanes(endpoints=...)``. With ``apply`` the detected
    centers are persisted exactly as ``set_lanes`` would, so downstream caches invalidate.
    """
    paths, recipe, manifest = _load(request.run_dir)
    n = request.n_lanes or len(recipe.lanes.lanes)
    if n < 1:
        raise ValueError("n_lanes required (no existing lanes to infer the count from)")

    pipeline = Pipeline(paths, recipe, manifest)
    image = pipeline.ensure_background_image() if request.use_corrected else pipeline.ensure_image()

    detection = detect_lane_centers(
        image,
        n,
        y_start=request.y_start,
        y_end=request.y_end,
        smoothing_window=request.smoothing_window,
        min_prominence=request.min_prominence,
        min_distance=request.min_distance,
        min_width=request.min_width,
        projection_strategy=request.projection_strategy,
    )

    applied = False
    if request.apply:
        labels = request.labels
        if labels is None:
            existing = recipe.lanes.labels()
            labels = existing if len(existing) == n else [str(i + 1) for i in range(n)]
        if len(labels) != n:
            raise ValueError(f"expected {n} labels, got {len(labels)}")
        rotation = (
            request.label_rotation
            if request.label_rotation is not None
            else recipe.lanes.label_rotation
        )
        recipe.lanes = _build_layout(detection.centers, labels, detection.profile_width, rotation)
        _persist(paths, recipe, manifest)
        _append_history(
            paths,
            "detect_lanes",
            {"n": n, "pitch": detection.pitch, "confidence": detection.confidence},
        )
        applied = True

    return LaneDetectionResult(
        centers=detection.centers,
        pitch=detection.pitch,
        profile_width=detection.profile_width,
        n_detected=detection.n_detected,
        n_expected=detection.n_expected,
        confidence=detection.confidence,
        confidence_reason=detection.confidence_reason,
        projection_source=detection.projection_source,
        y_band=_normalize_y_band(detection.y_band),
        applied=applied,
    )


def set_contrast(request: SetContrastRequest) -> RunUpdateSummary:
    """Adjust display contrast (cosmetic; never affects analysis)."""
    paths, recipe, manifest = _load(request.run_dir)
    contrast = recipe.contrast or ContrastConfig()
    g = contrast.gamma
    contrast.gamma = GammaCorrection(
        gamma=request.gamma if request.gamma is not None else g.gamma,
        gain=request.gain if request.gain is not None else g.gain,
    )
    if request.intensity_low is not None and request.intensity_high is not None:
        contrast.intensity_range = IntensityRange(
            low=request.intensity_low,
            high=request.intensity_high,
            mode=request.intensity_mode or "percentile",
        )
    recipe.contrast = contrast
    _persist(paths, recipe, manifest)
    _append_history(paths, "set_contrast", {"gamma": request.gamma, "gain": request.gain})
    return RunUpdateSummary(files={"recipe": _rel(paths, paths.recipe)})


def remove_background(request: RemoveBackgroundRequest) -> RunUpdateSummary:
    """Configure background removal. Applied lazily by the pipeline, not eagerly."""
    paths, recipe, manifest = _load(request.run_dir)
    model = request.model
    background_registry.get(model)  # validates the name (raises RegistryError if unknown)
    params: dict[str, Any] = dict(request.params)
    if request.stripes is not None:  # plane_ransac-only shim
        params["stripes"] = request.stripes
    # Validate params against the model's schema and store the *canonical* form
    # (defaults resolved, types coerced) so the recipe and its hash stay consistent. An
    # unknown param (e.g. --stripe on robust_plane) fails here with a clear ValidationError.
    config_model = background_registry.config_model(model)
    if config_model is not None:
        params = config_model.model_validate(params).model_dump(mode="json")
    recipe.background = BackgroundStageConfig(model=model, params=params)
    _persist(paths, recipe, manifest)
    _append_history(paths, "remove_background", {"model": model})
    return RunUpdateSummary(files={"recipe": _rel(paths, paths.recipe)})


def extract_profiles(request: ExtractProfilesRequest) -> RunUpdateSummary:
    """Extract (and cache) the raw line profiles for a lane selection."""
    paths, recipe, manifest = _load(request.run_dir)
    selected = _resolve(recipe, request.lanes)
    pipeline = Pipeline(paths, recipe, manifest)
    for lane in selected:
        pipeline.ensure_profile(lane)
    save_manifest(paths, manifest)
    _append_history(paths, "extract_profiles", {"lanes": request.lanes})
    return RunUpdateSummary(
        recomputed=pipeline.recomputed,
        reused=pipeline.reused,
    )


#: The escalation ladder: cheap deterministic default -> deconvolution seeder -> Bayesian.
_TIER_LADDER = ["grow_select", "sparse_seed", "bayesian"]


def _fit_with_escalation(pipeline, lane, profile_config, fit_config, auto_escalate):
    """Fit a lane; if ``auto_escalate`` and confidence stays 'low', climb the tier ladder.

    Returns ``(result, used_fit_config)`` — the config actually applied (so an escalated
    method is what gets persisted for the lane)."""
    result = pipeline.ensure_fit(lane, profile_config, fit_config)
    used = fit_config
    if not auto_escalate or result.confidence == "ok":
        return result, used
    try:
        start = _TIER_LADDER.index(fit_config.method)
    except ValueError:
        start = 0
    for method in _TIER_LADDER[start + 1 :]:
        used = fit_config.model_copy(update={"method": method})
        result = pipeline.ensure_fit(lane, profile_config, used)
        if result.confidence == "ok":
            break
    return result, used


def fit_run(request: FitRunRequest) -> RunUpdateSummary:
    """Run a full fit pass over a lane selection."""
    paths, recipe, manifest = _load(request.run_dir)
    selected = _resolve(recipe, request.lanes)
    pipeline = Pipeline(paths, recipe, manifest)
    store = load_fit_config(paths)

    summary = RunUpdateSummary()
    for lane in selected:
        result, used = _fit_with_escalation(
            pipeline, lane, request.profile, request.fit, request.auto_escalate
        )
        # One config applies to the whole selection: seed every lane (last-write-wins).
        store.lanes[lane.index] = PerLaneFitConfig(profile=request.profile, fit=used)
        summary.lanes[f"lane={lane.index}"] = lane_summary(result)

    write_results(paths, manifest, recipe, store)
    save_manifest(paths, manifest)
    save_fit_config(paths, store)
    _append_history(paths, "fit_run", request.model_dump(mode="json"))

    summary.recomputed = pipeline.recomputed
    summary.reused = pipeline.reused
    summary.files = _result_files(paths)
    return summary


def refit_lane(request: RefitLaneRequest) -> RunUpdateSummary:
    """Refit a single lane, merging overrides onto its last-used settings."""
    paths, recipe, manifest = _load(request.run_dir)
    selected = _resolve(recipe, request.lane)
    if len(selected) != 1:
        raise LaneSelectionError(
            f"refit-lane needs exactly one lane; {request.lane!r} matched {len(selected)}.",
            available=recipe.lanes.labels(),
        )
    lane = selected[0]

    store = load_fit_config(paths)
    base = store.lanes.get(lane.index)  # None if this lane was never fit
    profile_config = _merge_profile(base.profile if base else None, request.profile_overrides)
    fit_config = _merge_fit(base.fit if base else None, request.fit_overrides)

    pipeline = Pipeline(paths, recipe, manifest)
    result, used = _fit_with_escalation(
        pipeline, lane, profile_config, fit_config, request.auto_escalate
    )
    # Persist the merged result so the next refit builds on it (last-write-wins).
    store.lanes[lane.index] = PerLaneFitConfig(profile=profile_config, fit=used)

    write_results(paths, manifest, recipe, store)
    save_manifest(paths, manifest)
    save_fit_config(paths, store)
    _append_history(paths, "refit_lane", request.model_dump(mode="json"))

    summary = RunUpdateSummary(
        recomputed=pipeline.recomputed,
        reused=pipeline.reused,
        files=_result_files(paths, lane_index=lane.index),
    )
    summary.lanes[f"lane={lane.index}"] = lane_summary(result)
    return summary


def get_results(request: GetResultsRequest) -> AnalysisResult:
    """Return the assembled results from cache without recomputing anything."""
    paths, recipe, manifest = _load(request.run_dir)
    return collect_results(manifest, recipe, load_fit_config(paths))


def status(request: StatusRequest) -> RunStatus:
    """Report the current state of a run folder — configured / fit / stale lanes.

    Pure assembly over the recipe, manifest, and fit-config store; nothing is
    recomputed. This is the single call an agent (or human) uses to orient in a run
    folder it did not just create.
    """
    paths, recipe, manifest = _load(request.run_dir)
    store = load_fit_config(paths)

    lanes: list[LaneStatus] = []
    n_fit = n_stale = n_failed = 0
    for lane in recipe.lanes.lanes:
        entry = LaneStatus(index=lane.index, label=lane.label)
        record = manifest.stages.get(stage_key(StageName.FIT, lane.index))
        if record is not None and Path(record.cache_path).exists():
            entry.has_fit = True
            entry.fit_current = _fit_is_current(record, recipe, store)
            result = LaneFitResult.model_validate_json(Path(record.cache_path).read_text())
            entry.status = result.status
            entry.n_bands = len(result.bands)
            if entry.fit_current:
                n_fit += 1
                if result.status == "failed":
                    n_failed += 1
            else:
                n_stale += 1
        lanes.append(entry)

    return RunStatus(
        run_dir=request.run_dir,
        source_image=recipe.source_image.path,
        image_width=recipe.source_image.width,
        image_height=recipe.source_image.height,
        background=recipe.background.model if recipe.background else None,
        polarity=recipe.image.polarity,
        polarity_confidence=recipe.image.polarity_confidence,
        warnings=image_warnings(recipe.image),
        n_lanes=len(recipe.lanes.lanes),
        n_fit=n_fit,
        n_stale=n_stale,
        n_failed=n_failed,
        lanes=lanes,
    )


# --- helpers -----------------------------------------------------------------
def _build_layout(positions, labels, width, rotation) -> LaneLayout:
    lanes = [
        LaneDefinition(index=i, label=str(label), x_center_px=float(pos))
        for i, (pos, label) in enumerate(zip(positions, labels))
    ]
    return LaneLayout(lanes=lanes, profile_width_px=int(width), label_rotation=float(rotation))


def _resolve(recipe: Recipe, selector) -> list[LaneDefinition]:
    indices = resolve_lane_selector(selector, recipe.lanes)
    return [recipe.lanes.lanes[i] for i in indices]


def _result_files(paths: RunPaths, lane_index: int | None = None) -> dict[str, str]:
    files = {
        "results_csv": _rel(paths, paths.results_csv),
        "results_json": _rel(paths, paths.results_json),
    }
    if lane_index is not None:
        files["plot"] = _rel(paths, paths.fit_render(lane_index))
    return files


def _merge_profile(
    base: LineProfileConfig | None, overrides: LineProfileConfigOverrides
) -> LineProfileConfig:
    config = base or LineProfileConfig()
    patch = overrides.model_dump(exclude_none=True)
    return config.model_copy(update=patch)


def _merge_fit(base: FitConfig | None, overrides: FitConfigOverrides) -> FitConfig:
    config = base or FitConfig()
    # NOTE: build the patch from the actual field values, not `.model_dump()` — dump
    # recursively converts nested models (baseline, priors) to plain dicts, and
    # `model_copy(update=...)` does not re-validate, so a dumped BaselineConfig would
    # silently become an unvalidated dict on the merged FitConfig (AttributeError at use).
    patch = {name: value for name, value in overrides.__dict__.items() if value is not None}
    return config.model_copy(update=patch)
