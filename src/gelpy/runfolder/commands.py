"""Programmatic command API: ``f(request) -> result``.

Every function here is pure request-in / result-out with no knowledge of argv or stdout. A CLI
step is more than its command (``lanes detect`` also renders the overlay), so ``cli/tools.py``
pairs each request with a handler that runs the command and its rendering; the ``plot`` flags
are inert here. Domain errors are ``GelpyError`` subclasses, turned into ``ErrorResponse`` upstream.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..core.background import background_registry
from ..core.errors import (
    BackgroundRequiredError,
    InvalidValueError,
    LaneSelectionError,
    RunExistsError,
    shown,
)
from ..core.lanes import (
    LaneDetection,
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
from ..schemas.report import REPORT_MARKER
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
from ..utils.atomic import is_temp_name
from .fit_config_io import load_fit_config, save_fit_config
from . import stages
from .hashing import content_hash
from .layout import RunPaths, open_no_follow
from .lock import locked, run_lock
from .manifest import load_manifest, save_manifest
from .pipeline import Pipeline
from .recipe_io import (
    build_image_ref,
    build_image_ref_and_meta,
    load_recipe,
    require_source,
    require_unchanged,
    save_recipe,
)
from .report_folder import clear_report, is_empty, report_fingerprint, report_status
from .results import (
    _fit_is_current,
    _lane_index_of,
    collect_results,
    follow_the_cache,
    lane_summary,
    read_fit_result,
    write_results,
)
from .stages import StageName, code_version, stage_key


# --- internal helpers --------------------------------------------------------
def _load(run_dir: str) -> tuple[RunPaths, Recipe, RunManifest]:
    paths = RunPaths(run_dir)
    if not paths.exists():
        raise FileNotFoundError(f"No run folder at {shown(run_dir)} (missing recipe.json)")
    return paths, load_recipe(paths), load_manifest(paths)


def _persist(paths: RunPaths, recipe: Recipe, manifest: RunManifest) -> None:
    manifest.recipe_hash = content_hash(recipe)
    save_recipe(paths, recipe)
    save_manifest(paths, manifest)


def _append_history(paths: RunPaths, command: str, payload: dict[str, Any]) -> None:
    """Append one line to ``log/history.jsonl``, which is opened in place and never through a link.

    A link there is refused (``invalid_value``); ``RunPaths.require_real_folders`` already refused
    it when the command took the run lock, so this is the check for one planted since."""
    paths.log.mkdir(parents=True, exist_ok=True)
    entry = {"t": time.time(), "command": command, "payload": payload, "gelpy_version": code_version()}
    line = (json.dumps(entry, default=str) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0)
    fd = open_no_follow(paths.history, flags, 0o666, name="log/history.jsonl")
    try:
        handle = os.fdopen(fd, "ab")
    except BaseException:
        os.close(fd)
        raise
    with handle:
        handle.write(line)


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
    if meta.reader_warnings:
        warnings.append(
            f"tifffile reported while reading the image: {'; '.join(meta.reader_warnings)}. The file "
            "may be damaged or truncated: check source_pages, and export it again if pages are missing."
        )
    return warnings


# --- commands ----------------------------------------------------------------
def init_run(request: InitRunRequest) -> RunUpdateSummary:
    """Create a run folder with a default lane layout (refine later with set_lanes).

    The image is read before the folder is touched, so a wrong path or a refused file leaves
    nothing behind. A folder that already holds a run is refused (``run_exists``) unless
    ``request.force``: then the run's intent is replaced and the outputs derived from the old
    one (results, report, renders) are removed; the content-addressed cache and the history
    log are kept.
    """
    paths = RunPaths(request.run_dir)
    ref, image_meta = build_image_ref_and_meta(
        request.image_path,
        channel=request.channel,
        invert="auto" if request.invert is None else request.invert,
        page=request.page,
    )
    n_lanes = request.n_lanes
    labels = request.labels or [str(i + 1) for i in range(n_lanes)]
    if len(labels) != n_lanes:
        raise InvalidValueError(
            f"expected {n_lanes} labels, got {len(labels)}",
            detail="Pass one label per lane, or set n_lanes to the number of labels.",
        )
    cleared: list[str] = []  # what init --force removed and left, in words
    with run_lock(paths.root, create=True):
        exists = paths.exists()
        if exists and not request.force:
            raise RunExistsError(f"{shown(request.run_dir)} already holds a run (it has a recipe.json).")
        # Before anything is removed or written: a link where a folder belongs is refused. init
        # --force removes render/ whole, and a link there with it (``_clear_derived``).
        paths.require_real_folders(replaced=("render",) if exists else ())
        if exists:
            cleared = _clear_derived(paths)
        paths.ensure_dirs()
        # recorded now that the folder exists: a relative record is checked against it
        ref = ref.model_copy(update={"path": paths.store_path(request.image_path)})
        positions = compute_lane_positions(0.05 * ref.width, 0.95 * ref.width, n_lanes)
        width = request.width or guess_profile_width(n_lanes, ref.width)
        layout = _build_layout(positions, labels, width, request.label_rotation)
        recipe = Recipe(source_image=ref, lanes=layout, image=image_meta)
        manifest = RunManifest(recipe_hash=content_hash(recipe), image=ref)
        _persist(paths, recipe, manifest)
        save_fit_config(paths, FitConfigStore())  # empty per-lane store; the file always exists
        _append_history(
            paths,
            "init",
            {"image": ref.path, "n_lanes": n_lanes, "force": request.force},
        )
    return RunUpdateSummary(
        files={"recipe": _rel(paths, paths.recipe)},
        warnings=[*image_warnings(image_meta), *cleared],
        image=image_meta.model_dump(mode="json"),
    )


@locked
def set_image(request: SetImageRequest) -> RunUpdateSummary:
    """Re-interpret the source image; changing channel/invert/page invalidates caches by hash.

    Refused (``source_image_changed``) when the file at the recorded path is not the image the
    recipe records: its pixels would be interpreted in the recipe's name.
    """
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
    source = paths.resolve(recipe.source_image.path)
    require_source(recipe.source_image, source)
    ref, image_meta = build_image_ref_and_meta(source, channel=channel, invert=invert, page=page)
    require_unchanged(recipe.source_image, source, ref.sha256)
    recipe.image = image_meta
    _persist(paths, recipe, manifest)
    _append_history(paths, "set_image", {"channel": channel, "invert": invert, "page": page})
    return RunUpdateSummary(
        files={"recipe": _rel(paths, paths.recipe)},
        warnings=image_warnings(image_meta),
        image=image_meta.model_dump(mode="json"),
    )


@locked
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
            raise InvalidValueError(
                "endpoints needs n_lanes (the run has no lanes to infer the count from)",
                detail="Pass n_lanes (--n-lanes) together with endpoints.",
            )
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
        raise InvalidValueError(
            f"labels ({len(labels)}) must match positions ({len(positions)})",
            detail="Pass one label per lane position.",
        )

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


@dataclass(frozen=True, eq=False)
class LaneDetectionOutcome:
    """A lane detection's result plus what its verification overlay is drawn from, so a caller
    that also renders the overlay never runs the detection, or loads the image, twice.

    ``y_start`` and ``y_end`` are the row bounds the detection was given (``None``: the edge of
    the image), so the overlay is drawn with the window the projection was built from. The
    outcome holds arrays: it compares and hashes by identity."""

    result: LaneDetectionResult
    image: np.ndarray
    detection: LaneDetection
    y_start: int | None = None
    y_end: int | None = None


@locked
def run_lane_detection(request: DetectLanesRequest) -> LaneDetectionOutcome:
    """``detect_lanes``, also returning the image, the detection's arrays and the row bounds."""
    paths, recipe, manifest = _load(request.run_dir)
    n = request.n_lanes or len(recipe.lanes.lanes)
    if n < 1:
        raise InvalidValueError(
            "n_lanes required (the run has no lanes to infer the count from)",
            detail="Pass n_lanes (--n-lanes).",
        )

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
            raise InvalidValueError(
                f"expected {n} labels, got {len(labels)}",
                detail="Pass one label per detected lane, or change n_lanes.",
            )
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

    return LaneDetectionOutcome(
        result=LaneDetectionResult(
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
        ),
        image=image,
        detection=detection,
        y_start=request.y_start,
        y_end=request.y_end,
    )


@locked  # re-entrant with run_lane_detection's own lock; keeps every public writer visibly locked
def detect_lanes(request: DetectLanesRequest) -> LaneDetectionResult:
    """Detect lane centers from the image by fitting an even N-lane grid to its y-projection.

    N is ground truth (never inferred). The result carries a ``confidence`` flag; callers
    should render the verification overlay (``run_lane_detection`` returns what
    ``cli/_render.render_lane_overlay`` draws it from) and, on ``"low"`` confidence, fall back
    to ``set_lanes(endpoints=...)``. With ``apply`` the detected centers are persisted exactly
    as ``set_lanes`` would, so downstream caches invalidate.
    """
    return run_lane_detection(request).result


@locked
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


@locked
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


@locked
def extract_profiles(request: ExtractProfilesRequest) -> RunUpdateSummary:
    """Extract (and cache) the raw line profiles for a lane selection.

    Refused (``background_required``) on a run without background removal unless
    ``request.allow_raw``; a lane selector that matches nothing is reported first.
    """
    paths, recipe, manifest = _load(request.run_dir)
    selected = _resolve(recipe, request.lanes)
    _require_background(recipe, request.allow_raw)
    pipeline = Pipeline(paths, recipe, manifest)
    for lane in selected:
        pipeline.ensure_profile(lane)
    save_manifest(paths, manifest)
    _append_history(
        paths, "extract_profiles", {"lanes": request.lanes, "allow_raw": request.allow_raw}
    )
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


@locked
def fit_run(request: FitRunRequest) -> RunUpdateSummary:
    """Run a full fit pass over a lane selection.

    Refused (``background_required``) on a run without background removal unless
    ``request.allow_raw``; a lane selector that matches nothing is reported first.
    """
    paths, recipe, manifest = _load(request.run_dir)
    selected = _resolve(recipe, request.lanes)
    _require_background(recipe, request.allow_raw)
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
    _append_history(paths, "fit_run", request.model_dump(mode="json", exclude={"run_dir"}))

    summary.recomputed = pipeline.recomputed
    summary.reused = pipeline.reused
    summary.files = _result_files(paths)
    return summary


@locked
def refit_lane(request: RefitLaneRequest) -> RunUpdateSummary:
    """Refit a single lane, merging overrides onto its last-used settings.

    Refused (``background_required``) on a run without background removal unless
    ``request.allow_raw``; a selector that does not match exactly one lane is reported first.
    """
    paths, recipe, manifest = _load(request.run_dir)
    selected = _resolve(recipe, request.lane)
    if len(selected) != 1:
        raise LaneSelectionError(
            f"refit-lane needs exactly one lane; {shown(request.lane)} matched {len(selected)}.",
            available=recipe.lanes.labels(), detail="Pass exactly one lane: a label or an index.",
        )
    _require_background(recipe, request.allow_raw)
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
    _append_history(paths, "refit_lane", request.model_dump(mode="json", exclude={"run_dir"}))

    summary = RunUpdateSummary(
        recomputed=pipeline.recomputed,
        reused=pipeline.reused,
        files=_result_files(paths),
    )
    summary.lanes[f"lane={lane.index}"] = lane_summary(result)
    return summary


def get_results(request: GetResultsRequest) -> AnalysisResult:
    """Return the assembled results from cache without recomputing anything."""
    paths, recipe, manifest = _load(request.run_dir)
    damaged: list[int] = []
    result = collect_results(paths, manifest, recipe, load_fit_config(paths), damaged=damaged)
    result.warnings = [
        f"Lane {index} is left out: its fit file is damaged. Fit it again with "
        f"`gelpy fit refit-lane RUN --lane {index}` (tool `fit_refit_lane`)."
        for index in damaged
    ]
    return result


def status(request: StatusRequest) -> RunStatus:
    """Report the current state of a run folder — configured / fit / stale lanes and stages.

    Pure assembly over the recipe, manifest, and fit-config store; nothing is
    recomputed. This is the single call an agent (or human) uses to orient in a run
    folder it did not just create. A fit is current when the recipe, the stored configuration
    and the running gelpy version call for it and it is in the cache (``follow_the_cache``), so
    after an undone edit it is current at once. Only the cache file of a current fit is read:
    after an upgrade the file of a stale fit was written by another release and need not
    validate, and a current fit whose file does not validate counts as missing, as it does for
    ``Pipeline.ensure_fit`` and for ``results show``.
    """
    paths, recipe, manifest = _load(request.run_dir)
    store = load_fit_config(paths)
    follow_the_cache(paths, recipe, manifest, store)

    lanes: list[LaneStatus] = []
    n_fit = n_stale = n_failed = 0
    damaged: set[str] = set()  # keys of current fits whose cache file does not validate
    current: list[int] = []  # lanes whose fit can be used as it stands
    for lane in recipe.lanes.lanes:
        entry = LaneStatus(index=lane.index, label=lane.label)
        key = stage_key(StageName.FIT, lane.index)
        record = manifest.stages.get(key)
        if record is not None:
            result = None
            fit_file = paths.resolve(record.cache_path)
            if fit_file.exists():
                entry.has_fit = True
                entry.fit_current = _fit_is_current(record, recipe, store)
                if entry.fit_current:
                    result = _read_fit(fit_file)
                    if result is None:
                        entry.has_fit = entry.fit_current = False
                        damaged.add(key)
            if result is None:  # stale, missing or damaged: the recorded fit cannot be used
                n_stale += 1
            else:
                entry.status = result.status
                entry.n_bands = len(result.bands)
                n_fit += 1
                current.append(lane.index)
                if result.status == "failed":
                    n_failed += 1
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
        stale_stages=_stale_stages(paths, recipe, manifest, store, damaged),
        lanes=lanes,
        report=report_status(paths, report_fingerprint(recipe, manifest, current, store)),
        gelpy_version=code_version(),
    )


# --- helpers -----------------------------------------------------------------
def _require_background(recipe: Recipe, allow_raw: bool) -> None:
    """Refuse profiles and fits of a raw image unless the caller says it is deliberate."""
    if recipe.background is None and not allow_raw:
        raise BackgroundRequiredError(
            "No background removal is configured for this run, so profiles and fits would be "
            "measured on the raw image."
        )


def _read_fit(path: Path) -> LaneFitResult | None:
    """The cached fit in ``path``, or None when the file does not validate."""
    try:
        return read_fit_result(path)
    except Exception:  # a truncated or foreign file: ``Pipeline.ensure_fit`` calls it a miss too
        return None


def _stale_stages(
    paths: RunPaths,
    recipe: Recipe,
    manifest: RunManifest,
    store: FitConfigStore,
    damaged: set[str],
) -> list[str]:
    """Recorded stages that the next command needing them would recompute.

    A stage is stale when the ledger records it, it is not the stage of a lane the recipe no
    longer has, and the cache holds no artifact for the current recipe and running gelpy version
    (a damaged fit file counts as missing). The recorded hash is not consulted: the ledger lags
    the cache, because undoing an edit makes older artifacts current again while the ledger still
    names the newer ones, and only the cache says whether a command would compute anything. (A fit
    is judged by the same rule everywhere: ``follow_the_cache``.) A stage whose artifact was
    deleted stays listed until a command that needs it computes it again: a fit that is cached
    pulls on nothing upstream, so ``fit run`` leaves a missing image, background or profile
    listed. Records of a kind this release does not know, or a lane stage without a lane, are
    skipped. Lane stages are listed by kind, then by lane number.
    """
    lanes = {lane.index: lane for lane in recipe.lanes.lanes}
    stale: list[tuple[str, int, str]] = []  # (kind, lane index or -1, key)
    for key, record in manifest.stages.items():
        kind, lane_index = record.stage, -1
        if kind == StageName.IMAGE.value:
            artifact = paths.image_cache(stages.image_hash(recipe))
        elif kind == StageName.BACKGROUND.value:
            if recipe.background is None:
                continue  # nothing is computed for a run without background removal
            artifact = paths.background_cache(stages.background_hash(recipe))
        elif kind in (StageName.PROFILE.value, StageName.FIT.value):
            try:
                lane = lanes.get(_lane_index_of(key))
            except (IndexError, ValueError):
                continue
            if lane is None:
                continue  # a dropped lane is never computed again
            lane_index = lane.index
            if kind == StageName.PROFILE.value:
                artifact = paths.profile_cache(lane.index, stages.profile_hash(recipe, lane))
            elif lane.index in store.lanes:
                config = store.lanes[lane.index]
                fit_hash = stages.fit_hash(recipe, lane, config.profile, config.fit)
                artifact = paths.fit_cache(lane.index, fit_hash)
            elif record.gelpy_version == code_version():
                artifact = paths.resolve(record.cache_path)  # no config to recompute the hash from
            else:  # no config, and another gelpy version computed it
                stale.append((kind, lane_index, key))
                continue
        else:
            continue  # a kind this release does not know
        if not artifact.exists() or key in damaged:
            stale.append((kind, lane_index, key))
    return [key for _, _, key in sorted(stale)]


def _clear_derived(paths: RunPaths) -> list[str]:
    """Remove what a replaced run produced: results, the markdown and HTML reports, the renders.

    The content-addressed cache stays (an entry is keyed by its inputs, so it cannot be misread)
    and so does the append-only history log. A failure to remove something is raised (it reaches
    the caller as ``os_error``): ignoring it would report success while the old renders stayed
    reachable. A ``render/`` or ``report/`` that is a symlink is unlinked, never followed.

    ``render/`` is gelpy's by name and goes whole. ``report/`` is a name a person may use too:
    only the files the report's marker lists are removed, and a ``report/`` without a valid marker
    stays as it is (``_clear_report``). The lines it returns say what was removed and what was left.
    """
    for file in (paths.results_csv, paths.results_json, paths.report):
        file.unlink(missing_ok=True)
    said = _clear_report(paths.html_report)
    if paths.render.is_symlink():
        paths.render.unlink()
    elif paths.render.exists():
        shutil.rmtree(paths.render)
    return said


def _paths_in_words(names: list[str], limit: int) -> str:
    return ", ".join(names[:limit]) + (f" and {len(names) - limit} more" if len(names) > limit else "")


def _clear_report(folder: Path) -> list[str]:
    """``init --force`` and the HTML report folder; what it removed and left, as warnings."""
    if folder.is_symlink():
        folder.unlink()
        return [f"Unlinked {folder.name}: it was a link; what it pointed to was not touched."]
    if not folder.is_dir():
        return []
    cleared = clear_report(folder)
    if cleared is None:
        if is_empty(folder):
            return []
        return [
            f"Left {folder.name}/ as it is: it has no valid {REPORT_MARKER}, so report html did "
            "not write it."
        ]
    said = []
    leftovers = [name for name in cleared.removed if is_temp_name(name.rsplit("/", 1)[-1])]
    written = [name for name in cleared.removed if name not in leftovers]
    if written:
        n = len(written)
        said.append(
            f"Removed the HTML report from {folder.name}/: {n} file{'s' * (n != 1)} "
            f"({_paths_in_words(written, 5)})."
        )
    if leftovers:
        n = len(leftovers)
        said.append(
            f"Removed {n} temporary file{'s' * (n != 1)} a killed report html left in {folder.name}/ "
            f"({_paths_in_words(leftovers, 5)})."
        )
    if cleared.left:
        left = [f"{folder.name}/{name}" for name in cleared.left]
        said.append(f"Left {_paths_in_words(left, 10)}: not files report html wrote.")
    return said


def _build_layout(positions, labels, width, rotation) -> LaneLayout:
    lanes = [
        LaneDefinition(index=i, label=str(label), x_center_px=float(pos))
        for i, (pos, label) in enumerate(zip(positions, labels))
    ]
    return LaneLayout(lanes=lanes, profile_width_px=int(width), label_rotation=float(rotation))


def _resolve(recipe: Recipe, selector) -> list[LaneDefinition]:
    indices = resolve_lane_selector(selector, recipe.lanes)
    return [recipe.lanes.lanes[i] for i in indices]


def _result_files(paths: RunPaths) -> dict[str, str]:
    return {
        "results_csv": _rel(paths, paths.results_csv),
        "results_json": _rel(paths, paths.results_json),
    }


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
