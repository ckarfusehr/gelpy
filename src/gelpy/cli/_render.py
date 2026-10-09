"""CLI-side rendering orchestration.

Reads run-folder artifacts (recipe, caches) and calls the pure ``render.*`` functions.
Kept out of ``runfolder.commands`` so that layer stays matplotlib-free (and trivially
MCP-wrappable). Profile reads are cache hits, so plotting is cheap.
"""

from __future__ import annotations

from pathlib import Path

from ..core.line_profiles import apply_slice, normalize_profile
from ..schemas.peaks import LaneFitResult
from ..schemas.profiles import LineProfileConfig
from ..runfolder.layout import RunPaths
from ..runfolder.manifest import load_manifest
from ..runfolder.pipeline import Pipeline
from ..runfolder.recipe_io import load_recipe
from ..runfolder.stages import StageName, stage_key


def _selected(recipe, selector):
    from ..core.lanes import resolve_lane_selector

    return [recipe.lanes.lanes[i] for i in resolve_lane_selector(selector, recipe.lanes)]


def render_image(run_dir: str, kind: str = "corrected", show_lanes: bool = False) -> Path:
    """Render the gel image. ``kind``: raw|linear|non_linear|corrected.

    The four kinds differ in *which* image and *which* display contrast:
    ``raw`` = the canonical source array with no contrast applied (dark for most
    fluorescence gels; this is the one that shows what canonicalization and polarity
    detection actually handed the pipeline); ``linear``/``non_linear`` = the same source
    array under the recipe's linear range or gamma; ``corrected`` = the
    background-corrected array under gamma.

    ``show_lanes`` draws the semi-transparent red lane-width overlay; off by default so the
    plain view shows the sample unobstructed. The output filename gets a ``_lanes`` suffix
    when the overlay is on, so it never clobbers the plain render of the same ``kind``.
    """
    from .. import render

    paths = RunPaths(run_dir)
    recipe = load_recipe(paths)
    manifest = load_manifest(paths)
    pipeline = Pipeline(paths, recipe, manifest)

    if kind == "corrected":
        image = pipeline.ensure_background_image()
        mode = "non_linear"
    else:
        image = pipeline.ensure_image()
        mode = kind if kind in ("raw", "linear", "non_linear") else "non_linear"

    suffix = "_lanes" if show_lanes else ""
    out = paths.render / f"{kind}{suffix}.png"
    return render.render_gel_image(
        image,
        out,
        contrast=recipe.contrast,
        layout=recipe.lanes,
        mode=mode,
        title=kind,
        show_lanes=show_lanes,
    )


def render_profiles(run_dir: str, selector, profile_config: LineProfileConfig) -> Path:
    from .. import render

    paths = RunPaths(run_dir)
    recipe = load_recipe(paths)
    manifest = load_manifest(paths)
    pipeline = Pipeline(paths, recipe, manifest)

    lanes = _selected(recipe, selector)
    profiles = []
    for lane in lanes:
        full = pipeline.ensure_profile(lane)
        prepared = normalize_profile(
            apply_slice(full, profile_config.slice_start, profile_config.slice_end),
            profile_config.normalization,
        )
        profiles.append((lane.label, prepared))

    # Naming a selection's plot after its first lane lets callers render one PNG per
    # sample group (e.g. a sample + its fractions) without filenames colliding.
    if selector == "all":
        name = "profiles_overview.png"
    else:
        safe_label = "".join(c if c.isalnum() else "_" for c in lanes[0].label)
        name = f"profiles_{safe_label}.png"
    out = paths.render / name
    return render.render_profiles_overview(
        profiles,
        out,
        slice_start=profile_config.slice_start,
        normalization=profile_config.normalization,
    )


def render_fits(run_dir: str, selector) -> list[Path]:
    """Render per-lane fit overlays, recomputing each lane's baseline from its stored config
    (the strategies fit the baseline-subtracted signal, so the overlay needs the baseline)."""
    from .. import render
    from ..core.peaks.baseline import estimate_baseline
    from ..runfolder.fit_config_io import load_fit_config

    paths = RunPaths(run_dir)
    recipe = load_recipe(paths)
    manifest = load_manifest(paths)
    pipeline = Pipeline(paths, recipe, manifest)
    store = load_fit_config(paths)

    outputs: list[Path] = []
    for lane in _selected(recipe, selector):
        record = manifest.stages.get(stage_key(StageName.FIT, lane.index))
        if record is None or not Path(record.cache_path).exists():
            continue
        result = LaneFitResult.model_validate_json(Path(record.cache_path).read_text())
        cfg = store.lanes.get(lane.index)
        pconf = cfg.profile if cfg else LineProfileConfig()
        full = pipeline.ensure_profile(lane)
        prepared = normalize_profile(
            apply_slice(full, pconf.slice_start, pconf.slice_end), pconf.normalization
        )
        baseline = estimate_baseline(prepared, cfg.fit.baseline) if cfg else prepared * 0.0
        outputs.append(
            render.render_lane_fit(prepared, baseline, result, paths.fit_render(lane.index))
        )
    return outputs


def write_report(run_dir: str) -> Path:
    """Regenerate report.md from the recipe + assembled results (stdlib templating)."""
    from ..render.report import render_report
    from ..runfolder.fit_config_io import load_fit_config
    from ..runfolder.results import collect_results

    paths = RunPaths(run_dir)
    recipe = load_recipe(paths)
    manifest = load_manifest(paths)
    result = collect_results(manifest, recipe, load_fit_config(paths))
    return render_report(recipe, result, paths.report)


def render_lanes(
    run_dir: str,
    *,
    n_lanes: int | None = None,
    use_corrected: bool = False,
    y_start: int | None = None,
    y_end: int | None = None,
    smoothing_window: float | None = None,
    min_prominence: float | None = None,
    min_distance: float | None = None,
    min_width: float | None = None,
    projection_strategy: str = "d4_union",
) -> Path:
    """Render the lane-detection verification overlay.

    Recomputes the (cheap, pure) detection on the loaded image — mirroring
    ``render_background`` / ``render_fits`` — rather than routing the projection array
    through the command result. Uses the same knobs as ``commands.detect_lanes`` so the
    overlay reflects exactly what that command produced.
    """
    from .. import render
    from ..core.lanes import detect_lane_centers

    paths = RunPaths(run_dir)
    recipe = load_recipe(paths)
    manifest = load_manifest(paths)
    pipeline = Pipeline(paths, recipe, manifest)
    image = pipeline.ensure_background_image() if use_corrected else pipeline.ensure_image()

    n = n_lanes or len(recipe.lanes.lanes)
    detection = detect_lane_centers(
        image,
        n,
        y_start=y_start,
        y_end=y_end,
        smoothing_window=smoothing_window,
        min_prominence=min_prominence,
        min_distance=min_distance,
        min_width=min_width,
        projection_strategy=projection_strategy,
    )
    out = paths.render / "lane_detect.png"
    return render.render_lane_detection(
        image,
        detection.projection,
        detection.peak_indices,
        detection.centers,
        out,
        y_start=y_start,
        y_end=y_end,
        y_band=detection.y_band,
    )


def render_background(run_dir: str) -> Path | None:
    from .. import render
    from ..core.background import background_registry

    paths = RunPaths(run_dir)
    recipe = load_recipe(paths)
    if recipe.background is None:
        return None
    manifest = load_manifest(paths)
    pipeline = Pipeline(paths, recipe, manifest)
    image = pipeline.ensure_image()
    out = paths.render / "background_diagnostic.png"

    # plane_ransac keeps its stripe-overlay diagnostic; every other (fitted-surface) model
    # is fit via the registry to expose its surface + per-pixel weight map.
    if recipe.background.model == "plane_ransac":
        from ..core.background.plane_ransac import default_stripes, fit_plane

        stripes = recipe.background.params.get("stripes") or default_stripes(image.shape[0])
        _, plane, _ = fit_plane(image, [tuple(s) for s in stripes])
        return render.render_background_diagnostic(image, plane, [tuple(s) for s in stripes], out)

    model = background_registry.get(recipe.background.model)()
    config_model = background_registry.config_model(recipe.background.model)
    params = (
        config_model.model_validate(recipe.background.params)
        if config_model is not None
        else recipe.background.params
    )
    result = model.remove(image, params)
    return render.render_background_surface(
        image, result.background_plane, result.weights, result.corrected_image, out
    )
