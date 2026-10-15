"""CLI-side rendering orchestration.

Reads run-folder artifacts (recipe, caches) and calls the pure ``render.*`` functions.
Kept out of ``runfolder.commands`` so that layer stays matplotlib-free (and trivially
MCP-wrappable). Profile reads are cache hits, so plotting is cheap.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import get_args

from ..core.errors import InvalidValueError, shown
from ..core.line_profiles import apply_slice, normalize_profile
from ..core.text import decode_utf8
from ..schemas.profiles import LineProfileConfig
from ..schemas.report import ReportMarker
from ..schemas.requests import ImageKind, StatusRequest
from ..schemas.tools import HtmlReportResult
from ..runfolder.cache import write_json
from ..runfolder.commands import LaneDetectionOutcome
from ..runfolder.fit_config_io import load_fit_config
from ..runfolder.layout import RunPaths
from ..runfolder.lock import locked_run_dir
from ..runfolder.manifest import load_manifest
from ..runfolder.pipeline import Pipeline
from ..runfolder.recipe_io import load_recipe
from ..runfolder.report_folder import read_regular_file
from ..runfolder.results import follow_the_cache, read_fit_result
from ..runfolder.stages import StageName, stage_key
from ._io import run_path

#: The kinds `render_image` accepts; anything else would also become a file name.
IMAGE_KINDS: tuple[str, ...] = get_args(ImageKind)


def _selected(recipe, selector):
    from ..core.lanes import resolve_lane_selector

    return [recipe.lanes.lanes[i] for i in resolve_lane_selector(selector, recipe.lanes)]


@locked_run_dir
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

    Any other ``kind`` raises ``InvalidValueError`` listing the four: the kind is also the file
    name, so an unchecked one could write outside ``render/``.
    """
    if kind not in IMAGE_KINDS:
        raise InvalidValueError(
            f"Unknown image kind {kind!r}.",
            available=list(IMAGE_KINDS),
            detail="Use one of the kinds in `available`.",
        )

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


@locked_run_dir
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


def _plot_state(record, label: str) -> dict[str, str]:
    """What a fit plot shows: the fit (by hash) and the label the plot is titled with.

    The fit hash leaves the label out, so a plot of the current fit can still carry an old label.
    """
    return {"fit": record.hash, "label": label}


def _shown_fits(paths: RunPaths) -> dict[str, dict[str, str]]:
    """Lane index (as a string) -> what its plot shows (``_plot_state``)."""
    try:
        index = json.loads(paths.fit_plot_index.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return index if isinstance(index, dict) else {}


@locked_run_dir
def render_fits(run_dir: str, selector) -> list[Path]:
    """Render per-lane fit overlays, recomputing each lane's baseline from its stored config
    (the strategies fit the baseline-subtracted signal, so the overlay needs the baseline).

    Each plot is titled with the lane's current label. Records in ``render/fits/index.json``
    which fit each plot shows, and under which label, so ``ensure_fit_plots`` can tell a current
    plot from a stale one. The index never claims more than the plots show: a lane's entry is
    dropped before its plot is drawn and written once the plot is on disk (a plot is written
    atomically), so a draw that is interrupted leaves the lane unknown, and it is drawn again."""
    from .. import render
    from ..core.peaks.baseline import estimate_baseline

    paths = RunPaths(run_dir)
    recipe = load_recipe(paths)
    manifest = load_manifest(paths)
    pipeline = Pipeline(paths, recipe, manifest)
    store = load_fit_config(paths)
    follow_the_cache(paths, recipe, manifest, store)
    plots = _shown_fits(paths)

    outputs: list[Path] = []
    for lane in _selected(recipe, selector):
        record = manifest.stages.get(stage_key(StageName.FIT, lane.index))
        if record is None:
            continue
        fit_file = paths.resolve(record.cache_path)
        if not fit_file.exists():
            continue
        result = read_fit_result(fit_file)  # a file that does not validate is named, never quoted
        result = result.model_copy(update={"label": lane.label})  # the fit may hold an older one
        cfg = store.lanes.get(lane.index)
        pconf = cfg.profile if cfg else LineProfileConfig()
        full = pipeline.ensure_profile(lane)
        prepared = normalize_profile(
            apply_slice(full, pconf.slice_start, pconf.slice_end), pconf.normalization
        )
        baseline = estimate_baseline(prepared, cfg.fit.baseline) if cfg else prepared * 0.0
        if plots.pop(str(lane.index), None) is not None:
            write_json(paths.fit_plot_index, plots)  # while it is drawn, the old entry is not true
        outputs.append(
            render.render_lane_fit(prepared, baseline, result, paths.fit_render(lane.index))
        )
        plots[str(lane.index)] = _plot_state(record, lane.label)
        write_json(paths.fit_plot_index, plots)
    return outputs


@locked_run_dir
def ensure_fit_plots(run_dir: str, lane_indices: list[int]) -> list[Path]:
    """The fit plot of each lane, rendered again only where the plot on disk shows another fit
    or another label than the lane's current one."""
    paths = RunPaths(run_dir)
    recipe = load_recipe(paths)
    labels = {lane.index: lane.label for lane in recipe.lanes.lanes}
    manifest = load_manifest(paths)
    follow_the_cache(paths, recipe, manifest, load_fit_config(paths))
    on_disk = _shown_fits(paths)
    stale = []
    for index in lane_indices:
        record = manifest.stages.get(stage_key(StageName.FIT, index))
        if record is None:
            continue
        current = _plot_state(record, labels.get(index, ""))
        if not paths.fit_render(index).exists() or on_disk.get(str(index)) != current:
            stale.append(index)
    if stale:
        render_fits(run_dir, stale)
    return [paths.fit_render(index) for index in lane_indices]


@locked_run_dir
def write_report(run_dir: str) -> Path:
    """Regenerate report.md from the recipe + assembled results (stdlib templating)."""
    from ..render.report import render_report
    from ..runfolder.results import collect_results

    paths = RunPaths(run_dir)
    recipe = load_recipe(paths)
    manifest = load_manifest(paths)
    result = collect_results(paths, manifest, recipe, load_fit_config(paths))
    return render_report(recipe, result, paths.report)


#: What a stylesheet given with ``--css`` may weigh.
_STYLESHEET_LIMIT = 1 << 20


def _read_stylesheet(css_path: str) -> bytes:
    """The bytes of the stylesheet ``--css`` names: a regular file of at most 1 MiB that is UTF-8.

    The bytes are kept as they are, so that the copy is byte for byte; they are decoded only to
    name the line where the file stops being UTF-8.
    """
    data = read_regular_file(
        css_path,
        what="stylesheet",
        next_step="Pass the path of a UTF-8 stylesheet file of at most 1 MiB.",
        limit=_STYLESHEET_LIMIT,
    )
    decode_utf8(  # only to check it: the bytes are what is kept
        data,
        what=f"The stylesheet {shown(css_path)}",
        detail=f"Save {shown(css_path)} as UTF-8 text and repeat the command.",
    )
    return data


def _read_plot(png: Path) -> bytes:
    """The bytes of a plot gelpy drew, to copy into the report: a regular file, and not a link.

    A link there (a run folder can come from anyone) would copy what it points at into a folder
    that is made to be opened, so it is refused.
    """
    return read_regular_file(
        png,
        what="plot",
        next_step="Delete it and repeat the command: gelpy draws the plot again.",
        follow_links=False,
    )


def _png_size(data: bytes) -> tuple[int | None, int | None]:
    """Width and height of a PNG, from its header; (None, None) for anything else."""
    if data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
        width, height = struct.unpack(">II", data[16:24])
        return width, height
    return None, None


@locked_run_dir
def write_html_report(
    run_dir: str, *, out_dir: str | None = None, css_path: str | None = None
) -> HtmlReportResult:
    """Build the static HTML report (``render/html_report.py``) of a run folder.

    Renders the three gel views fresh (the lane overlay and the clean linear and gamma views),
    the all-lanes profile plot when background removal is configured, and every current fit's
    plot whose image shows another fit or label; reads them (a plot that is a link is refused),
    builds the page, and writes it with the images into ``<out>``, by default ``<run>/report``.

    The folder is claimed first (``runfolder.report_folder``): it must be new, empty, or carry the
    marker of an earlier report; it may be neither the run folder, a folder inside it other than
    ``report/``, a folder around it, nor one that holds a recipe.json or a manifest.json, whatever
    marker it carries; and ``report/`` may not be a link, as the default or typed as ``--out``.
    Anything else is refused before a plot is drawn. The page is built before a file is written,
    and a build that would write over an existing file its marker does not list is refused before
    it writes anything (``check_overwrites``); a marker with an empty fingerprint is written before
    the files, so that a build that stops half way is not "current"; the files the earlier build
    wrote and this one did not are removed afterwards, and so are the temp files of killed writes
    (best effort: one that cannot be removed stays); the final marker comes last.
    """
    from ..render.html_report import Figure, ReportContent, build_report, write_report
    from ..runfolder import report_folder
    from ..runfolder.commands import status as run_status
    from ..runfolder.results import collect_results
    from ..runfolder.stages import code_version

    paths = RunPaths(run_dir)
    if not paths.exists():
        raise FileNotFoundError(f"No run folder at {shown(run_dir)} (missing recipe.json)")
    theme = _read_stylesheet(css_path) if css_path else None
    target = Path(out_dir) if out_dir else paths.html_report
    earlier = report_folder.claim_folder(target, paths, chosen=bool(out_dir))

    images: dict[str, bytes] = {}

    def figure(png: Path, caption: str, alt: str, subdir: str = "") -> Figure:
        name = f"img/{subdir}/{png.name}" if subdir else f"img/{png.name}"
        images[name] = _read_plot(png)
        width, height = _png_size(images[name])
        return Figure(src=name, caption=caption, alt=alt, width=width, height=height)

    gel = [
        figure(
            render_image(run_dir, kind="non_linear", show_lanes=True),
            "Lane overlay: the profile strips gelpy measures, over the gamma-contrast view",
            "Gel image with the lane strips drawn over it",
        ),
        figure(
            render_image(run_dir, kind="linear"),
            "Gel, linear contrast",
            "Gel image under a linear intensity scale",
        ),
        figure(
            render_image(run_dir, kind="non_linear"),
            "Gel, gamma contrast",
            "Gel image under a gamma intensity scale",
        ),
    ]
    recipe = load_recipe(paths)
    profiles = []
    if recipe.background is not None:
        png = render_profiles(run_dir, "all", LineProfileConfig())
        profiles.append(
            figure(
                png,
                "Line profiles of all lanes, background-corrected, area-normalised",
                "Line profile of every lane, overlaid",
            )
        )
    status = run_status(StatusRequest(run_dir=run_dir))
    current = [lane.index for lane in status.lanes if lane.has_fit and lane.fit_current]
    labels = {lane.index: lane.label for lane in recipe.lanes.lanes}
    fits = [
        figure(
            png,
            f"Lane {index}: {labels[index]}",
            f"Fit of lane {index}: profile, baseline and bands",
            "fits",
        )
        for index, png in zip(current, ensure_fit_plots(run_dir, current))
    ]
    store = load_fit_config(paths)
    manifest = load_manifest(paths)
    damaged: list[int] = []
    result = collect_results(paths, manifest, recipe, store, damaged=damaged)  # follows the cache
    fingerprint = report_folder.report_fingerprint(recipe, manifest, current, store)
    content = ReportContent(
        run_name=Path(os.path.abspath(run_dir)).name,
        gelpy_version=code_version(),
        recipe=recipe,
        status=status,
        result=result,
        fit_config=store,
        gel=gel,
        profiles=profiles,
        fits=fits,
        fingerprint=fingerprint,
        damaged=damaged,
    )
    files = build_report(content, theme_css=theme, images=images)  # nothing is written if this fails
    names = list(files)
    report_folder.check_destinations(target, names)
    report_folder.check_overwrites(target, names, earlier)
    version = code_version()
    planned = sorted({*names, *(earlier.files if earlier else [])})
    report_folder.write_marker(target, ReportMarker(gelpy_version=version, fingerprint="", files=planned))
    index = write_report(files, target)
    report_folder.drop_stale(target, earlier, names)
    report_folder.sweep_debris(target)
    report_folder.write_marker(
        target, ReportMarker(gelpy_version=version, fingerprint=fingerprint, files=sorted(names))
    )
    return HtmlReportResult(index=run_path(run_dir, index), files=sorted(names))


@locked_run_dir
def render_lane_overlay(run_dir: str, outcome: LaneDetectionOutcome) -> Path:
    """Render the lane-detection verification overlay from a detection that already ran
    (``commands.run_lane_detection``): no second detection, no second image load. The row
    bounds come from the outcome too, so the overlay cannot disagree with its detection."""
    from .. import render

    detection = outcome.detection
    return render.render_lane_detection(
        outcome.image,
        detection.projection,
        detection.peak_indices,
        detection.centers,
        RunPaths(run_dir).render / "lane_detect.png",
        y_start=outcome.y_start,
        y_end=outcome.y_end,
        y_band=detection.y_band,
    )


@locked_run_dir
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
