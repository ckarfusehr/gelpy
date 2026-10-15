"""The ``gelpy`` command-line application.

Each command is a thin adapter: parse argv, build a request, run the tool of the same step
(``cli/tools.py``: the command and its rendering, under one lock), emit the result. A command
that touches a run folder takes the run directory as an explicit positional argument (``gelpy
fit run RUN_DIR ...``) -- there is no ambient "current run" state, so an agent's shell history
is self-contained and replayable. ``schema``, ``models list`` and ``tool list`` need no run
folder; ``tool call`` takes it in its JSON; ``mcp`` serves the same tools over MCP.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from typer.core import TyperGroup

from ..core.errors import InvalidValueError, RegistryError, shown, to_error_response
from ..core.text import decode_utf8
from ..runfolder import commands
from ..runfolder.lock import run_lock
from ..runfolder.stages import code_version
from ..schemas.baseline import BaselineConfig
from ..schemas.fit_config import FitConfig
from ..schemas.profiles import LineProfileConfig
from ..schemas.requests import (
    DetectLanesRequest,
    ExtractProfilesRequest,
    FitConfigOverrides,
    FitRunRequest,
    GetResultsRequest,
    HtmlReportRequest,
    InitRunRequest,
    LineProfileConfigOverrides,
    RefitLaneRequest,
    RemoveBackgroundRequest,
    RenderImageRequest,
    SetContrastRequest,
    SetImageRequest,
    SetLanesRequest,
    StatusRequest,
)
from . import _io, _render, tools


class _StructuredErrors(TyperGroup):
    """The root command group: a failure leaves as one ``ErrorResponse`` on stderr.

    Commands answer their own failures (``_io.handle_errors``). What reaches this level is a
    malformed command line, which click would print as a usage panel, or an error outside every
    command body. Both get the same JSON shape; a malformed command line keeps exit code 2.
    """

    def main(self, args=None, prog_name=None, complete_var=None, standalone_mode=True, **extra):
        if not standalone_mode:
            return super().main(
                args=args, prog_name=prog_name, complete_var=complete_var,
                standalone_mode=False, **extra,
            )
        try:
            code = super().main(
                args=args, prog_name=prog_name, complete_var=complete_var,
                standalone_mode=False, **extra,
            )
        except _io.USAGE_ERROR as exc:
            _io.emit_error(_io.usage_error_response(exc))
            sys.exit(2)
        except typer.Abort:
            sys.exit(1)
        except Exception as exc:  # outside every command body; still one structured error
            _io.emit_error(to_error_response(exc))
            sys.exit(1)
        sys.exit(code if isinstance(code, int) else 0)


app = typer.Typer(
    cls=_StructuredErrors,
    help="gelpy — gel electrophoresis image analysis (agent-native CLI).",
    no_args_is_help=True,
)
lanes_app = typer.Typer(help="Configure lane positions and labels.", no_args_is_help=True)
contrast_app = typer.Typer(help="Configure display contrast.", no_args_is_help=True)
background_app = typer.Typer(help="Configure and inspect background removal.", no_args_is_help=True)
image_app = typer.Typer(help="Render gel images.", no_args_is_help=True)
profiles_app = typer.Typer(help="Extract and plot line profiles.", no_args_is_help=True)
fit_app = typer.Typer(help="Fit peaks to line profiles.", no_args_is_help=True)
results_app = typer.Typer(help="Read assembled results.", no_args_is_help=True)
report_app = typer.Typer(help="Write reports of a run folder.", no_args_is_help=True)
models_app = typer.Typer(help="Discover available models.", no_args_is_help=True)
tool_app = typer.Typer(
    help="Discover and call the agent tools: one JSON object in, one JSON object out.",
    no_args_is_help=True,
)

app.add_typer(lanes_app, name="lanes")
app.add_typer(contrast_app, name="contrast")
app.add_typer(background_app, name="background")
app.add_typer(image_app, name="image")
app.add_typer(profiles_app, name="profiles")
app.add_typer(fit_app, name="fit")
app.add_typer(results_app, name="results")
app.add_typer(report_app, name="report")
app.add_typer(models_app, name="models")
app.add_typer(tool_app, name="tool")


def _print_version(value: bool) -> None:
    if value:
        typer.echo(f"gelpy {code_version()}")
        raise typer.Exit()


@app.callback()
def _root(
    version: bool = typer.Option(
        False, "--version", callback=_print_version, is_eager=True,
        help="Print the gelpy version and exit.",
    ),
) -> None:
    """gelpy — gel electrophoresis image analysis (agent-native CLI)."""


def _read_labels(labels_file: Optional[str]) -> Optional[list[str]]:
    if labels_file is None:
        return None
    text = decode_utf8(
        Path(labels_file).read_bytes(),
        what=f"The labels file {labels_file!r}",
        detail=f"Save {labels_file!r} as UTF-8 text, one label per line, and repeat the command.",
        bom=True,  # reads plain UTF-8 too, and drops the byte order mark Notepad and Excel write
    )
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


# --- init --------------------------------------------------------------------
@app.command()
@_io.handle_errors
def init(
    run_dir: str = typer.Argument(..., help="Run folder to create."),
    image: str = typer.Argument(..., help="Path to the source gel image."),
    n_lanes: int = typer.Option(10, help="Number of lanes for the default layout."),
    width: Optional[int] = typer.Option(None, help="Profile strip width (px); guessed if omitted."),
    labels_file: Optional[str] = typer.Option(None, help="Newline-separated labels file."),
    channel: str = typer.Option(
        "auto", help="Channel->grayscale: auto|mean|luminance|r|g|b|dominant|max."
    ),
    invert: Optional[bool] = typer.Option(
        None, "--invert/--no-invert", help="Force polarity; omit to auto-detect."
    ),
    page: int = typer.Option(0, help="Page index for a multi-page source."),
    force: bool = typer.Option(
        False,
        "--force",
        help="Replace the run if the folder already holds one (results, report and renders are "
        "removed; cache and history are kept).",
    ),
    fmt: str = typer.Option("json", "--format", help="Output format: json|text."),
):
    """Create a run folder with a default lane layout."""
    request = InitRunRequest(
        run_dir=run_dir,
        image_path=image,
        n_lanes=n_lanes,
        width=width,
        labels=_read_labels(labels_file),
        channel=channel,
        invert=invert,
        page=page,
        force=force,
    )
    _io.emit(tools.run_tool("init", request), fmt)


# --- status ------------------------------------------------------------------
@app.command()
@_io.handle_errors
def status(
    run_dir: str = typer.Argument(..., help="Existing run folder."),
    fmt: str = typer.Option("json", "--format", help="Output format: json|text."),
):
    """Report a run folder's state: source, background, and per-lane fit/stale status."""
    _io.emit(tools.run_tool("status", StatusRequest(run_dir=run_dir)), fmt)


# --- lanes -------------------------------------------------------------------
@lanes_app.command("set")
@_io.handle_errors
def lanes_set(
    run_dir: str = typer.Argument(..., help="Existing run folder."),
    linspace: Optional[tuple[float, float, int]] = typer.Option(
        None, help="left right n — evenly spaced lane centers."
    ),
    endpoints: Optional[tuple[float, float]] = typer.Option(
        None, help="left right — outer lane centers; linspaced with --n-lanes (manual fallback)."
    ),
    x_centers: Optional[list[float]] = typer.Option(
        None, "--x-center", help="Explicit lane center x-position (repeatable)."
    ),
    n_lanes: Optional[int] = typer.Option(None, help="Lane count for --endpoints."),
    labels_file: Optional[str] = typer.Option(None, help="Newline-separated labels file."),
    width: Optional[int] = typer.Option(None, help="Profile strip width (px)."),
    label_rotation: Optional[float] = typer.Option(None, help="Label rotation (deg)."),
    fmt: str = typer.Option("json", "--format", help="Output format: json|text."),
):
    """Set lane positions / labels / width."""
    request = SetLanesRequest(
        run_dir=run_dir,
        linspace=linspace,
        endpoints=endpoints,
        x_centers=x_centers,
        n_lanes=n_lanes,
        labels=_read_labels(labels_file),
        width=width,
        label_rotation=label_rotation,
    )
    _io.emit(tools.run_tool("lanes_set", request), fmt)


@lanes_app.command("detect")
@_io.handle_errors
def lanes_detect(
    run_dir: str = typer.Argument(..., help="Existing run folder."),
    n_lanes: Optional[int] = typer.Option(
        None, help="True lane count (ground truth); defaults to the current recipe count."
    ),
    y_start: Optional[int] = typer.Option(None, help="Top row of the projection band."),
    y_end: Optional[int] = typer.Option(
        None, help="Bottom row of the projection band (try the well region for faint lanes)."
    ),
    smoothing_window: Optional[float] = typer.Option(
        None,
        help="Moving-window average width (px) applied before peak detection (default: "
        "0.7x the expected pitch). Wells are plateaus, often brighter on one edge than the "
        "other — averaging over a considerable width prevents that from biasing the center.",
    ),
    prominence: Optional[float] = typer.Option(
        None, "--prominence", help="Peak prominence floor; raise to drop spurious peaks."
    ),
    distance: Optional[float] = typer.Option(None, "--distance", help="Min px between peaks."),
    min_width: Optional[float] = typer.Option(
        None,
        "--min-width",
        help="Minimum peak width in px (default: off). No universally safe scaled default exists "
        "(a real narrow band and a dust spike can be equally narrow) — compare widths in the "
        "overlay first, then set this above the spurious ones but below the genuine ones.",
    ),
    corrected: bool = typer.Option(
        False, "--corrected", help="Detect on the background-corrected image (default: raw)."
    ),
    projection_strategy: str = typer.Option(
        "d4_union",
        "--projection-strategy",
        help="Row-aggregation strategy for the projection ('d4_union' = multi-band union, default; "
        "'full' = full-height mean). The default is only adopted if it beats full-height, so it is "
        "never worse than 'full'.",
    ),
    apply: bool = typer.Option(False, "--apply", help="Persist the detected centers to the recipe."),
    plot: bool = typer.Option(True, "--plot/--no-plot", help="Render the verification overlay."),
    labels_file: Optional[str] = typer.Option(None, help="Labels to apply (with --apply)."),
    label_rotation: Optional[float] = typer.Option(None, help="Label rotation (deg)."),
    fmt: str = typer.Option("json", "--format", help="Output format: json|text."),
):
    """Detect lane centers from the image (fit an even N-lane grid) and render an overlay to verify."""
    request = DetectLanesRequest(
        run_dir=run_dir,
        n_lanes=n_lanes,
        y_start=y_start,
        y_end=y_end,
        smoothing_window=smoothing_window,
        min_prominence=prominence,
        min_distance=distance,
        min_width=min_width,
        use_corrected=corrected,
        projection_strategy=projection_strategy,
        apply=apply,
        labels=_read_labels(labels_file),
        label_rotation=label_rotation,
        plot=plot,
    )
    _io.emit(tools.run_tool("lanes_detect", request), fmt)


@lanes_app.command("pick")
@_io.handle_errors
def lanes_pick(
    run_dir: str = typer.Argument(..., help="Existing run folder."),
    n_lanes: Optional[int] = typer.Option(
        None, help="True lane count (ground truth); defaults to the current recipe count."
    ),
    all_lanes: bool = typer.Option(
        False, "--all", help="Click every lane center instead of just the two outer ones."
    ),
    corrected: bool = typer.Option(
        False, "--corrected", help="Show the background-corrected image (default: raw)."
    ),
    labels_file: Optional[str] = typer.Option(None, help="Newline-separated labels file."),
    width: Optional[int] = typer.Option(None, help="Profile strip width (px)."),
    fmt: str = typer.Option("json", "--format", help="Output format: json|text."),
):
    """Interactively click lane centers on the image — opt-in human fallback for `lanes
    detect`. Needs a GUI/display, not for headless/agent use. Click the two outer lane
    centers (or, with --all, every lane) on the high-contrast image, then press Enter;
    with just the two outer clicks the rest are spread evenly (same as `lanes set
    --endpoints`)."""
    from ..runfolder.layout import RunPaths
    from ..runfolder.manifest import load_manifest
    from ..runfolder.pipeline import Pipeline
    from ..runfolder.recipe_io import load_recipe
    from . import _pick

    paths = RunPaths(run_dir)
    with run_lock(run_dir):  # held across the picking; commands.set_lanes re-enters it
        recipe = load_recipe(paths)
        manifest = load_manifest(paths)
        n = n_lanes or len(recipe.lanes.lanes)
        if n < 1:
            raise InvalidValueError(
                "n_lanes required (no existing lanes to infer the count from)",
                detail="Pass --n-lanes.",
            )
        # the cap on the count is the request's: refuse it before an image is loaded or a window opens
        SetLanesRequest(run_dir=run_dir, n_lanes=n)

        pipeline = Pipeline(paths, recipe, manifest)
        image = pipeline.ensure_background_image() if corrected else pipeline.ensure_image()

        n_clicks = n if all_lanes else 2
        points = _pick.pick_lane_centers(
            image, recipe.contrast, n_clicks, title=f"{run_dir} — {n} lanes"
        )

        result = commands.set_lanes(
            SetLanesRequest(
                run_dir=run_dir,
                endpoints=None if all_lanes else (points[0], points[-1]),
                x_centers=points if all_lanes else None,
                n_lanes=None if all_lanes else n,
                labels=_read_labels(labels_file),
                width=width,
            )
        )
        _io.emit(result, fmt)


# --- contrast ----------------------------------------------------------------
@contrast_app.command("set")
@_io.handle_errors
def contrast_set(
    run_dir: str = typer.Argument(..., help="Existing run folder."),
    gamma: Optional[float] = typer.Option(None),
    gain: Optional[float] = typer.Option(None),
    intensity_low: Optional[float] = typer.Option(None),
    intensity_high: Optional[float] = typer.Option(None),
    intensity_mode: Optional[str] = typer.Option(None, help="percentile|absolute"),
    fmt: str = typer.Option("json", "--format", help="Output format: json|text."),
):
    """Adjust display contrast (cosmetic only)."""
    request = SetContrastRequest(
        run_dir=run_dir,
        gamma=gamma,
        gain=gain,
        intensity_low=intensity_low,
        intensity_high=intensity_high,
        intensity_mode=intensity_mode,
    )
    _io.emit(tools.run_tool("contrast_set", request), fmt)


# --- background --------------------------------------------------------------
def _coerce_param(value: str):
    """Coerce a CLI 'key=value' value to int/float/bool where it clearly is one."""
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


@background_app.command("remove")
@_io.handle_errors
def background_remove(
    run_dir: str = typer.Argument(..., help="Existing run folder."),
    model: str = typer.Option("robust_plane", help="Background model name (see `models list`)."),
    order: int = typer.Option(None, "--order", help="robust_plane surface order (0/1/2)."),
    mode: str = typer.Option(
        None, "--mode", help="robust_plane weight mode: 'probabilistic' or 'asymmetric'."
    ),
    clip_negative: bool = typer.Option(
        None,
        "--clip-negative/--no-clip-negative",
        help="Clamp negatives to 0 (default: keep signed residuals — no rectification bias).",
    ),
    param: list[str] = typer.Option(
        None, "--param", help="Extra model param 'key=value' (repeatable)."
    ),
    stripe: list[str] = typer.Option(
        None, "--stripe", help="plane_ransac stripe 'pos,height' (repeatable)."
    ),
    plot: bool = typer.Option(False, help="Also render the background diagnostic figure."),
    fmt: str = typer.Option("json", "--format", help="Output format: json|text."),
):
    """Configure background removal (applied lazily by the pipeline)."""
    params: dict = {}
    if order is not None:
        params["order"] = order
    if mode is not None:
        params["weight_mode"] = mode
    if clip_negative is not None:
        params["clip_negative"] = clip_negative
    for kv in param or []:
        key, _, val = kv.partition("=")
        params[key.strip()] = _coerce_param(val.strip())
    stripes = None
    if stripe:
        stripes = [tuple(int(x) for x in s.split(",")) for s in stripe]
    request = RemoveBackgroundRequest(
        run_dir=run_dir, model=model, params=params, stripes=stripes, plot=plot
    )
    _io.emit(tools.run_tool("background_remove", request), fmt)


@background_app.command("show")
@_io.handle_errors
def background_show(
    run_dir: str = typer.Argument(..., help="Existing run folder."),
    fmt: str = typer.Option("json", "--format", help="Output format: json|text."),
):
    """Render the background diagnostic figure."""
    path = _render.render_background(run_dir)
    # Run-relative, like every other emitted path: one base for the whole JSON surface.
    _io.emit(
        {"background_diagnostic": _io.run_relative(run_dir, path) if path else None}, fmt
    )


# --- image -------------------------------------------------------------------
@image_app.command("show")
@_io.handle_errors
def image_show(
    run_dir: str = typer.Argument(..., help="Existing run folder."),
    kind: str = typer.Option("corrected", help="raw|linear|non_linear|corrected"),
    lanes: bool = typer.Option(
        False, "--lanes", help="Overlay semi-transparent red lane-width markers."
    ),
    fmt: str = typer.Option("json", "--format", help="Output format: json|text."),
):
    """Render the gel image to a PNG."""
    request = RenderImageRequest(run_dir=run_dir, kind=kind, lanes=lanes)
    _io.emit(tools.run_tool("image_render", request), fmt)


@image_app.command("set")
@_io.handle_errors
def image_set(
    run_dir: str = typer.Argument(..., help="Existing run folder."),
    channel: Optional[str] = typer.Option(None, help="New channel strategy."),
    invert: Optional[bool] = typer.Option(
        None, "--invert/--no-invert", help="Force polarity."
    ),
    page: Optional[int] = typer.Option(None, help="New page index."),
    auto: bool = typer.Option(False, "--auto", help="Re-run polarity auto-detection."),
    fmt: str = typer.Option("json", "--format", help="Output format: json|text."),
):
    """Re-interpret the source image (polarity/channel/page) of an existing run."""
    request = SetImageRequest(run_dir=run_dir, channel=channel, invert=invert, page=page, auto=auto)
    _io.emit(tools.run_tool("image_set", request), fmt)


# --- profiles ----------------------------------------------------------------
@profiles_app.command("extract")
@_io.handle_errors
def profiles_extract(
    run_dir: str = typer.Argument(..., help="Existing run folder."),
    lane: list[str] = typer.Option(None, "--lane", help="Lane selector term (repeatable)."),
    normalization: str = typer.Option("area", help="area|min_max|safe (for the plot)."),
    slice_: Optional[tuple[int, int]] = typer.Option(None, "--slice", help="start end."),
    plot: bool = typer.Option(False, help="Render the profiles overview PNG."),
    allow_raw: bool = typer.Option(
        False, "--allow-raw",
        help="Measure the raw image although no background removal is configured "
        "(refused otherwise: background_required).",
    ),
    fmt: str = typer.Option("json", "--format", help="Output format: json|text."),
):
    """Extract (and cache) the raw line profiles for a lane selection."""
    request = ExtractProfilesRequest(
        run_dir=run_dir,
        lanes=_io.parse_lane_selector(lane),
        plot=plot,
        profile=_profile_config(normalization, slice_),
        allow_raw=allow_raw,
    )
    _io.emit(tools.run_tool("profiles_extract", request), fmt)


# --- fit ---------------------------------------------------------------------
def _build_fit_config(method, shapes, baseline, poly_order, seed) -> FitConfig:
    kwargs: dict = {"method": method, "seed": seed}
    if shapes:
        kwargs["shapes"] = [s.strip() for s in shapes.split(",") if s.strip()]
    if baseline or poly_order is not None:
        baseline_kwargs: dict = {"method": baseline or "poly"}
        if poly_order is not None:
            baseline_kwargs["poly_order"] = poly_order
        kwargs["baseline"] = BaselineConfig(**baseline_kwargs)
    return FitConfig(**kwargs)


@fit_app.command("run")
@_io.handle_errors
def fit_run(
    run_dir: str = typer.Argument(..., help="Existing run folder."),
    lane: list[str] = typer.Option(None, "--lane", help="Lane selector term (repeatable)."),
    method: str = typer.Option("grow_select", help="Fit strategy: grow_select|sparse_seed|bayesian."),
    slice_: Optional[tuple[int, int]] = typer.Option(None, "--slice", help="start end."),
    normalization: str = typer.Option("area", help="area|min_max|safe."),
    shapes: Optional[str] = typer.Option(None, help="Comma-separated allowed shapes (default: all)."),
    baseline: Optional[str] = typer.Option(None, help="Baseline: poly|arpls|asls|none (default poly)."),
    poly_order: Optional[int] = typer.Option(
        None, "--poly-order", help="Polynomial baseline order (default 2). Use 0 (constant) when "
        "fitting a slice that starts right after a hard exclusion boundary (e.g. past a well-cavity "
        "dip) — order >=1 has enough freedom to bend up and absorb a real smear once there's no "
        "anchoring feature left in the window to hold it down."
    ),
    seed: int = typer.Option(0, help="RNG seed (reproducible bayesian tier)."),
    auto_escalate: bool = typer.Option(False, "--auto-escalate", help="Climb tiers on low confidence."),
    plot: bool = typer.Option(False),
    allow_raw: bool = typer.Option(
        False, "--allow-raw",
        help="Measure the raw image although no background removal is configured "
        "(refused otherwise: background_required).",
    ),
    fmt: str = typer.Option("json", "--format", help="Output format: json|text."),
):
    """Run a full fit pass over a lane selection (no manual peak-detection tuning needed)."""
    selector = _io.parse_lane_selector(lane)
    request = FitRunRequest(
        run_dir=run_dir,
        lanes=selector,
        profile=_profile_config(normalization, slice_),
        fit=_build_fit_config(method, shapes, baseline, poly_order, seed),
        auto_escalate=auto_escalate,
        plot=plot,
        allow_raw=allow_raw,
    )
    _io.emit(tools.run_tool("fit_run", request), fmt)


@fit_app.command("refit-lane")
@_io.handle_errors
def fit_refit_lane(
    run_dir: str = typer.Argument(..., help="Existing run folder."),
    lane: str = typer.Option(..., "--lane", help="Single lane selector term."),
    method: Optional[str] = typer.Option(None, help="Fit strategy to switch to."),
    slice_: Optional[tuple[int, int]] = typer.Option(None, "--slice"),
    normalization: Optional[str] = typer.Option(None),
    shapes: Optional[str] = typer.Option(None, help="Comma-separated allowed shapes."),
    baseline: Optional[str] = typer.Option(None, help="Baseline method to switch to."),
    poly_order: Optional[int] = typer.Option(
        None, "--poly-order", help="Polynomial baseline order (default 2; see `fit run --help`)."
    ),
    seed: Optional[int] = typer.Option(None, help="RNG seed."),
    auto_escalate: bool = typer.Option(False, "--auto-escalate", help="Climb tiers on low confidence."),
    plot: bool = typer.Option(False),
    allow_raw: bool = typer.Option(
        False, "--allow-raw",
        help="Measure the raw image although no background removal is configured "
        "(refused otherwise: background_required).",
    ),
    fmt: str = typer.Option("json", "--format", help="Output format: json|text."),
):
    """Refit one lane, changing only the options you pass (merged onto its last settings)."""
    baseline_override = None
    if baseline or poly_order is not None:
        baseline_kwargs: dict = {"method": baseline or "poly"}
        if poly_order is not None:
            baseline_kwargs["poly_order"] = poly_order
        baseline_override = BaselineConfig(**baseline_kwargs)
    request = RefitLaneRequest(
        run_dir=run_dir,
        lane=_io.parse_lane_term(lane),
        fit_overrides=FitConfigOverrides(
            method=method,
            shapes=[s.strip() for s in shapes.split(",")] if shapes else None,
            baseline=baseline_override,
            seed=seed,
        ),
        profile_overrides=LineProfileConfigOverrides(
            normalization=normalization,
            slice_start=slice_[0] if slice_ else None,
            slice_end=slice_[1] if slice_ else None,
        ),
        auto_escalate=auto_escalate,
        plot=plot,
        allow_raw=allow_raw,
    )
    _io.emit(tools.run_tool("fit_refit_lane", request), fmt)


# --- results -----------------------------------------------------------------
@results_app.command("show")
@_io.handle_errors
def results_show(
    run_dir: str = typer.Argument(..., help="Existing run folder."),
    fmt: str = typer.Option("json", "--format", help="json|csv|table"),
):
    """Show assembled results (from cache, no recompute)."""
    result = tools.run_tool("results_get", GetResultsRequest(run_dir=run_dir))
    if fmt in ("csv", "table"):  # json carries `warnings` itself; here stdout stays the data
        for warning in result.warnings:
            typer.echo(f"warning: {warning}", err=True)
    if fmt == "csv":
        # Rendered from the live assembly, NOT read back from results.csv: that file is
        # written by a fit pass, so after a recipe edit staled a lane it would still list
        # rows that the json/table formats of this same command correctly drop.
        from ..runfolder.results import results_csv_text

        typer.echo(results_csv_text(result), nl=False)
    elif fmt == "table":
        for row in result.to_rows():
            if row.get("band_number") is None:
                typer.echo(
                    f"{row['lane_index']:>4}  {row['label']:<16}  {row['status']} ({row['confidence']})"
                )
            else:
                typer.echo(
                    f"{row['lane_index']:>4}  {row['label']:<16}  band {row['band_number']} "
                    f"{row['component_class']:<7}  frac={row['area_fraction']:.3f}  "
                    f"x={row['maximum_position']:.1f}"
                )
    else:
        _io.emit(result, "json")


# --- report ------------------------------------------------------------------
@report_app.command("html")
@_io.handle_errors
def report_html(
    run_dir: str = typer.Argument(..., help="Existing run folder."),
    css: Optional[str] = typer.Option(
        None, "--css",
        help="Path of a UTF-8 stylesheet file (at most 1 MiB), copied unparsed as theme.css and "
        "loaded after the default; its references are yours (a relative url() resolves against "
        "the report folder).",
    ),
    out: Optional[str] = typer.Option(
        None, "--out",
        help="Folder to write the report into (default: RUN_DIR/report). It must be new, empty or "
        "hold the output of an earlier report html: any other non-empty folder is refused, and so "
        "are the run folder and the folders inside or around it (but report/). A rebuild replaces "
        "only the files that earlier build wrote. One report per folder.",
    ),
    fmt: str = typer.Option("json", "--format", help="Output format: json|text."),
):
    """Write a static HTML report of the run folder: one page, a stylesheet and its images, no scripts."""
    request = HtmlReportRequest(run_dir=run_dir, css=css, out_dir=out)
    _io.emit(tools.run_tool("report_html", request), fmt)


# --- discovery ---------------------------------------------------------------
@models_app.command("list")
@_io.handle_errors
def models_list(
    kind: str = typer.Option("fit", help="fit|shape|background"),
    fmt: str = typer.Option("json", "--format", help="Output format: json|text."),
):
    """List available models, their descriptions, and their parameter schemas.

    ``fit`` = fit strategies (grow_select/sparse_seed/bayesian); ``shape`` = peak shapes
    (gaussian/emg/bi_gaussian/pseudo_voigt); ``background`` = background models.
    """
    if kind == "fit":
        from ..core.peaks.strategies import fit_strategy_registry as reg
    elif kind == "shape":
        from ..core.peaks import peak_registry as reg
    elif kind == "background":
        from ..core.background import background_registry as reg
    else:
        raise RegistryError(
            "kind must be 'fit', 'shape', or 'background'", available=["fit", "shape", "background"]
        )
    _io.emit(
        {
            "names": reg.names(),
            "descriptions": reg.descriptions(),
            "schemas": reg.json_schemas(),
        },
        fmt,
    )


# --- tools -------------------------------------------------------------------
@tool_app.command("list")
@_io.handle_errors
def tool_list(
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Print every tool's name, title, description, input and output JSON Schema, "
        "annotations and path arguments.",
    ),
):
    """List the agent tools (`gelpy tool call NAME` runs one)."""
    listing = tools.tool_list()
    if as_json:
        _io.emit(listing, "json")
        return
    for entry in listing.tools:
        typer.echo(f"{entry.name:<18} {entry.title}")


def _read_tool_input(input_path: str, ctx: Optional[typer.Context] = None) -> str:
    """The JSON text of a tool call, from a file or from stdin ('-').

    It is UTF-8, with or without the byte order mark that Notepad and PowerShell write. Input
    that is not UTF-8 says which input it is; a closed stdin (the program that ran the command
    closed it with ``<&-``) is a usage error, not a bug.
    """
    if input_path == "-":
        if sys.stdin is None:
            raise _io.USAGE_ERROR(
                "stdin is closed, so there is no JSON object to read: pipe it into the command "
                "or pass --input FILE.",
                ctx=ctx,
            )
        data, source = typer.get_binary_stream("stdin").read(), "stdin"
    else:
        data, source = Path(input_path).read_bytes(), f"the file {shown(input_path)}"
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise InvalidValueError(
            f"The tool input ({source}) must be UTF-8 JSON, but byte {exc.start} is not valid UTF-8.",
            detail="Write the JSON object as UTF-8 text and repeat the call.",
        ) from exc


@tool_app.command("call")
@_io.handle_errors
def tool_call(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Tool name (see `gelpy tool list`)."),
    input_path: str = typer.Option(
        "-",
        "--input",
        help="File holding the JSON object (UTF-8; a byte order mark is ignored); '-' (the "
        "default) reads stdin.",
    ),
):
    """Run one tool: read one JSON object, write one JSON object (an error on stderr)."""
    tools.get_tool(name)  # a mistyped name fails now, not after stdin has been read
    _io.emit(tools.call_tool_text(name, _read_tool_input(input_path, ctx)), "json")


def _public_schema_models() -> dict[str, type]:
    """Every pydantic contract exported from ``gelpy.schemas`` (name -> class)."""
    from pydantic import BaseModel

    from .. import schemas

    out: dict[str, type] = {}
    for name in schemas.__all__:
        obj = getattr(schemas, name)
        if isinstance(obj, type) and issubclass(obj, BaseModel):
            out[name] = obj
    return out


@app.command()
@_io.handle_errors
def schema(model: str = typer.Argument(..., help="Model name, e.g. RefitLaneRequest.")):
    """Print the JSON schema of a gelpy data contract (for discovery)."""
    models = _public_schema_models()
    if model not in models:
        raise RegistryError(f"Unknown model {model!r}.", available=sorted(models))
    _io.emit(models[model].model_json_schema(), "json")


# --- mcp ---------------------------------------------------------------------
@app.command("mcp")
@_io.handle_errors
def mcp_command():
    """Serve the agent tools over the Model Context Protocol on stdin/stdout.

    Needs the optional dependency: pip install "gelpy[mcp]". Register it with an MCP client,
    for example `claude mcp add gelpy -- gelpy mcp`.
    """
    from . import mcp_server

    mcp_server.serve()


def _profile_config(normalization: str, slice_) -> LineProfileConfig:
    if slice_ is not None:
        return LineProfileConfig(
            normalization=normalization, slice_start=slice_[0], slice_end=slice_[1]
        )
    return LineProfileConfig(normalization=normalization)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
