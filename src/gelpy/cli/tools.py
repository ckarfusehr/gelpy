"""The agent tool layer: gelpy's workflow steps as tools that read one JSON object and write one.

`gelpy tool list --json` describes them, `gelpy tool call NAME` runs one, `gelpy mcp` serves
them over MCP, and the typer commands in ``app.py`` run the same handlers, so the front-ends
cannot disagree about what a step does. A tool is one CLI step: its request model is the
contract (``gelpy schema``); its handler calls ``runfolder.commands`` and renders what the CLI
step renders, holding the run-folder lock for the whole step. Tool names are public API: they
change only with a major version.

Not tools: ``lanes pick`` (needs a display), ``background show`` (``background_remove`` with
``plot`` renders the same figure), and the discovery commands ``schema`` and ``models list``
(the tool schemas carry the registry names; a model's parameters and descriptions come from
``gelpy models list`` and ``gelpy schema``).
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from ..core.errors import NON_DOMAIN_CODES, RegistryError, shown, to_error_response
from ..runfolder import commands
from ..runfolder.lock import locked
from ..runfolder.stages import code_version
from ..schemas.common import CanonicalModel, ErrorResponse
from ..schemas.lanes import LaneDetectionResult
from ..schemas.requests import (
    DetectLanesRequest,
    ExtractProfilesRequest,
    FitRunRequest,
    GetResultsRequest,
    HtmlReportRequest,
    InitRunRequest,
    RefitLaneRequest,
    RemoveBackgroundRequest,
    RenderImageRequest,
    SetContrastRequest,
    SetImageRequest,
    SetLanesRequest,
    StatusRequest,
)
from ..schemas.results import AnalysisResult
from ..schemas.run import RunStatus, RunUpdateSummary
from ..schemas.tools import (
    HtmlReportResult,
    ImageRenderResult,
    PathRole,
    ToolAnnotations,
    ToolList,
    ToolSpec,
)
from . import _io, _render


@dataclass(frozen=True)
class Tool:
    """One workflow step as a tool."""

    name: str
    title: str
    description: str
    request: type[CanonicalModel]
    result: type[BaseModel]
    handler: Callable[[Any], BaseModel]
    read_only: bool = False
    destructive: bool = False
    idempotent: bool = True
    paths: Mapping[str, PathRole] = field(default_factory=lambda: {"run_dir": "run"})


def _whole_step(handler: Callable[[Any], BaseModel]) -> Callable[[Any], BaseModel]:
    """Hold the run-folder lock across the whole step (the command and its renders).

    ``locked`` marks what it returns, which ``takes_run_lock`` reads: a test then sees a handler
    whose later steps (the report, the plots) would run outside the lock, which a held lock alone
    does not show, because its first step locks itself.
    """
    return locked(handler)


# --- handlers: the command plus the rendering its CLI step does ------------------------------
def _lanes_detect(request: DetectLanesRequest) -> LaneDetectionResult:
    outcome = commands.run_lane_detection(request)
    if not request.plot:
        return outcome.result
    png = _render.render_lane_overlay(request.run_dir, outcome)
    return outcome.result.model_copy(update={"overlay": _io.run_relative(request.run_dir, png)})


def _background_remove(request: RemoveBackgroundRequest) -> RunUpdateSummary:
    result = commands.remove_background(request)
    if request.plot:
        path = _render.render_background(request.run_dir)
        if path is not None:
            result.files["background_diagnostic"] = _io.run_relative(request.run_dir, path)
    return result


def _image_render(request: RenderImageRequest) -> ImageRenderResult:
    path = _render.render_image(request.run_dir, kind=request.kind, show_lanes=request.lanes)
    return ImageRenderResult(image=_io.run_relative(request.run_dir, path))


def _profiles_extract(request: ExtractProfilesRequest) -> RunUpdateSummary:
    result = commands.extract_profiles(request)
    if request.plot:
        path = _render.render_profiles(request.run_dir, request.lanes, request.profile)
        result.files["profiles_overview"] = _io.run_relative(request.run_dir, path)
    return result


def _fit_run(request: FitRunRequest) -> RunUpdateSummary:
    result = commands.fit_run(request)
    report = _render.write_report(request.run_dir)
    result.files["report"] = _io.run_relative(request.run_dir, report)
    if request.plot:
        _render.render_fits(request.run_dir, request.lanes)
    return result


def _fit_refit_lane(request: RefitLaneRequest) -> RunUpdateSummary:
    result = commands.refit_lane(request)
    report = _render.write_report(request.run_dir)
    result.files["report"] = _io.run_relative(request.run_dir, report)
    if request.plot:
        plots = _render.render_fits(request.run_dir, [request.lane])
        if plots:
            result.files["plot"] = _io.run_relative(request.run_dir, plots[0])
    return result


def _report_html(request: HtmlReportRequest) -> HtmlReportResult:
    return _render.write_html_report(request.run_dir, out_dir=request.out_dir, css_path=request.css)


_TOOL_LIST: tuple[Tool, ...] = (
    Tool(
        "init",
        "Create a run folder",
        "Create a run folder for one gel image with a default lane layout. The lane count and "
        "the labels come from the user's sample sheet, never from guessing. Refuses a folder "
        "that already holds a run unless force=true.",
        InitRunRequest,
        RunUpdateSummary,
        commands.init_run,  # locks itself, creating the folder
        destructive=True,
        idempotent=False,
        paths={"run_dir": "run", "image_path": "input"},
    ),
    Tool(
        "status",
        "Read a run folder's state",
        "The state of a run folder without recomputing anything: image, background, lanes, "
        "which lanes have a current or a stale fit, and stale_stages. Call it first on a run "
        "folder you did not just create.",
        StatusRequest,
        RunStatus,
        commands.status,
        read_only=True,
    ),
    Tool(
        "image_set",
        "Re-interpret the source image",
        "Change how the source image is read: polarity, colour channel, page. Stages computed "
        "from the old interpretation are recomputed on next use.",
        SetImageRequest,
        RunUpdateSummary,
        _whole_step(commands.set_image),
        destructive=True,
    ),
    Tool(
        "contrast_set",
        "Set the display contrast",
        "Set the display contrast (gamma, gain, linear intensity range) of the rendered gel "
        "views. Cosmetic: it never changes a measurement.",
        SetContrastRequest,
        RunUpdateSummary,
        _whole_step(commands.set_contrast),
        destructive=True,
    ),
    Tool(
        "lanes_detect",
        "Detect lane centres",
        "Fit an even grid of n_lanes lane centres to the image and render the verification "
        "overlay (render/lane_detect.png). Look at the overlay before any measurement: every "
        "line must sit on a lane. apply=true replaces the run's lane layout with the result. "
        "On confidence 'low' read confidence_reason, then adjust the detection or set the outer "
        "lane centres with lanes_set (endpoints).",
        DetectLanesRequest,
        LaneDetectionResult,
        _whole_step(_lanes_detect),
        destructive=True,
    ),
    Tool(
        "lanes_set",
        "Set lane positions, labels or width",
        "Set the lane layout explicitly: the outer lane centres (endpoints) with a lane count, "
        "evenly spaced centres (linspace) or every centre (x_centers); and/or the labels "
        "(verbatim, left to right) and the profile strip width.",
        SetLanesRequest,
        RunUpdateSummary,
        _whole_step(commands.set_lanes),
        destructive=True,
    ),
    Tool(
        "background_remove",
        "Configure background removal",
        "Choose and configure the background model; the default robust_plane needs no "
        "parameters. Profiles and fits require it unless allow_raw=true. plot=true also renders "
        "the background diagnostic.",
        RemoveBackgroundRequest,
        RunUpdateSummary,
        _whole_step(_background_remove),
        destructive=True,
    ),
    Tool(
        "image_render",
        "Render a gel view",
        "Render the gel image to a PNG: kind raw, linear, non_linear or corrected. lanes=true "
        "draws the lane strips: a QC view, never the sample view.",
        RenderImageRequest,
        ImageRenderResult,
        _whole_step(_image_render),
    ),
    Tool(
        "profiles_extract",
        "Extract lane profiles",
        "Extract and cache the selected lanes' line profiles; plot=true renders them overlaid "
        "(the slice and normalization in profile apply to the plot only).",
        ExtractProfilesRequest,
        RunUpdateSummary,
        _whole_step(_profiles_extract),
    ),
    Tool(
        "fit_run",
        "Fit bands in a lane selection",
        "Fit every selected lane with one fit configuration and write results.csv, results.json "
        "and report.md. Read each lane's confidence and reasons: status 'ok' only means the fit "
        "converged. plot=true renders render/fits/lane_NNN.png per lane.",
        FitRunRequest,
        RunUpdateSummary,
        _whole_step(_fit_run),
        destructive=True,
    ),
    Tool(
        "fit_refit_lane",
        "Refit one lane",
        "Refit one lane, changing only the fields given in fit_overrides and profile_overrides; "
        "everything else keeps that lane's last-used settings.",
        RefitLaneRequest,
        RunUpdateSummary,
        _whole_step(_fit_refit_lane),
        destructive=True,
    ),
    Tool(
        "results_get",
        "Read the results",
        "The current fit results of every lane, assembled from the cache with stale fits left "
        "out, without recomputing anything.",
        GetResultsRequest,
        AnalysisResult,
        commands.get_results,
        read_only=True,
    ),
    Tool(
        "report_html",
        "Write the HTML report",
        "Write a static HTML report of the run (index.html, a stylesheet and images; no "
        "scripts): the lane overlay, clean gel views, the lane and results tables, profiles, "
        "fits and provenance. css is the path of a UTF-8 stylesheet file, copied unparsed (its "
        "references are yours); it overrides the --gelpy-* CSS variables. out_dir must be new, "
        "empty or an earlier report's folder, and the tool replaces only the files that report "
        "held; any other non-empty folder is refused, and so are the run folder and the folders "
        "inside or around it. `status` says whether the report is current.",
        HtmlReportRequest,
        HtmlReportResult,
        _whole_step(_report_html),
        paths={"run_dir": "run", "css": "input", "out_dir": "output"},
    ),
)

#: Every tool by name, in workflow order. The names are public API (module docstring).
TOOLS: dict[str, Tool] = {tool.name: tool for tool in _TOOL_LIST}


# --- schemas ---------------------------------------------------------------------------------
def _registry_enums() -> dict[tuple[str, str], list[str]]:
    """(model, field) -> the names that field accepts, read from the registries."""
    from ..core.background import background_registry
    from ..core.lanes import projection_strategies
    from ..core.peaks import peak_registry
    from ..core.peaks.strategies import fit_strategy_registry

    fits, shapes = fit_strategy_registry.names(), peak_registry.names()
    return {
        ("RemoveBackgroundRequest", "model"): background_registry.names(),
        ("DetectLanesRequest", "projection_strategy"): projection_strategies.names(),
        ("FitConfig", "method"): fits,
        ("FitConfig", "shapes"): shapes,
        ("FitConfigOverrides", "method"): fits,
        ("FitConfigOverrides", "shapes"): shapes,
    }


def _constrain(prop: dict[str, Any], names: list[str]) -> None:
    """Constrain a string field -- or a list of strings, or an optional of either -- to ``names``."""
    if prop.get("type") == "string":
        prop["enum"] = list(names)
    elif prop.get("type") == "array" and prop.get("items", {}).get("type") == "string":
        prop["items"]["enum"] = list(names)
    for branch in prop.get("anyOf", []):
        _constrain(branch, names)


def input_schema(request: type[BaseModel]) -> dict[str, Any]:
    """The request model's JSON Schema with registry names as enums.

    The schemas package cannot import the registries (they live in ``core``), so the names are
    filled in here, at the top of the layering.
    """
    schema = copy.deepcopy(request.model_json_schema())
    enums = _registry_enums()
    models = [(schema.get("title", ""), schema), *schema.get("$defs", {}).items()]
    for model_name, model_schema in models:
        for field_name, prop in model_schema.get("properties", {}).items():
            names = enums.get((model_name, field_name))
            if names is not None:
                _constrain(prop, names)
    return schema


def spec(tool: Tool) -> ToolSpec:
    """The published description of one tool."""
    return ToolSpec(
        name=tool.name,
        title=tool.title,
        description=tool.description,
        input_schema=input_schema(tool.request),
        output_schema=tool.result.model_json_schema(mode="serialization"),
        annotations=ToolAnnotations(
            read_only_hint=tool.read_only,
            destructive_hint=tool.destructive,
            idempotent_hint=tool.idempotent,
        ),
        paths=dict(tool.paths),
    )


def tool_list() -> ToolList:
    """What `gelpy tool list --json` prints."""
    return ToolList(gelpy_version=code_version(), tools=[spec(tool) for tool in TOOLS.values()])


# --- calling ---------------------------------------------------------------------------------
def get_tool(name: str) -> Tool:
    try:
        return TOOLS[name]
    except (KeyError, TypeError):  # TypeError: a name that cannot be a key at all (a list, a dict)
        raise RegistryError(
            f"Unknown tool {shown(name)}.",
            available=sorted(TOOLS),
            detail="Use one of the names in `available`; `gelpy tool list` describes them.",
        ) from None


def run_tool(name: str, request: BaseModel) -> BaseModel:
    """Run a tool on an already validated request (the typer commands' path)."""
    return get_tool(name).handler(request)


def call_tool_text(name: str, text: str) -> BaseModel:
    """`gelpy tool call`: validate JSON text against the tool's request model, then run it."""
    tool = get_tool(name)
    return tool.handler(tool.request.model_validate_json(text))


def call_tool(name: str, arguments: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate ``arguments``, run the tool, return its result as JSON-ready data. Raises.

    ``arguments`` is validated as it comes (``None`` is no arguments): a list of pairs or a
    generator is not an object, and is refused rather than made one. The result goes through
    ``model_dump_json`` and back, as ``gelpy tool call`` prints it, so a number that is not finite
    is ``null`` here as it is there.
    """
    tool = get_tool(name)
    request = tool.request.model_validate({} if arguments is None else arguments)
    return json.loads(tool.handler(request).model_dump_json())


def _undescribable() -> dict[str, Any]:
    """The answer when a failure cannot even be described: fixed, so nothing in it can fail."""
    return ErrorResponse(
        error="The failure could not be described.",
        code="internal_error",
        detail=NON_DOMAIN_CODES["internal_error"][1],
    ).model_dump(mode="json")


def call_tool_safely(
    name: str, arguments: Mapping[str, Any] | None
) -> tuple[bool, dict[str, Any]]:
    """``(True, result)`` or ``(False, the ErrorResponse as a dict)``; never raises."""
    try:
        return True, call_tool(name, arguments)
    except Exception as exc:  # every failure becomes one structured error
        try:
            return False, to_error_response(exc).model_dump(mode="json")
        except Exception:  # describing it failed too: an object whose str() raises
            return False, _undescribable()
