"""The tool layer: one registry of workflow steps behind `tool list`, `tool call` and the CLI."""

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gelpy.cli import tools
from gelpy.cli.app import app
from gelpy.core.background import background_registry
from gelpy.core.lanes import projection_strategies
from gelpy.core.peaks import peak_registry
from gelpy.core.peaks.strategies import fit_strategy_registry
from gelpy.runfolder.lock import takes_run_lock

runner = CliRunner()

PUBLIC_TOOLS = [
    "background_remove",
    "contrast_set",
    "fit_refit_lane",
    "fit_run",
    "image_render",
    "image_set",
    "init",
    "lanes_detect",
    "lanes_set",
    "profiles_extract",
    "report_html",
    "results_get",
    "status",
]


def _call(name, arguments):
    result = runner.invoke(app, ["tool", "call", name], input=json.dumps(arguments))
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def _fail(name, text):
    result = runner.invoke(app, ["tool", "call", name], input=text)
    assert result.exit_code == 1, result.output
    return json.loads(result.stderr)


def _prepared(gel_image, run_dir):
    _call("init", {"run_dir": run_dir, "image_path": gel_image, "n_lanes": 4, "width": 30})
    _call("lanes_set", {"run_dir": run_dir, "linspace": [40, 250, 4]})
    _call("background_remove", {"run_dir": run_dir, "model": "plane_ransac"})
    return run_dir


def test_the_tool_names_are_a_stable_public_set():
    """Tool names are public API: adding one is a minor change, renaming or removing one a major."""
    assert sorted(tools.TOOLS) == PUBLIC_TOOLS


def test_tool_list_json_describes_every_tool():
    listing = json.loads(runner.invoke(app, ["tool", "list", "--json"]).stdout)
    assert listing["gelpy_version"]
    for entry in listing["tools"]:
        assert set(entry) == {
            "name", "title", "description", "input_schema", "output_schema", "annotations", "paths",
        }
        assert entry["input_schema"]["type"] == "object"
        assert entry["output_schema"]["type"] == "object"
        assert entry["paths"]["run_dir"] == "run"
        assert entry["annotations"]["open_world_hint"] is False
    by_name = {entry["name"]: entry for entry in listing["tools"]}
    assert by_name["init"]["paths"] == {"run_dir": "run", "image_path": "input"}
    assert by_name["report_html"]["paths"] == {"run_dir": "run", "css": "input", "out_dir": "output"}


def test_tool_list_without_json_prints_one_line_per_tool():
    lines = runner.invoke(app, ["tool", "list"]).stdout.strip().splitlines()
    assert [line.split()[0] for line in lines] == list(tools.TOOLS)


def test_registry_names_are_enums_in_the_input_schemas():
    fit = tools.input_schema(tools.TOOLS["fit_run"].request)["$defs"]["FitConfig"]["properties"]
    assert fit["method"]["enum"] == fit_strategy_registry.names()
    assert fit["shapes"]["items"]["enum"] == peak_registry.names()
    refit = tools.input_schema(tools.TOOLS["fit_refit_lane"].request)["$defs"]["FitConfigOverrides"]
    assert refit["properties"]["method"]["anyOf"][0]["enum"] == fit_strategy_registry.names()
    background = tools.input_schema(tools.TOOLS["background_remove"].request)["properties"]
    assert background["model"]["enum"] == background_registry.names()
    detect = tools.input_schema(tools.TOOLS["lanes_detect"].request)["properties"]
    assert detect["projection_strategy"]["enum"] == projection_strategies.names()


def test_annotations():
    read_only = {name for name, tool in tools.TOOLS.items() if tool.read_only}
    harmless = {name for name, tool in tools.TOOLS.items() if not tool.destructive}
    repeatable = {name for name, tool in tools.TOOLS.items() if tool.idempotent}
    assert read_only == {"status", "results_get"}
    assert harmless == {"status", "results_get", "image_render", "profiles_extract", "report_html"}
    assert set(tools.TOOLS) - repeatable == {"init"}


def test_a_workflow_through_tool_call(gel_image, run_dir):
    _prepared(gel_image, run_dir)
    image = _call("image_render", {"run_dir": run_dir, "kind": "linear"})
    assert (Path(run_dir) / image["image"]).exists()
    fit = _call("fit_run", {"run_dir": run_dir, "lanes": [0, 1]})
    assert fit["files"]["report"] == "report.md"
    results = _call("results_get", {"run_dir": run_dir})
    assert [lane["lane_index"] for lane in results["lanes"]] == [0, 1]
    assert _call("status", {"run_dir": run_dir})["n_fit"] == 2


def test_report_html_through_tool_call(gel_image, run_dir, tmp_path):
    _prepared(gel_image, run_dir)
    inside = _call("report_html", {"run_dir": run_dir})
    assert inside["index"] == "report/index.html"
    assert (Path(run_dir) / inside["index"]).is_file()
    theme = tmp_path / "theme-source.css"
    theme.write_text(":root { --gelpy-bg: #fafafa; }\n", encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    outside = _call("report_html", {"run_dir": run_dir, "css": str(theme), "out_dir": str(elsewhere)})
    assert Path(outside["index"]) == elsewhere / "index.html"
    assert {"index.html", "report.css", "theme.css"} <= set(outside["files"])
    assert (elsewhere / "theme.css").read_text(encoding="utf-8") == theme.read_text(encoding="utf-8")


def test_the_cli_and_the_tool_give_the_same_answer(gel_image, tmp_path):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    for run in (a, b):
        _call("init", {"run_dir": run, "image_path": gel_image, "n_lanes": 4})
    by_cli = runner.invoke(app, ["lanes", "set", a, "--endpoints", "40", "250", "--n-lanes", "4"])
    by_tool = _call("lanes_set", {"run_dir": b, "endpoints": [40, 250], "n_lanes": 4})
    assert json.loads(by_cli.stdout) == by_tool

    def lanes(run):
        return json.loads((Path(run) / "recipe.json").read_text(encoding="utf-8"))["lanes"]

    assert lanes(a) == lanes(b)


def test_lanes_detect_reports_its_overlay(gel_image, run_dir):
    _call("init", {"run_dir": run_dir, "image_path": gel_image, "n_lanes": 4})
    _call("background_remove", {"run_dir": run_dir, "model": "plane_ransac"})
    args = {"run_dir": run_dir, "n_lanes": 4, "use_corrected": True}
    detected = _call("lanes_detect", {**args, "apply": True})
    assert detected["confidence"] == "ok"
    assert (Path(run_dir) / detected["overlay"]).exists()
    assert _call("lanes_detect", {**args, "plot": False})["overlay"] is None


def test_refit_reports_a_plot_only_when_it_rendered_one(gel_image, run_dir):
    _prepared(gel_image, run_dir)
    _call("fit_run", {"run_dir": run_dir, "lanes": [0]})
    quiet = _call("fit_refit_lane", {"run_dir": run_dir, "lane": 0, "fit_overrides": {"seed": 1}})
    assert "plot" not in quiet["files"]
    plotted = _call("fit_refit_lane", {"run_dir": run_dir, "lane": 0, "plot": True})
    assert (Path(run_dir) / plotted["files"]["plot"]).exists()


def test_tool_call_reads_an_input_file(gel_image, run_dir, tmp_path):
    arguments = tmp_path / "arguments.json"
    arguments.write_text(
        json.dumps({"run_dir": run_dir, "image_path": gel_image, "n_lanes": 3}), encoding="utf-8"
    )
    result = runner.invoke(app, ["tool", "call", "init", "--input", str(arguments)])
    assert result.exit_code == 0, result.output


def test_an_unknown_tool_lists_the_tools():
    error = _fail("nope", "{}")
    assert error["code"] == "registry"
    assert error["available"] == PUBLIC_TOOLS
    assert "gelpy tool list" in error["detail"]


def test_malformed_json_is_a_validation_error():
    error = _fail("status", "{not json")
    assert error["code"] == "validation_error"
    assert json.loads(error["detail"])[0]["type"] == "json_invalid"


def test_an_unknown_argument_is_rejected(run_dir):
    assert _fail("status", json.dumps({"run_dir": run_dir, "run": "typo"}))["code"] == (
        "validation_error"
    )


def test_text_output_of_lanes_detect_is_json(gel_image, run_dir):
    _call("init", {"run_dir": run_dir, "image_path": gel_image, "n_lanes": 4})
    out = runner.invoke(app, ["lanes", "detect", run_dir, "--no-plot", "--format", "text"])
    assert json.loads(out.stdout)["n_expected"] == 4


# --- beyond the contract above: argument errors, the one entry point of a server, the lock ------------
def test_lanes_detect_on_the_command_line_reports_no_overlay_without_plot(gel_image, run_dir):
    _call("init", {"run_dir": run_dir, "image_path": gel_image, "n_lanes": 4})
    out = runner.invoke(app, ["lanes", "detect", run_dir, "--no-plot"])
    assert out.exit_code == 0, out.output
    assert json.loads(out.stdout)["overlay"] is None


def test_a_bad_normalization_is_refused_with_or_without_plot(gel_image, run_dir):
    """The option travels as ExtractProfilesRequest.profile, so it is validated even when only
    the plot would read it; the error lists the allowed values."""
    _prepared(gel_image, run_dir)
    result = runner.invoke(
        app, ["profiles", "extract", run_dir, "--normalization", "bogus", "--no-plot"]
    )
    assert result.exit_code == 1, result.output
    error = json.loads(result.stderr)
    assert error["code"] == "validation_error"
    assert error["available"] == ["area", "min_max", "safe"]


def test_a_tool_lists_the_allowed_values_of_an_argument_and_cuts_a_huge_one(run_dir):
    huge = "k" * 1_000_000
    arguments = {"run_dir": run_dir, "kind": huge, "lanes": "maybe"}
    result = runner.invoke(app, ["tool", "call", "image_render"], input=json.dumps(arguments))
    assert result.exit_code == 1
    assert len(result.stderr) < 2000  # not a megabyte
    error = json.loads(result.stderr)
    assert error["code"] == "validation_error"
    assert error["available"] == ["raw", "linear", "non_linear", "corrected"]
    fields = json.loads(error["detail"])
    assert [field["loc"] for field in fields] == [["kind"], ["lanes"]]  # the whole list is there
    assert fields[0]["input"] == "k" * 200 + "... [cut: 1000000 characters in all]"
    assert fields[1]["input"] == "maybe"  # a short input is echoed whole


def test_call_tool_safely_answers_every_outcome_and_never_raises(gel_image, run_dir):
    _prepared(gel_image, run_dir)
    ok, result = tools.call_tool_safely("status", {"run_dir": run_dir})
    assert ok and result["n_lanes"] == 4
    ok, error = tools.call_tool_safely("nope", {})
    assert (ok, error["code"]) == (False, "registry")
    ok, error = tools.call_tool_safely("status", {"run_dir": 5})
    assert (ok, error["code"]) == (False, "validation_error")
    ok, error = tools.call_tool_safely("status", None)  # no arguments: run_dir is missing
    assert (ok, error["code"]) == (False, "validation_error")
    ok, error = tools.call_tool_safely("status", {"run_dir": run_dir + "-missing"})
    assert (ok, error["code"]) == (False, "not_found")


HOLDER = (
    "import sys, time\n"
    "from gelpy.runfolder.lock import run_lock\n"
    "with run_lock(sys.argv[1]):\n"
    "    print('held', flush=True)\n"
    "    time.sleep(60)\n"
)

#: Request fields a tool needs besides run_dir (and image_path).
EXTRA_ARGUMENTS = {"fit_refit_lane": {"lane": 0}}


@pytest.fixture
def held_elsewhere(gel_image, run_dir):
    """A run folder whose lock another process holds for the whole test."""
    _call("init", {"run_dir": run_dir, "image_path": gel_image, "n_lanes": 4, "width": 30})
    holder = subprocess.Popen(
        [sys.executable, "-c", HOLDER, run_dir], stdout=subprocess.PIPE, text=True
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        yield run_dir
    finally:
        holder.kill()
        holder.wait()
        holder.stdout.close()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file locks")
@pytest.mark.parametrize("name", sorted(tools.TOOLS))
def test_a_held_lock_refuses_every_tool_that_writes_and_no_tool_that_reads(
    held_elsewhere, gel_image, name
):
    """A tool added later is checked without anyone editing this file: it fails here until its
    handler holds the run lock (a read-only tool does not take it)."""
    tool = tools.TOOLS[name]
    arguments = {"run_dir": held_elsewhere, **EXTRA_ARGUMENTS.get(name, {})}
    if "image_path" in tool.request.model_fields:
        arguments["image_path"] = gel_image
    ok, payload = tools.call_tool_safely(name, arguments)
    if tool.read_only:
        assert ok, payload
    else:
        assert (ok, payload["code"]) == (False, "run_locked")


# --- the one entry point of a server answers anything it is given ------------------------------------
FIELDS = {"error", "code", "detail", "available"}


class _Boom:
    """An object that cannot be turned into text."""

    def __str__(self):
        raise RuntimeError("str boom")

    __repr__ = __str__


def _deep_list(depth):
    value = []
    for _ in range(depth):
        value = [value]
    return value


#: Arguments that are not a JSON object, or hold what cannot be shown. Factories, because a generator
#: is used up by one call.
ARGUMENTS = {
    "none": lambda: None,
    "a list": lambda: [1, 2],
    "a list of pairs": lambda: [["run_dir", "x"]],
    "an int": lambda: 5,
    "a bool": lambda: True,
    "bytes": lambda: b"bytes",
    "a generator": lambda: (pair for pair in [("run_dir", "x")]),
    "a huge string": lambda: "x" * 5_000_000,
    "deep nesting": lambda: _deep_list(100_000),
    "a value that cannot be text": lambda: {"run_dir": _Boom()},
    "a key that cannot be text": lambda: {_Boom(): 1, "run_dir": "x"},
}


@pytest.mark.parametrize("make", ARGUMENTS.values(), ids=list(ARGUMENTS))
def test_call_tool_safely_never_raises_and_refuses_what_is_not_an_object(make):
    ok, payload = tools.call_tool_safely("status", make())
    assert ok is False and set(payload) == FIELDS
    assert payload["code"] == "validation_error", payload  # not internal_error, not coerced into a call
    assert len(json.dumps(payload)) < 2000


@pytest.mark.parametrize(
    "name",
    [["status"], {"status": 1}, {1}, 5, b"status", _Boom(), "s" * 1_000_000],
    ids=["list", "dict", "set", "int", "bytes", "object", "huge"],
)
def test_a_name_that_is_not_a_tool_is_an_unknown_tool(name):
    ok, payload = tools.call_tool_safely(name, {"run_dir": "x"})
    assert (ok, payload["code"]) == (False, "registry")
    assert payload["available"] == PUBLIC_TOOLS
    assert len(json.dumps(payload)) < 2000


def test_when_even_describing_the_failure_fails_the_answer_is_a_fixed_internal_error(monkeypatch):
    def failing(exc):
        raise RuntimeError("the mapping failed too")

    monkeypatch.setattr(tools, "to_error_response", failing)
    ok, payload = tools.call_tool_safely("status", {"run_dir": 5})
    assert (ok, payload["code"]) == (False, "internal_error") and set(payload) == FIELDS
    assert payload["detail"]  # it still names a next step


def test_a_result_with_a_number_that_is_not_finite_is_null_in_every_front_end(monkeypatch, run_dir):
    """The command line writes it as null (model_dump_json): so does the function a server calls."""
    from gelpy.schemas.lanes import LaneDetectionResult

    result = LaneDetectionResult(
        centers=[float("nan")], pitch=float("inf"), profile_width=1, n_detected=1, n_expected=1, confidence="low"
    )
    stub = dataclasses.replace(tools.TOOLS["status"], handler=lambda request: result)
    monkeypatch.setitem(tools.TOOLS, "status", stub)
    ok, payload = tools.call_tool_safely("status", {"run_dir": run_dir})
    assert ok and payload["centers"] == [None] and payload["pitch"] is None
    json.dumps(payload, allow_nan=False)  # strict JSON, whatever reads it
    printed = runner.invoke(app, ["tool", "call", "status"], input=json.dumps({"run_dir": run_dir}))
    assert json.loads(printed.stdout) == payload


# --- the lock: a handler's whole step is marked, so a missing one is seen --------------------------
#: init takes the lock itself, with create=True: it is the command that makes the folder a run.
SELF_LOCKING = {"init"}


def _unlocked_writers(registry):
    return sorted(
        name
        for name, tool in registry.items()
        if not tool.read_only and name not in SELF_LOCKING and not takes_run_lock(tool.handler)
    )


def test_every_tool_that_writes_runs_its_whole_step_under_the_lock():
    """The handler of a tool is marked as locked, which a held lock alone would not show: a handler
    whose first step locks itself refuses a second writer even if its later steps (the report, the
    plots) run outside the lock. A tool added later is checked without editing this file."""
    assert _unlocked_writers(tools.TOOLS) == []
    assert [n for n, t in tools.TOOLS.items() if t.read_only and takes_run_lock(t.handler)] == []


def test_the_structural_check_sees_a_handler_that_forgot_the_whole_step_lock():
    forgot = dataclasses.replace(tools.TOOLS["fit_run"], handler=tools._fit_run)
    assert _unlocked_writers({**tools.TOOLS, "fit_run": forgot}) == ["fit_run"]


# --- gelpy tool call: what it reads, and when ---------------------------------------------------------
def test_tool_call_reads_stdin_and_a_file_as_utf8_with_or_without_a_byte_order_mark(
    gel_image, run_dir, tmp_path
):
    _prepared(gel_image, run_dir)
    text = json.dumps({"run_dir": run_dir})
    for raw in (text.encode("utf-8"), b"\xef\xbb\xbf" + text.encode("utf-8")):  # Notepad, PowerShell
        result = runner.invoke(app, ["tool", "call", "status"], input=raw)
        assert result.exit_code == 0, result.output
    arguments = tmp_path / "arguments.json"
    arguments.write_text(text, encoding="utf-8-sig")
    result = runner.invoke(app, ["tool", "call", "status", "--input", str(arguments)])
    assert result.exit_code == 0, result.output


def test_input_that_is_not_utf8_is_an_invalid_value_that_names_the_input(tmp_path):
    result = runner.invoke(app, ["tool", "call", "status"], input=b'{"run_dir": "\xff"}')
    error = json.loads(result.stderr)
    assert (result.exit_code, error["code"]) == (1, "invalid_value")
    assert "stdin" in error["error"] and "UTF-8" in error["error"]
    assert "UTF-8" in error["detail"] and "bug" not in error["detail"]  # a next step, not the generic one
    broken = tmp_path / "arguments.json"
    broken.write_bytes(b'{"run_dir": "\xff"}')
    result = runner.invoke(app, ["tool", "call", "status", "--input", str(broken)])
    error = json.loads(result.stderr)
    assert (result.exit_code, error["code"]) == (1, "invalid_value")
    assert "arguments.json" in error["error"] and "UTF-8" in error["detail"]


def test_a_mistyped_tool_name_fails_before_stdin_is_read(monkeypatch):
    from gelpy.cli import app as app_module

    reads = []
    monkeypatch.setattr(app_module, "_read_tool_input", lambda *args: reads.append(args) or "{}")
    result = runner.invoke(app, ["tool", "call", "nope"], input="{}")
    assert (result.exit_code, json.loads(result.stderr)["code"]) == (1, "registry")
    assert reads == []  # a person who mistyped it is not left waiting for input


def test_a_closed_stdin_is_a_usage_error_that_names_the_way_out(monkeypatch):
    """The program that ran the command closed fd 0: Python then has no sys.stdin at all."""
    from gelpy.cli import app as app_module

    monkeypatch.setattr(sys, "stdin", None)
    with pytest.raises(app_module._io.USAGE_ERROR) as info:
        app_module._read_tool_input("-")
    assert "stdin" in info.value.format_message() and "--input" in info.value.format_message()


@pytest.mark.skipif(sys.platform == "win32", reason="a POSIX shell closes the descriptor")
def test_a_closed_stdin_answers_one_usage_error_object_with_exit_code_2():
    done = subprocess.run(
        ["sh", "-c", f'"{sys.executable}" -m gelpy tool call status <&-'],
        capture_output=True, text=True, timeout=120,
    )
    assert (done.returncode, done.stdout) == (2, "")
    error = json.loads(done.stderr)
    assert error["code"] == "usage_error" and "--input" in error["available"]
    assert "stdin" in error["error"] and "gelpy tool call --help" in error["detail"]


# --- no argument comes back in full inside an error ---------------------------------------------------
BIG = "k" * 1_000_000

#: (label, tool, arguments given a prepared run folder and the image, the codes a front-end may answer).
HOSTILE = [
    ("run_dir", "status", lambda run, gel: {"run_dir": BIG}, {"not_found"}),
    ("report run_dir", "report_html", lambda run, gel: {"run_dir": BIG}, {"not_found"}),
    ("report stylesheet", "report_html", lambda run, gel: {"run_dir": run, "css": BIG}, {"os_error", "not_found"}),
    ("report out_dir", "report_html", lambda run, gel: {"run_dir": run, "out_dir": BIG}, {"os_error", "invalid_value"}),
    ("tool name", BIG, lambda run, gel: {"run_dir": run}, {"registry"}),
    ("image_path", "init", lambda run, gel: {"run_dir": run + "-new", "image_path": BIG}, {"os_error", "not_found"}),
    ("background model", "background_remove", lambda run, gel: {"run_dir": run, "model": BIG}, {"registry"}),
    ("projection strategy", "lanes_detect", lambda run, gel: {"run_dir": run, "projection_strategy": BIG}, {"registry"}),
    ("fit method", "fit_run", lambda run, gel: {"run_dir": run, "fit": {"method": BIG}}, {"registry"}),
    ("lane selector", "fit_run", lambda run, gel: {"run_dir": run, "lanes": [BIG]}, {"lane_selection"}),
    ("lane regex", "fit_run", lambda run, gel: {"run_dir": run, "lanes": ["re:" + BIG]}, {"lane_selection"}),
    ("refit lane", "fit_refit_lane", lambda run, gel: {"run_dir": run, "lane": BIG}, {"lane_selection"}),
    ("unknown key", "status", lambda run, gel: {"run_dir": run, BIG: 1}, {"validation_error"}),
]


@pytest.mark.parametrize(("tool", "arguments", "codes"), [h[1:] for h in HOSTILE], ids=[h[0] for h in HOSTILE])
def test_no_argument_comes_back_in_full_inside_an_error(gel_image, run_dir, tool, arguments, codes):
    _prepared(gel_image, run_dir)
    ok, payload = tools.call_tool_safely(tool, arguments(run_dir, gel_image))
    assert ok is False and payload["code"] in codes, payload["error"][:200]
    assert len(json.dumps(payload)) < 2000  # a megabyte was sent; a few hundred characters come back
