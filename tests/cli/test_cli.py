import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import gelpy
from gelpy.cli.app import app
from gelpy.core.errors import InvalidValueError

runner = CliRunner()


def _run(*args):
    result = runner.invoke(app, list(args))
    assert result.exit_code == 0, f"args={args}\n{result.output}"
    return result


def test_full_workflow(gel_image, run_dir):
    # init
    _run("init", run_dir, gel_image, "--n-lanes", "4", "--width", "30")

    # lanes set (evenly spaced)
    _run("lanes", "set", run_dir, "--linspace", "40", "250", "4", "--width", "30")

    # background remove
    _run("background", "remove", run_dir, "--model", "plane_ransac")

    # image show corrected -> PNG on disk
    out = _run("image", "show", run_dir, "--kind", "corrected")
    # Every emitted path is relative to the run folder, so a caller joins it with run_dir.
    # One base for the whole JSON surface: mixing bases means a caller cannot resolve them.
    png = json.loads(out.output)["image"]
    assert not Path(png).is_absolute()
    assert (Path(run_dir) / png).exists()

    # profiles extract with plot
    out = _run("profiles", "extract", run_dir, "--lane", "0", "--lane", "1", "--plot")
    overview = json.loads(out.output)["files"]["profiles_overview"]
    assert (Path(run_dir) / overview).exists()

    # broad fit pass
    out = _run(
        "fit", "run", run_dir,
        "--lane", "0", "--lane", "1",
        "--method", "grow_select", "--plot",
    )
    summary = json.loads(out.output)
    assert "fit:lane=0" in summary["recomputed"]
    assert Path(run_dir, "render", "fits", "lane_000.png").exists()

    # targeted refit of one lane (seed change -> new fit_hash) -> only that fit recomputes
    out = _run("fit", "refit-lane", run_dir, "--lane", "1", "--seed", "1")
    summary = json.loads(out.output)
    assert summary["recomputed"] == ["fit:lane=1"]
    assert "background" in summary["reused"]

    # results as csv
    out = _run("results", "show", run_dir, "--format", "csv")
    assert "lane_index" in out.output


def test_models_list(gel_image, run_dir):
    out = _run("models", "list", "--kind", "fit")
    assert set(json.loads(out.output)["names"]) == {"grow_select", "sparse_seed", "bayesian"}
    shapes = _run("models", "list", "--kind", "shape")
    assert set(json.loads(shapes.output)["names"]) == {"gaussian", "emg", "bi_gaussian", "pseudo_voigt"}


def test_schema_command():
    out = _run("schema", "RefitLaneRequest")
    schema = json.loads(out.output)
    assert schema["title"] == "RefitLaneRequest"
    assert "lane" in schema["properties"]


def test_lanes_detect_apply_seeds_recipe(gel_image, run_dir):
    # The gel_image fixture has 4 evenly-spaced lanes at x = 40, 110, 180, 250 on a
    # strong background gradient, so detection needs the corrected image here (--corrected);
    # real dark-background fluorescence gels detect fine on raw (the default).
    _run("init", run_dir, gel_image, "--n-lanes", "4")
    _run("background", "remove", run_dir, "--model", "plane_ransac")
    out = _run("lanes", "detect", run_dir, "--n-lanes", "4", "--corrected", "--apply")
    result = json.loads(out.output)
    assert result["confidence"] == "ok"
    assert len(result["centers"]) == 4
    assert (Path(run_dir) / result["overlay"]).exists()  # run-relative, like every path
    # Detected centers land on the true lane positions (well within half a lane pitch of
    # 70px), not the init 5%-95% placeholder that ignores where the gel actually is.
    for got, want in zip(sorted(result["centers"]), [40, 110, 180, 250]):
        assert abs(got - want) < 10.0
    # The recipe now carries the detected positions.
    recipe = json.loads(Path(run_dir, "recipe.json").read_text())
    xs = sorted(lane["x_center_px"] for lane in recipe["lanes"]["lanes"])
    assert abs(xs[0] - 40) < 10.0 and abs(xs[-1] - 250) < 10.0


def test_lanes_set_endpoints_linspaces(gel_image, run_dir):
    _run("init", run_dir, gel_image, "--n-lanes", "4")
    _run("lanes", "set", run_dir, "--endpoints", "40", "250", "--n-lanes", "4")
    recipe = json.loads(Path(run_dir, "recipe.json").read_text())
    xs = sorted(lane["x_center_px"] for lane in recipe["lanes"]["lanes"])
    assert xs == [40.0, 110.0, 180.0, 250.0]


def test_unknown_lane_errors_with_available(gel_image, run_dir):
    _run("init", run_dir, gel_image, "--n-lanes", "3")
    _run("lanes", "set", run_dir, "--linspace", "40", "250", "3")
    result = runner.invoke(app, ["fit", "run", run_dir, "--lane", "nonexistent"])
    assert result.exit_code == 1
    err = json.loads(result.stderr)
    assert err["code"] == "lane_selection"
    assert err["available"] is not None


def test_unknown_background_model_errors(gel_image, run_dir):
    _run("init", run_dir, gel_image, "--n-lanes", "3")
    result = runner.invoke(app, ["background", "remove", run_dir, "--model", "bogus"])
    assert result.exit_code == 1
    err = json.loads(result.stderr)
    assert err["code"] == "registry"
    assert "plane_ransac" in (err["available"] or [])


def test_lane_all_explicit_matches_omitted(gel_image, run_dir):
    # Regression: --lane all used to be parsed as a literal label term (matching
    # nothing) rather than the documented "all lanes" selector; only omitting
    # --lane entirely worked.
    _run("init", run_dir, gel_image, "--n-lanes", "3")
    _run("lanes", "set", run_dir, "--linspace", "40", "250", "3")
    _run("background", "remove", run_dir, "--model", "plane_ransac")
    out = _run("profiles", "extract", run_dir, "--lane", "all")
    recomputed = json.loads(out.output)["recomputed"]
    assert sum(1 for k in recomputed if k.startswith("profile:lane=")) == 3

    out_upper = _run("profiles", "extract", run_dir, "--lane", "ALL")
    second = json.loads(out_upper.output)
    assert second["recomputed"] == []
    assert sum(1 for k in second["reused"] if k.startswith("profile:lane=")) == 3


def test_status_and_editable_fit_config(gel_image, run_dir):
    """`status` reports per-lane fit state, and a hand-edited fit_config.json is
    honored on the next refit — the 'edit intent, then apply' loop."""
    _run("init", run_dir, gel_image, "--n-lanes", "4", "--width", "30")
    _run("lanes", "set", run_dir, "--linspace", "40", "250", "4", "--width", "30")
    _run("background", "remove", run_dir, "--model", "plane_ransac")
    _run(
        "fit", "run", run_dir,
        "--lane", "0", "--lane", "1",
        "--method", "grow_select",
    )

    # fit_config.json exists, is human-readable, and holds the two fit lanes.
    fc_path = Path(run_dir, "fit_config.json")
    assert fc_path.exists()
    store = json.loads(fc_path.read_text())
    assert set(store["lanes"]) == {"0", "1"}

    # status: two current fits, nothing stale, unfit lanes flagged.
    st = json.loads(_run("status", run_dir).output)
    assert st["n_lanes"] == 4 and st["n_fit"] == 2 and st["n_stale"] == 0
    by_index = {lane["index"]: lane for lane in st["lanes"]}
    assert by_index[0]["has_fit"] and by_index[0]["fit_current"]
    assert not by_index[2]["has_fit"]

    # Hand-edit lane 1's stored fit config, then refit with NO overrides: it must
    # re-fit from the edited value (hash changes -> recompute), and persist it.
    store["lanes"]["1"]["fit"]["seed"] = 5
    fc_path.write_text(json.dumps(store))
    summary = json.loads(_run("fit", "refit-lane", run_dir, "--lane", "1").output)
    assert summary["recomputed"] == ["fit:lane=1"]
    reloaded = json.loads(fc_path.read_text())
    assert reloaded["lanes"]["1"]["fit"]["seed"] == 5


def test_init_invert_flag_and_image_set(gel_image, run_dir):
    _run("init", run_dir, gel_image, "--n-lanes", "4", "--no-invert")
    out = _run("status", run_dir)
    assert json.loads(out.output)["polarity"] == "bright_on_dark"
    _run("image", "set", run_dir, "--invert")
    out = _run("status", run_dir)
    assert json.loads(out.output)["polarity"] == "dark_on_light"


def test_version_flag_prints_the_package_version():
    out = _run("--version")
    assert out.stdout.strip() == f"gelpy {gelpy.__version__}"


def test_init_twice_needs_force(gel_image, run_dir):
    _run("init", run_dir, gel_image, "--n-lanes", "4")
    again = runner.invoke(app, ["init", run_dir, gel_image, "--n-lanes", "4"])
    assert again.exit_code == 1
    assert json.loads(again.stderr)["code"] == "run_exists"
    _run("init", run_dir, gel_image, "--n-lanes", "3", "--force")
    assert json.loads(_run("status", run_dir).stdout)["n_lanes"] == 3


def test_image_show_accepts_only_the_four_kinds(gel_image, run_dir, tmp_path):
    _run("init", run_dir, gel_image, "--n-lanes", "4")
    result = runner.invoke(app, ["image", "show", run_dir, "--kind", "../../escaped"])
    assert result.exit_code == 1
    assert json.loads(result.stderr)["code"] == "validation_error"
    assert not list(tmp_path.rglob("escaped*"))


def test_render_image_checks_the_kind_for_python_callers(gel_image, run_dir):
    from gelpy.cli import _render

    _run("init", run_dir, gel_image, "--n-lanes", "4")
    with pytest.raises(InvalidValueError) as info:
        _render.render_image(run_dir, kind="../x")
    assert info.value.available == ["raw", "linear", "non_linear", "corrected"]


def test_fit_without_background_needs_allow_raw(gel_image, run_dir):
    _run("init", run_dir, gel_image, "--n-lanes", "4")
    _run("lanes", "set", run_dir, "--linspace", "40", "250", "4")
    refused = runner.invoke(app, ["fit", "run", run_dir, "--lane", "0"])
    assert refused.exit_code == 1
    assert json.loads(refused.stderr)["code"] == "background_required"
    _run("fit", "run", run_dir, "--lane", "0", "--allow-raw")


@pytest.mark.parametrize(
    "args",
    [("profiles", "extract", "--lane", "0"), ("fit", "refit-lane", "--lane", "0")],
    ids=["profiles-extract", "fit-refit-lane"],
)
def test_allow_raw_reaches_every_command_that_needs_a_background(gel_image, run_dir, args):
    _run("init", run_dir, gel_image, "--n-lanes", "4")
    _run("lanes", "set", run_dir, "--linspace", "40", "250", "4")
    command = (*args[:2], run_dir, *args[2:])
    refused = runner.invoke(app, list(command))
    assert refused.exit_code == 1
    assert json.loads(refused.stderr)["code"] == "background_required"
    _run(*command, "--allow-raw")


def test_a_run_made_with_relative_paths_is_read_from_another_directory(gel_image, tmp_path, monkeypatch):
    """An agent changes directory between calls: status, results and the renders find everything
    through the run folder, never through the working directory."""
    (tmp_path / "work").mkdir()
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path)
    _run("init", "work/run", "gel.tif", "--n-lanes", "4", "--width", "30")
    _run("lanes", "set", "work/run", "--linspace", "40", "250", "4", "--width", "30")
    _run("background", "remove", "work/run", "--model", "plane_ransac")
    _run("fit", "run", "work/run", "--lane", "0", "--lane", "1")

    monkeypatch.chdir(tmp_path / "elsewhere")
    run = "../work/run"
    status = json.loads(_run("status", run).output)
    assert (status["n_fit"], status["n_stale"], status["stale_stages"]) == (2, 0, [])
    assert status["source_image"] == "../../gel.tif"  # as the recipe records it: from the run folder
    results = json.loads(_run("results", "show", run).output)
    assert {lane["lane_index"] for lane in results["lanes"]} == {0, 1}
    assert "lane_index" in _run("results", "show", run, "--format", "csv").output
    refit = json.loads(_run("fit", "refit-lane", run, "--lane", "1", "--plot").output)
    assert (Path(run) / refit["files"]["plot"]).exists()
    assert (Path(run) / refit["files"]["report"]).exists()


def test_lanes_detect_runs_the_detection_once_with_the_overlay(gel_image, run_dir, monkeypatch):
    from gelpy.core import lanes as core_lanes
    from gelpy.runfolder import commands as run_commands

    calls = []
    original = core_lanes.detect_lane_centers

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(core_lanes, "detect_lane_centers", counting)
    monkeypatch.setattr(run_commands, "detect_lane_centers", counting)
    _run("init", run_dir, gel_image, "--n-lanes", "4")
    detected = json.loads(_run("lanes", "detect", run_dir, "--n-lanes", "4").stdout)
    assert (Path(run_dir) / detected["overlay"]).exists()
    assert len(calls) == 1


def test_the_overlay_uses_the_row_bounds_of_the_detection(gel_image, run_dir, monkeypatch):
    from gelpy import render

    drawn = {}
    original = render.render_lane_detection

    def spy(*args, **kwargs):
        drawn.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(render, "render_lane_detection", spy)
    _run("init", run_dir, gel_image, "--n-lanes", "4")
    _run("lanes", "detect", run_dir, "--n-lanes", "4", "--y-start", "30", "--y-end", "200")
    assert (drawn["y_start"], drawn["y_end"]) == (30, 200)
