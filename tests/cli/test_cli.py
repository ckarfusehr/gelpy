import json
from pathlib import Path

from typer.testing import CliRunner

from gelpy.cli.app import app

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
    err = json.loads(result.output)
    assert err["code"] == "lane_selection"
    assert err["available"] is not None


def test_unknown_background_model_errors(gel_image, run_dir):
    _run("init", run_dir, gel_image, "--n-lanes", "3")
    result = runner.invoke(app, ["background", "remove", run_dir, "--model", "bogus"])
    assert result.exit_code == 1
    err = json.loads(result.output)
    assert err["code"] == "registry"
    assert "plane_ransac" in (err["available"] or [])


def test_lane_all_explicit_matches_omitted(gel_image, run_dir):
    # Regression: --lane all used to be parsed as a literal label term (matching
    # nothing) rather than the documented "all lanes" selector; only omitting
    # --lane entirely worked.
    _run("init", run_dir, gel_image, "--n-lanes", "3")
    _run("lanes", "set", run_dir, "--linspace", "40", "250", "3")
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
