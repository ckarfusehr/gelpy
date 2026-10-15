"""A lane count over 1000 is refused at once, before any work is done, on every front-end.

The cap itself is a field constraint (tests/schemas/test_lane_cap.py); these tests check what a
caller sees: one structured error, no run folder made, no image decoded, no detection run, and a
recipe or fit configuration edited by hand to hold more lanes is refused when it is loaded.
"""

import json
import time

import numpy as np
import pytest
from skimage import io
from typer.testing import CliRunner

from gelpy.cli import _pick, tools
from gelpy.cli.app import app
from gelpy.runfolder import commands
from gelpy.runfolder.layout import RunPaths

runner = CliRunner()


def _fails(*args):
    """The one structured error a failing command writes to stderr (stdout stays empty)."""
    result = runner.invoke(app, list(args))
    assert result.exit_code == 1, result.output
    assert result.stdout == ""
    return json.loads(result.stderr)


def _limit_is_named(error, field):
    assert error["code"] == "validation_error"
    assert field in error["error"]
    assert "1000" in error["detail"]  # the limit is in the field error


@pytest.fixture
def no_work(monkeypatch):
    """Fail the test if anything decodes an image or looks for lanes."""

    def forbidden(*args, **kwargs):
        raise AssertionError("work was done for a request that validation refuses")

    monkeypatch.setattr(commands, "build_image_ref_and_meta", forbidden)
    monkeypatch.setattr(commands, "detect_lane_centers", forbidden)
    monkeypatch.setattr(commands, "Pipeline", forbidden)
    monkeypatch.setattr(_pick, "pick_lane_centers", forbidden)


def test_init_with_1001_lanes_is_refused_at_once_and_makes_nothing(gel_image, run_dir, no_work, tmp_path):
    error = _fails("init", run_dir, gel_image, "--n-lanes", "1001")
    _limit_is_named(error, "n_lanes")
    assert not (tmp_path / "run").exists()


def test_init_with_a_labels_file_of_1001_lines_is_refused_at_once(gel_image, run_dir, no_work, tmp_path):
    labels = tmp_path / "labels.txt"
    labels.write_text("\n".join(f"s{i}" for i in range(1001)) + "\n", encoding="utf-8")
    error = _fails("init", run_dir, gel_image, "--n-lanes", "1000", "--labels-file", str(labels))
    _limit_is_named(error, "labels")
    assert not (tmp_path / "run").exists()


def test_init_with_1000_lanes_passes(tmp_path):
    """The limit itself is allowed: the whole step works on an image wide enough for 1000 lanes."""
    wide = tmp_path / "wide.tif"
    io.imsave(str(wide), np.random.default_rng(0).random((40, 4000)).astype(np.float32))
    result = runner.invoke(app, ["init", str(tmp_path / "run"), str(wide), "--n-lanes", "1000"])
    assert result.exit_code == 0, result.output
    status = json.loads(runner.invoke(app, ["status", str(tmp_path / "run")]).stdout)
    assert status["n_lanes"] == 1000


@pytest.fixture
def started(gel_image, run_dir):
    assert runner.invoke(app, ["init", run_dir, gel_image, "--n-lanes", "4", "--width", "30"]).exit_code == 0
    return run_dir


def test_lanes_detect_with_1001_lanes_is_refused_at_once(started, no_work):
    error = _fails("lanes", "detect", started, "--n-lanes", "1001")
    _limit_is_named(error, "n_lanes")


def test_lanes_set_with_1001_evenly_spaced_lanes_is_refused(started, no_work):
    error = _fails("lanes", "set", started, "--linspace", "0", "100", "1001")
    assert error["code"] == "validation_error" and "linspace" in error["error"]
    assert "1000" in error["detail"]


def test_lanes_set_with_1001_centres_is_refused(started, no_work):
    centres = [arg for x in range(1001) for arg in ("--x-center", str(x))]
    error = _fails("lanes", "set", started, *centres)
    _limit_is_named(error, "x_centers")


def test_lanes_set_with_n_lanes_over_the_cap_is_refused(started, no_work):
    error = _fails("lanes", "set", started, "--endpoints", "10", "200", "--n-lanes", "1001")
    _limit_is_named(error, "n_lanes")


def test_lanes_pick_checks_the_count_before_a_window_opens(started, no_work):
    """Clicking 1001 lanes just to be told that it is too many would be the worst order."""
    error = _fails("lanes", "pick", started, "--n-lanes", "1001", "--all")
    _limit_is_named(error, "n_lanes")


@pytest.mark.parametrize(
    ("tool", "arguments", "field"),
    [
        ("init", {"image_path": "unused.tif", "n_lanes": 1001}, "n_lanes"),
        ("lanes_detect", {"n_lanes": 1001}, "n_lanes"),
        ("lanes_set", {"linspace": [0, 100, 1001]}, "linspace"),
        ("lanes_set", {"x_centers": [1.0] * 1001}, "x_centers"),
        ("fit_run", {"lanes": list(range(1001))}, "lanes"),
    ],
)
def test_a_tool_call_over_the_cap_is_a_validation_error(started, no_work, tool, arguments, field):
    ok, answer = tools.call_tool_safely(tool, {"run_dir": started, **arguments})
    assert not ok
    assert answer["code"] == "validation_error" and field in answer["error"]
    assert "1000" in answer["detail"]


# --- a record edited by hand to hold more lanes ----------------------------------------------------
def _edit(path, change):
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_a_recipe_edited_to_hold_1001_lanes_is_refused_when_it_is_loaded(started):
    def grow(data):
        data["lanes"]["lanes"] = [
            {"index": i, "label": str(i), "x_center_px": float(i % 250)} for i in range(1001)
        ]

    _edit(RunPaths(started).recipe, grow)

    began = time.monotonic()
    error = _fails("status", started)
    assert time.monotonic() - began < 30  # refused, not worked through
    assert error["code"] == "validation_error" and "lanes" in error["error"]
    assert "1000" in error["detail"]


def test_a_fit_config_edited_to_hold_1001_lanes_is_refused_when_it_is_loaded(started):
    def grow(data):
        data["lanes"] = {str(i): {} for i in range(1001)}

    _edit(RunPaths(started).fit_config, grow)

    error = _fails("status", started)
    assert error["code"] == "validation_error" and "lanes" in error["error"]
    assert "1000" in error["detail"]
