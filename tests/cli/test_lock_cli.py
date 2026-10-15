"""A multi-step CLI command holds one run-folder lock from its first write to its last.

`fit run` is commands.fit_run, then the report, then (with --plot) the fit plots. A retry that
starts between two of those steps must be refused with run_locked. It must not slip in between
them and leave the first command to fail after its main step has already succeeded.
"""

import json
import subprocess
import sys

import pytest
from typer.testing import CliRunner

from gelpy.cli import _render
from gelpy.cli.app import app
from gelpy.runfolder import commands
from gelpy.schemas.requests import SetLanesRequest

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX file locks")

runner = CliRunner()

#: Every CLI command that writes in more than one step: its command line (RUN stands for the
#: run folder) and the functions of cli/_render.py it calls after its first step, in order.
MULTI_STEP = [
    pytest.param(
        ["fit", "run", "RUN", "--lane", "0", "--plot"],
        ["write_report", "render_fits"],
        id="fit-run",
    ),
    pytest.param(
        ["fit", "refit-lane", "RUN", "--lane", "0", "--seed", "1", "--plot"],
        ["write_report", "render_fits"],
        id="fit-refit-lane",
    ),
    pytest.param(
        ["lanes", "detect", "RUN", "--n-lanes", "4"], ["render_lane_overlay"], id="lanes-detect"
    ),
    pytest.param(
        ["background", "remove", "RUN", "--model", "plane_ransac", "--plot"],
        ["render_background"],
        id="background-remove",
    ),
    pytest.param(
        ["profiles", "extract", "RUN", "--lane", "0", "--plot"],
        ["render_profiles"],
        id="profiles-extract",
    ),
    # The report renders its three gel views with one function, so the later steps named here
    # are the ones that run once: the profiles plot, then the fit plots.
    pytest.param(
        ["report", "html", "RUN"], ["render_profiles", "ensure_fit_plots"], id="report-html"
    ),
]


def _with_run(argv, run):
    return [run if arg == "RUN" else arg for arg in argv]


def _second_writer(run_dir):
    """Another process tries to write the run folder: its exit code and its error code."""
    done = subprocess.run(
        [sys.executable, "-m", "gelpy", "lanes", "set", run_dir, "--width", "20"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return done.returncode, (json.loads(done.stderr)["code"] if done.stderr else None)


@pytest.fixture
def prepared(gel_image, run_dir):
    for args in (
        ("init", run_dir, gel_image, "--n-lanes", "4", "--width", "30"),
        ("lanes", "set", run_dir, "--linspace", "40", "250", "4", "--width", "30"),
        ("background", "remove", run_dir, "--model", "plane_ransac"),
    ):
        result = runner.invoke(app, list(args))
        assert result.exit_code == 0, result.output
    return run_dir


@pytest.mark.parametrize(("argv", "later_steps"), MULTI_STEP)
def test_a_second_writer_is_refused_between_the_steps_of_one_command(
    prepared, monkeypatch, argv, later_steps
):
    outcomes = []

    def spy(name):
        real = getattr(_render, name)

        def step(*args, **kwargs):
            outcomes.append((name, _second_writer(prepared)))  # a second process tries to write now
            return real(*args, **kwargs)

        return step

    for name in later_steps:
        monkeypatch.setattr(_render, name, spy(name))

    result = runner.invoke(app, _with_run(argv, prepared))

    assert result.exit_code == 0, result.output  # the command itself completes
    assert [name for name, _ in outcomes] == later_steps  # and it reached each later step
    assert all(outcome == (1, "run_locked") for _, outcome in outcomes), outcomes
    commands.set_lanes(SetLanesRequest(run_dir=prepared, width=20))  # the lock is free again


@pytest.mark.parametrize(("argv", "later_steps"), MULTI_STEP)
def test_a_missing_run_folder_is_still_not_found_and_not_created(tmp_path, argv, later_steps):
    missing = tmp_path / "typo"
    result = runner.invoke(app, _with_run(argv, str(missing)))
    assert result.exit_code == 1, result.output
    assert json.loads(result.stderr)["code"] == "not_found"
    assert not missing.exists()


@pytest.mark.parametrize(("argv", "later_steps"), MULTI_STEP)
def test_a_directory_that_is_not_a_run_is_left_untouched(tmp_path, argv, later_steps):
    """Only init needs the lock without a recipe.json: any other writer fails with not_found and
    leaves no .gelpy.lock in an ordinary directory."""
    plain = tmp_path / "notes"
    plain.mkdir()
    result = runner.invoke(app, _with_run(argv, str(plain)))
    assert result.exit_code == 1, result.output
    assert json.loads(result.stderr)["code"] == "not_found"
    assert list(plain.iterdir()) == []


def test_lanes_set_on_a_plain_directory_leaves_no_file_behind(tmp_path):
    plain = tmp_path / "notes"
    plain.mkdir()
    result = runner.invoke(app, ["lanes", "set", str(plain), "--width", "20"])
    assert result.exit_code == 1, result.output
    assert json.loads(result.stderr)["code"] == "not_found"
    assert list(plain.iterdir()) == []
