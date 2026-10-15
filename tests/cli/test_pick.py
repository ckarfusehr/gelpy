"""`lanes pick` needs a display and a person: when it has none, or the person gives up, it says so.

It used to raise a bare RuntimeError for both, which was reported as ``internal_error`` ("this is a
bug in gelpy"). Each now has a code of its own and a next step.
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from typer.testing import CliRunner

from gelpy.cli import _pick
from gelpy.cli.app import app
from gelpy.core import errors
from gelpy.core.errors import to_error_response

runner = CliRunner()


@pytest.fixture
def no_display(monkeypatch):
    """No GUI backend can be selected."""

    def refuse(backend, *args, **kwargs):
        raise ImportError(f"cannot load {backend}")

    monkeypatch.setattr(matplotlib, "use", refuse)


class _Window:
    """A figure whose window the person closes after ``clicks`` clicks."""

    def __init__(self, clicks):
        self.clicks = clicks

    def ginput(self, n, timeout=0):
        return [(10.0 + 20.0 * i, 5.0) for i in range(min(n, self.clicks))]

    def __getattr__(self, name):  # title, colorbar, ...: whatever a drawn figure takes
        return lambda *args, **kwargs: None


class _Axes:
    def imshow(self, *args, **kwargs):
        return None

    def set_title(self, *args, **kwargs):
        return None


@pytest.fixture
def window(monkeypatch):
    """``window(clicks)``: a picker whose window is closed after that many clicks."""

    def make(clicks):
        monkeypatch.setattr(_pick, "_use_interactive_backend", lambda: None)
        monkeypatch.setattr(plt, "subplots", lambda *a, **k: (_Window(clicks), _Axes()))
        monkeypatch.setattr(plt, "close", lambda *a, **k: None)

    return make


def _pick_two(contrast=None):
    return _pick.pick_lane_centers(np.zeros((20, 40)), contrast, 2, title="a gel")


# --- no display -----------------------------------------------------------------------------------------
def test_no_gui_backend_is_display_required_with_the_two_ways_out(no_display):
    with pytest.raises(errors.DisplayRequiredError) as raised:
        _pick_two()
    response = to_error_response(raised.value)
    assert response.code == "display_required"
    assert "display" in response.error
    assert "--endpoints" in response.detail and "--x-center" in response.detail  # set the lanes by hand
    assert "display" in response.detail  # or run it where there is one
    assert "bug" not in response.detail


def test_a_window_that_cannot_be_opened_is_display_required_too(monkeypatch):
    """A GUI backend can be selected on a machine with no display and fail only when the window opens."""
    monkeypatch.setattr(_pick, "_use_interactive_backend", lambda: None)

    def no_window(*args, **kwargs):
        raise RuntimeError("no display name and no $DISPLAY environment variable")

    monkeypatch.setattr(plt, "subplots", no_window)
    with pytest.raises(errors.DisplayRequiredError) as raised:
        _pick_two()
    assert "no display name" in str(raised.value)  # what the system said
    assert to_error_response(raised.value).code == "display_required"


def test_the_command_answers_display_required_and_changes_nothing(gel_image, run_dir, no_display):
    assert runner.invoke(app, ["init", run_dir, gel_image, "--n-lanes", "4", "--width", "30"]).exit_code == 0
    recipe = (Path(run_dir) / "recipe.json").read_bytes()

    result = runner.invoke(app, ["lanes", "pick", run_dir, "--n-lanes", "4"])

    assert result.exit_code == 1 and result.stdout == ""
    error = json.loads(result.stderr)
    assert error["code"] == "display_required"
    assert "--endpoints" in error["detail"] and "--x-center" in error["detail"]
    assert (Path(run_dir) / "recipe.json").read_bytes() == recipe


# --- the window is closed early -----------------------------------------------------------------------
def test_a_window_closed_before_the_clicks_are_done_is_pick_cancelled(window):
    window(clicks=1)
    with pytest.raises(errors.PickCancelledError) as raised:
        _pick_two()
    response = to_error_response(raised.value)
    assert response.code == "pick_cancelled"
    assert "1 of 2" in response.error  # what was done, of what was asked
    assert "again" in response.detail and "bug" not in response.detail


def test_the_command_answers_pick_cancelled_and_changes_nothing(gel_image, run_dir, window):
    assert runner.invoke(app, ["init", run_dir, gel_image, "--n-lanes", "4", "--width", "30"]).exit_code == 0
    recipe = (Path(run_dir) / "recipe.json").read_bytes()
    window(clicks=0)

    result = runner.invoke(app, ["lanes", "pick", run_dir, "--n-lanes", "4"])

    assert result.exit_code == 1 and result.stdout == ""
    assert json.loads(result.stderr)["code"] == "pick_cancelled"
    assert (Path(run_dir) / "recipe.json").read_bytes() == recipe


def test_all_the_clicks_are_the_lane_centres_left_to_right(window):
    window(clicks=2)
    assert _pick_two() == [10.0, 30.0]


def test_both_codes_are_in_the_error_reference():
    reference = (Path(__file__).resolve().parents[2] / "docs" / "reference" / "errors.md").read_text(
        encoding="utf-8"
    )
    for code, exception in (("display_required", "DisplayRequiredError"), ("pick_cancelled", "PickCancelledError")):
        assert f"| `{code}` | `{exception}` |" in reference
