"""A request refuses a number that is not finite, so one can never reach a run folder.

JSON has no NaN or infinity. One that got past validation was written as null into recipe.json or
fit_config.json, and the next call on that run folder failed to load it: the run was bricked, and a
fit even reported its failure after it had already written the file. Every front-end builds the same
request models, so the refusal is tested through `tool call` and through the command line.
"""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gelpy.cli import tools
from gelpy.cli.app import app

runner = CliRunner()

#: What JSON can say that is not a finite number: an overflow, NaN, and a negative infinity.
TOKENS = ["1e999", "NaN", "-Infinity"]

#: (tool, the JSON object without its run_dir, TOKEN where the number goes). Direct fields of a
#: request first, then the nested models and the free-form parameters a request carries into
#: fit_config.json and recipe.json.
CASES = [
    ("contrast_set", '{"gamma": TOKEN}'),
    ("contrast_set", '{"gain": TOKEN}'),
    ("contrast_set", '{"intensity_low": TOKEN, "intensity_high": 5, "intensity_mode": "absolute"}'),
    ("contrast_set", '{"intensity_low": 1, "intensity_high": TOKEN, "intensity_mode": "absolute"}'),
    ("lanes_set", '{"x_centers": [TOKEN, 2, 3, 4]}'),
    ("lanes_set", '{"linspace": [TOKEN, 2, 4]}'),
    ("lanes_set", '{"endpoints": [TOKEN, 2], "n_lanes": 4}'),
    ("lanes_set", '{"label_rotation": TOKEN}'),
    ("lanes_detect", '{"n_lanes": 4, "apply": true, "label_rotation": TOKEN, "plot": false}'),
    ("lanes_detect", '{"n_lanes": 4, "smoothing_window": TOKEN, "plot": false}'),
    ("lanes_detect", '{"n_lanes": 4, "min_prominence": TOKEN, "plot": false}'),
    ("fit_run", '{"allow_raw": true, "lanes": [0], "fit": {"baseline": {"lam": TOKEN}}}'),
    ("fit_run", '{"allow_raw": true, "lanes": [0], "fit": {"priors": {"width_prior_strength": TOKEN}}}'),
    ("fit_refit_lane", '{"allow_raw": true, "lane": 0, "fit_overrides": {"priors": {"min_snr": TOKEN}}}'),
    ("background_remove", '{"model": "robust_plane", "params": {"tol": TOKEN}}'),
    ("background_remove", '{"model": "robust_plane", "params": {"contiguity": TOKEN}}'),
]

INTENT = ("recipe.json", "fit_config.json")


def _prepared(gel_image, run_dir):
    init = {"run_dir": run_dir, "image_path": gel_image, "n_lanes": 4, "width": 30}
    result = runner.invoke(app, ["tool", "call", "init"], input=json.dumps(init))
    assert result.exit_code == 0, result.output
    return run_dir


def _intent(run_dir):
    """The files that hold what a person or an agent decided, as bytes."""
    return {name: (Path(run_dir) / name).read_bytes() for name in INTENT}


def _refused(result):
    assert result.exit_code == 1, result.output
    error = json.loads(result.stderr)
    assert error["code"] == "validation_error", error
    return json.loads(error["detail"])


def _still_opens(run_dir):
    result = runner.invoke(app, ["status", run_dir])
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize("token", TOKENS)
@pytest.mark.parametrize(("tool", "template"), CASES, ids=[f"{t}-{i}" for i, (t, _) in enumerate(CASES)])
def test_tool_call_refuses_a_number_that_is_not_finite(gel_image, run_dir, tool, template, token):
    _prepared(gel_image, run_dir)
    before = _intent(run_dir)
    text = '{"run_dir": %s, ' % json.dumps(run_dir) + template.replace("TOKEN", token)[1:]
    fields = _refused(runner.invoke(app, ["tool", "call", tool], input=text))
    assert fields[0]["type"] == "finite_number"  # refused for that reason, not by a bound
    assert _intent(run_dir) == before  # recipe.json and fit_config.json are untouched
    _still_opens(run_dir)


@pytest.mark.parametrize("token", TOKENS)
def test_init_refuses_a_label_rotation_that_is_not_finite(gel_image, run_dir, token):
    text = '{"run_dir": %s, "image_path": %s, "n_lanes": 4, "label_rotation": %s}' % (
        json.dumps(run_dir), json.dumps(gel_image), token,
    )
    _refused(runner.invoke(app, ["tool", "call", "init"], input=text))
    assert not Path(run_dir).exists()


#: The same refusals from the command line: (argv after the run folder, which a test inserts).
FLAGS = [
    ["contrast", "set", "--gamma", "inf"],
    ["contrast", "set", "--gamma", "nan"],
    ["contrast", "set", "--gain", "inf"],
    ["contrast", "set", "--gain", "nan"],
    ["contrast", "set", "--intensity-low", "-inf", "--intensity-high", "5", "--intensity-mode", "absolute"],
    ["contrast", "set", "--intensity-low", "1", "--intensity-high", "inf", "--intensity-mode", "absolute"],
    ["lanes", "set", "--x-center", "inf", "--x-center", "2", "--x-center", "3", "--x-center", "4"],
    ["lanes", "set", "--linspace", "inf", "2", "4"],
    ["lanes", "set", "--endpoints", "nan", "2", "--n-lanes", "4"],
    ["lanes", "set", "--label-rotation", "inf"],
    ["lanes", "set", "--label-rotation", "nan"],
    ["lanes", "detect", "--n-lanes", "4", "--apply", "--label-rotation", "inf", "--no-plot"],
    ["lanes", "detect", "--n-lanes", "4", "--smoothing-window", "inf", "--no-plot"],
    ["background", "remove", "--model", "robust_plane", "--param", "tol=inf"],
    ["background", "remove", "--model", "robust_plane", "--param", "contiguity=nan"],
]


@pytest.mark.parametrize("flags", FLAGS, ids=lambda flags: " ".join(flags))
def test_the_command_line_refuses_a_number_that_is_not_finite(gel_image, run_dir, flags):
    _prepared(gel_image, run_dir)
    before = _intent(run_dir)
    argv = [*flags[:2], run_dir, *flags[2:]]
    fields = _refused(runner.invoke(app, argv))
    assert fields[0]["type"] == "finite_number"
    assert _intent(run_dir) == before
    _still_opens(run_dir)


def test_every_number_in_every_request_refuses_what_is_not_finite():
    """A float field added to a request, or to a model a request carries, is checked without anyone
    editing this file. Pydantic keeps allow_inf_nan on the config of the model a float sits in (the
    float's own key, if there is one, overrides it), so the walk follows the nearest model."""

    def lenient(node, path=(), allowed=True):
        if isinstance(node, dict):
            if node.get("type") == "model":
                allowed = (node.get("config") or {}).get("allow_inf_nan", True)
            if node.get("type") == "float" and node.get("allow_inf_nan", allowed):
                yield "/".join(path)
            for key, value in node.items():
                yield from lenient(value, (*path, str(key)), allowed)
        elif isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                yield from lenient(value, (*path, str(index)), allowed)

    found = [
        f"{name}: {where}"
        for name, tool in tools.TOOLS.items()
        for where in lenient(tool.request.__pydantic_core_schema__)
    ]
    assert found == []


def test_results_still_hold_a_number_that_is_not_finite():
    """Only requests are strict: a result is computed, and one NaN must not stop it being reported."""
    from gelpy.schemas.lanes import LaneDetectionResult

    result = LaneDetectionResult(
        centers=[float("nan")], pitch=1.0, profile_width=1, n_detected=1, n_expected=1, confidence="low"
    )
    assert result.centers[0] != result.centers[0]
