"""`image rotate`, `image crop` and `image grid`: the CLI, `tool call` and `tool list` expose them."""

import json
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from gelpy.cli.app import app

runner = CliRunner()


def _ok(*args, input=None):
    result = runner.invoke(app, list(args), input=input)
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def test_rotate_crop_and_grid_through_the_cli(gel_image, run_dir):
    _ok("init", run_dir, gel_image, "--n-lanes", "4")
    turned = _ok("image", "rotate", run_dir, "--turn", "-90", "--angle", "-2.5", "--flip")
    assert turned["image"]["geometry"] == {"turn": 270, "flip": True, "angle": -2.5, "crop": None}
    width, height = turned["image"]["oriented_width"], turned["image"]["oriented_height"]
    cropped = _ok("image", "crop", run_dir, "--box", "8", "9", str(width - 8), str(height - 9))
    assert cropped["image"]["geometry"]["crop"] == [8, 9, width - 8, height - 9]
    assert (cropped["image"]["width"], cropped["image"]["height"]) == (width - 16, height - 18)

    grid = _ok("image", "grid", run_dir, "--step", "50")
    png = Path(run_dir) / grid["image"]
    assert grid["image"] == "render/grid.png"
    with Image.open(png) as picture:  # a readable image of a useful size
        picture.load()
        assert min(picture.size) > 300
    shown = _ok("image", "show", run_dir, "--kind", "raw")  # the analysed (cropped) view still renders
    assert (Path(run_dir) / shown["image"]).is_file()

    removed = _ok("image", "crop", run_dir)
    assert removed["image"]["geometry"]["crop"] is None
    status = _ok("status", run_dir)
    assert status["geometry"]["turn"] == 270
    assert (status["analysis_width"], status["analysis_height"]) == (width, height)


def test_an_out_of_range_box_is_a_clear_error(gel_image, run_dir):
    _ok("init", run_dir, gel_image, "--n-lanes", "4")
    result = runner.invoke(app, ["image", "crop", run_dir, "--box", "0", "0", "301", "10"])
    assert result.exit_code != 0
    error = json.loads(result.stderr)
    assert error["code"] == "invalid_value"
    assert "x1 = 301 is beyond the width 300" in error["error"]
    bad_turn = runner.invoke(app, ["image", "rotate", run_dir, "--turn", "45"])
    assert bad_turn.exit_code != 0 and "multiple of 90" in bad_turn.stderr


def test_tool_call_and_tool_list_expose_the_geometry_tools(gel_image, run_dir):
    _ok("init", run_dir, gel_image, "--n-lanes", "4")
    turned = _ok("tool", "call", "image_rotate", input=json.dumps({"run_dir": run_dir, "turn": 180}))
    assert turned["image"]["geometry"]["turn"] == 180
    cropped = _ok("tool", "call", "image_crop", input=json.dumps({"run_dir": run_dir, "box": [1, 2, 299, 248]}))
    assert cropped["image"]["geometry"]["crop"] == [1, 2, 299, 248]
    grid = _ok("tool", "call", "image_grid", input=json.dumps({"run_dir": run_dir}))
    assert (Path(run_dir) / grid["image"]).is_file()

    listing = {tool["name"]: tool for tool in _ok("tool", "list", "--json")["tools"]}
    for name in ("image_rotate", "image_crop", "image_grid"):
        assert name in listing
    rotate = listing["image_rotate"]["input_schema"]["properties"]
    assert {"turn", "flip", "angle"} <= set(rotate)
    assert "box" in listing["image_crop"]["input_schema"]["required"]
    assert "wells are at the top" in listing["image_rotate"]["description"]
    assert "short margin below the lowest band" in listing["image_crop"]["description"]
