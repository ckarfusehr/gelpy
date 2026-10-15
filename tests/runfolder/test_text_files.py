"""A precious text file saved in the wrong encoding is reported by its name and the encoding it needs.

recipe.json and fit_config.json are meant to be edited by hand, and an editor that saves them in a
legacy encoding (Windows' ANSI code page, say) makes them unreadable as UTF-8. That used to come back
as the codec's own "can't decode byte 0xf6" under the generic advice "if the message names no value,
this is a bug in gelpy: report it".
"""

import json

import pytest
from typer.testing import CliRunner

from gelpy.cli.app import app
from gelpy.core.errors import InvalidValueError, to_error_response
from gelpy.runfolder import commands
from gelpy.runfolder.layout import RunPaths
from gelpy.schemas.requests import (
    InitRunRequest,
    SetContrastRequest,
    StatusRequest,
)

runner = CliRunner()
#: Gr, o-umlaut, sharp s, e: a word that Windows-1252 writes in single bytes that are not UTF-8.
GROESSE = "Gr" + chr(0xF6) + chr(0xDF) + "e"


@pytest.fixture
def started(gel_image_path, run_dir):
    commands.init_run(InitRunRequest(run_dir=run_dir, image_path=gel_image_path[0], n_lanes=4, width=30))
    return run_dir


def _save_as_cp1252(path, change):
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_bytes(json.dumps(data, ensure_ascii=False, indent=2).encode("cp1252"))


def _in_the_recipe(data):
    data["lanes"]["lanes"][0]["label"] = GROESSE


def _in_the_fit_config(data):
    data["lanes"] = {"0": {"fit": {"params": {"note": GROESSE}}}}


CASES = [
    pytest.param(lambda p: p.recipe, _in_the_recipe, "recipe.json", id="recipe.json"),
    pytest.param(lambda p: p.fit_config, _in_the_fit_config, "fit_config.json", id="fit_config.json"),
]


@pytest.mark.parametrize(("which", "change", "name"), CASES)
def test_a_file_saved_in_another_encoding_is_refused_by_name_and_encoding(started, which, change, name):
    path = which(RunPaths(started))
    _save_as_cp1252(path, change)

    with pytest.raises(InvalidValueError) as raised:
        commands.status(StatusRequest(run_dir=started))

    response = to_error_response(raised.value)
    assert response.code == "invalid_value"
    assert name in response.error and "UTF-8" in response.error  # which file, and what it must be
    assert "line " in response.error  # and where it stops being
    assert "UTF-8" in response.detail and name in response.detail  # the next step: save it as UTF-8
    assert "bug" not in response.detail and "codec" not in response.error


@pytest.mark.parametrize(("which", "change", "name"), CASES)
def test_the_command_line_says_the_same(started, which, change, name):
    _save_as_cp1252(which(RunPaths(started)), change)

    result = runner.invoke(app, ["status", started])

    assert result.exit_code == 1 and result.stdout == ""
    error = json.loads(result.stderr)
    assert error["code"] == "invalid_value" and name in error["error"] and "UTF-8" in error["detail"]


def test_a_writer_that_reads_the_recipe_first_refuses_it_before_it_writes(started):
    path = RunPaths(started).recipe
    _save_as_cp1252(path, _in_the_recipe)
    before = path.read_bytes()
    with pytest.raises(InvalidValueError):
        commands.set_contrast(SetContrastRequest(run_dir=started, gamma=0.5))
    assert path.read_bytes() == before  # the file the person saved is not overwritten


def test_the_line_of_the_first_byte_that_is_not_utf8_is_named(started):
    _save_as_cp1252(RunPaths(started).recipe, _in_the_recipe)
    lines = RunPaths(started).recipe.read_bytes().split(b"\n")
    first = next(i for i, line in enumerate(lines, start=1) if GROESSE.encode("cp1252") in line)
    with pytest.raises(InvalidValueError) as raised:
        commands.status(StatusRequest(run_dir=started))
    assert f"line {first} " in str(raised.value)


def test_a_file_in_utf8_with_the_same_word_loads(started):
    path = RunPaths(started).recipe
    data = json.loads(path.read_text(encoding="utf-8"))
    _in_the_recipe(data)
    path.write_bytes(json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"))
    assert commands.status(StatusRequest(run_dir=started)).lanes[0].label == GROESSE
