"""Every CLI failure is one JSON object on stderr: domain errors, bare exceptions, usage errors."""

import codecs
import dataclasses
import errno
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from gelpy.cli import tools
from gelpy.cli.app import _StructuredErrors, app

runner = CliRunner()
FIELDS = {"error", "code", "detail", "available"}


def _make_status_raise(monkeypatch, exc):
    """The `status` command runs the `status` tool, whose handler is bound when the registry is
    built: patching `commands.status` would not reach it, so the tool is replaced."""

    def broken(request):
        raise exc

    tool = dataclasses.replace(tools.TOOLS["status"], handler=broken)
    monkeypatch.setitem(tools.TOOLS, "status", tool)


def _error(result, exit_code=1):
    assert result.exit_code == exit_code, result.output
    assert result.stdout == "", result.stdout  # a failing command prints nothing to stdout
    assert "Traceback" not in result.stderr
    payload = json.loads(result.stderr)  # the one JSON object is stderr's: output also holds stdout
    assert set(payload) == FIELDS
    return payload


def test_a_text_file_given_as_image_is_a_decode_error(tmp_path):
    text = tmp_path / "notes.tif"
    text.write_text("not an image", encoding="utf-8")
    error = _error(runner.invoke(app, ["init", str(tmp_path / "run"), str(text), "--n-lanes", "3"]))
    assert error["code"] == "decode_error"
    assert error["detail"]


def test_a_file_with_only_a_tiff_header_leaves_exactly_one_error_object(tmp_path):
    """tifffile logs a warning line for a TIFF that has only a header; that line used to come
    before gelpy's error object, so stderr was not one JSON object. A subprocess keeps the real
    logging configuration: nothing configures logging, so Python's last-resort handler would
    print the line."""
    broken = tmp_path / "broken.tif"
    broken.write_bytes(b"II*\x00\x08\x00\x00\x00")  # a TIFF header pointing past the end
    run = tmp_path / "run"
    done = subprocess.run(
        [sys.executable, "-m", "gelpy", "init", str(run), str(broken), "--n-lanes", "3"],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 1
    assert done.stdout == ""
    payload = json.loads(done.stderr)  # a log line before or after the object fails here
    assert set(payload) == FIELDS
    assert payload["code"] == "decode_error"
    assert payload["detail"]


def test_a_label_count_mismatch_is_invalid_value(gel_image, run_dir, tmp_path):
    labels = tmp_path / "labels.txt"
    labels.write_text("a\nb\n", encoding="utf-8")
    result = runner.invoke(
        app, ["init", run_dir, gel_image, "--n-lanes", "4", "--labels-file", str(labels)]
    )
    error = _error(result)
    assert error["code"] == "invalid_value"
    assert "4" in error["error"]


def test_a_labels_file_that_is_not_utf8_is_refused_by_name(gel_image, run_dir, tmp_path):
    """A file saved in a legacy encoding is refused with its name and the encoding it needs, not
    with the decoder's bare "can't decode byte"; nothing is created."""
    labels = tmp_path / "labels.txt"
    labels.write_bytes("Größe\nProbe\n".encode("latin-1"))
    result = runner.invoke(
        app, ["init", run_dir, gel_image, "--n-lanes", "2", "--labels-file", str(labels)]
    )
    error = _error(result)
    assert error["code"] == "invalid_value"
    assert "labels.txt" in error["error"]
    assert "UTF-8" in error["error"]
    assert "UTF-8" in error["detail"]
    assert "line 1" in error["error"]
    assert not Path(run_dir).exists()


def test_the_line_of_a_bad_byte_counts_from_the_first_line_even_after_a_byte_order_mark(
    gel_image, run_dir, tmp_path
):
    labels = tmp_path / "labels.txt"
    labels.write_bytes(codecs.BOM_UTF8 + b"Probe\nGr\xf6\xdfe\n")
    result = runner.invoke(
        app, ["init", run_dir, gel_image, "--n-lanes", "2", "--labels-file", str(labels)]
    )
    assert "line 2" in _error(result)["error"]


def test_a_byte_order_mark_is_not_part_of_the_first_label(gel_image, run_dir, tmp_path):
    """Notepad and Excel write one at the start of a UTF-8 file."""
    labels = tmp_path / "labels.txt"
    labels.write_bytes(codecs.BOM_UTF8 + "Größe\nProbe\n".encode("utf-8"))
    result = runner.invoke(
        app, ["init", run_dir, gel_image, "--n-lanes", "2", "--labels-file", str(labels)]
    )
    assert result.exit_code == 0, result.output
    recipe = json.loads((Path(run_dir) / "recipe.json").read_text(encoding="utf-8"))
    assert [lane["label"] for lane in recipe["lanes"]["lanes"]] == ["Größe", "Probe"]


def test_an_unknown_option_is_a_structured_usage_error(run_dir):
    error = _error(runner.invoke(app, ["lanes", "detect", run_dir, "--no-such-flag"]), exit_code=2)
    assert error["code"] == "usage_error"
    assert "--n-lanes" in error["available"]
    assert "gelpy lanes detect --help" in error["detail"]


def test_an_unknown_command_lists_the_commands():
    error = _error(runner.invoke(app, ["lanez"]), exit_code=2)
    assert error["code"] == "usage_error"
    assert "lanes" in error["available"]


def test_help_still_prints_help():
    result = runner.invoke(app, ["lanes", "detect", "--help"])
    assert result.exit_code == 0
    assert "--n-lanes" in result.output


def test_a_group_without_a_command_leaves_an_error_object_whenever_it_exits_non_zero():
    """`no_args_is_help` prints the help; click >= 8.2 exits 2 for it (older click exits 0).
    The documented rule holds either way: on a non-zero exit stderr is one error object."""
    for args in ([], ["lanes"]):
        done = subprocess.run(
            [sys.executable, "-m", "gelpy", *args], capture_output=True, text=True
        )
        assert "Traceback" not in done.stderr, args
        if done.returncode != 0:
            payload = json.loads(done.stderr)
            assert set(payload) == FIELDS and payload["code"] == "usage_error", args
            assert payload["error"] and payload["available"], args


def test_a_usage_error_without_a_message_still_says_what_failed():
    from gelpy.cli._io import USAGE_ERROR, usage_error_response

    response = usage_error_response(USAGE_ERROR(""))
    assert (response.code, response.error) == ("usage_error", "Missing command.")


def test_the_entry_point_answers_a_malformed_command_line_with_json():
    done = subprocess.run(
        [sys.executable, "-m", "gelpy", "fit", "run"], capture_output=True, text=True
    )
    assert done.returncode == 2
    assert done.stdout == ""
    assert json.loads(done.stderr)["code"] == "usage_error"


def test_a_closed_pipe_exits_quietly():
    """`gelpy schema FitConfig | true`: the reader leaves before any output is written.

    Typer turns a closed pipe into a quiet exit 1. `handle_errors` must not intercept it: it
    printed an `os_error` object, then Python's "Exception ignored while flushing sys.stdout",
    and exited 120. The pipe's read end is closed before the child exists, so its first write
    fails on every run; nothing depends on timing.
    """
    read_end, write_end = os.pipe()
    os.close(read_end)
    proc = subprocess.Popen(
        [sys.executable, "-m", "gelpy", "schema", "FitConfig"],
        stdin=subprocess.DEVNULL,
        stdout=write_end,
        stderr=subprocess.PIPE,
    )
    os.close(write_end)
    stderr = proc.stderr.read()
    proc.stderr.close()
    returncode = proc.wait(timeout=120)
    assert stderr == b""
    assert returncode == 1


def test_a_broken_pipe_in_a_command_body_is_no_error_object_from_any_layer(monkeypatch, run_dir):
    """Neither `handle_errors` nor the root group's catch-all may turn EPIPE into JSON: typer's
    own branch, which runs before the root's, exits 1 and says nothing."""

    _make_status_raise(monkeypatch, BrokenPipeError(errno.EPIPE, "Broken pipe"))
    result = runner.invoke(app, ["status", run_dir])
    assert result.exit_code == 1
    assert result.output == ""


def test_ctrl_c_in_a_command_body_exits_130_without_an_error_object(monkeypatch, run_dir):
    """The shell's own code for a run ended by SIGINT, and no JSON: the person who typed it knows."""

    _make_status_raise(monkeypatch, KeyboardInterrupt())
    result = runner.invoke(app, ["status", run_dir])
    assert result.exit_code == 130
    assert result.stdout == ""


@pytest.mark.parametrize(
    ("exc", "code"),
    [(PermissionError("denied"), "os_error"), (RuntimeError("boom"), "internal_error")],
    ids=["PermissionError", "RuntimeError"],
)
def test_any_exception_in_a_command_body_is_one_error_object(monkeypatch, run_dir, exc, code):
    """`handle_errors` answers every exception, not only the domain ones: no traceback."""

    _make_status_raise(monkeypatch, exc)
    error = _error(runner.invoke(app, ["status", run_dir]))
    assert error["code"] == code
    assert error["detail"]


def test_an_exception_outside_every_command_body_is_one_error_object():
    """The root group answers what `handle_errors` never saw (here: an undecorated command)."""
    mini = typer.Typer(cls=_StructuredErrors)

    @mini.command()
    def boom():
        raise PermissionError("denied")

    @mini.command()
    def other():
        """A second command, so that typer builds a group and not a single command."""

    error = _error(runner.invoke(mini, ["boom"]))
    assert error["code"] == "os_error"
    assert error["error"] == "denied"


def test_a_truncated_recipe_does_not_repeat_the_model_title(gel_image, run_dir):
    assert runner.invoke(app, ["init", run_dir, gel_image, "--n-lanes", "3"]).exit_code == 0
    recipe = Path(run_dir, "recipe.json")
    recipe.write_text(recipe.read_text(encoding="utf-8")[:50], encoding="utf-8")
    error = _error(runner.invoke(app, ["status", run_dir]))
    assert error["code"] == "validation_error"
    assert error["error"].startswith("Recipe: Invalid JSON")
    assert "Recipe: Recipe" not in error["error"]


def test_a_refit_that_matches_several_lanes_says_to_pass_exactly_one(gel_image, run_dir):
    assert runner.invoke(app, ["init", run_dir, gel_image, "--n-lanes", "3"]).exit_code == 0
    error = _error(runner.invoke(app, ["fit", "refit-lane", run_dir, "--lane", "re:."]))
    assert error["code"] == "lane_selection"
    assert error["available"] == ["1", "2", "3"]
    assert error["detail"] == "Pass exactly one lane: a label or an index."


def test_a_registry_error_names_a_next_step_that_holds_for_every_raiser(gel_image, run_dir):
    """`gelpy models list` describes fit strategies, shapes and background models, not the
    names `schema` accepts or the projection strategies, so the default step does not cite it."""
    assert runner.invoke(app, ["init", run_dir, gel_image, "--n-lanes", "3"]).exit_code == 0
    unknown_projection = [
        "lanes", "detect", run_dir, "--n-lanes", "3", "--no-plot", "--projection-strategy", "nope",
    ]
    for args in (["schema", "NoSuchModel"], unknown_projection):
        error = _error(runner.invoke(app, args))
        assert error["code"] == "registry", args
        assert error["detail"] == "Use one of the names in `available`.", args
        assert error["available"], args


@pytest.mark.parametrize("ending", [b"\r", b"\r\n", b"\n"], ids=["CR", "CR LF", "LF"])
def test_the_line_of_a_bad_byte_counts_every_kind_of_line_ending(gel_image, run_dir, tmp_path, ending):
    """A file written on an old Mac separates its lines with a lone CR: counting only LF named line 1
    however far into the file the bad byte was."""
    labels = tmp_path / "labels.txt"
    labels.write_bytes(ending.join([b"Probe", b"Messung", b"Gr\xf6\xdfe", b"Ende"]))
    result = runner.invoke(
        app, ["init", run_dir, gel_image, "--n-lanes", "4", "--labels-file", str(labels)]
    )
    assert "line 3" in _error(result)["error"]
