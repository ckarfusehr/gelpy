"""The skill's example scripts carry inline copies of gelpy's helpers (an example must not import
what is not public API). These tests keep the copies from drifting: the spreadsheet guard and the
CSV file, UTF-8 text, and the way a path recorded in a run folder is resolved."""

from __future__ import annotations

import csv
import importlib.util
import io
import os
import sys
from pathlib import Path

import pytest

from gelpy.runfolder.layout import RunPaths
from gelpy.runfolder.results import spreadsheet_safe as gelpy_spreadsheet_safe

EXAMPLES = Path(__file__).resolve().parents[2] / "plugin" / "skills" / "gelpy-analyze" / "examples"

CELLS = ["=1+1", "+x", "-y", "@z", "\tq", "\rr", "plain", "Größe", "", 3, -2.5, 0.0, None]


def _load(name):
    spec = importlib.util.spec_from_file_location(f"example_{name}", EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # a dataclass in the script looks its module up here
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fit_example():
    return _load("constrained_component_fit")


@pytest.fixture(scope="module")
def pdf_example():
    pytest.importorskip("reportlab")
    return _load("report_pdf")


@pytest.mark.parametrize("value", CELLS, ids=repr)
def test_the_inline_spreadsheet_guard_is_gelpys(fit_example, value):
    assert fit_example.spreadsheet_safe(value) == gelpy_spreadsheet_safe(value)


def test_the_csv_is_utf8_and_guards_every_text_cell(fit_example, tmp_path):
    rows = [{"label": "=1+1", "name": "Größe", "area": -2.5}, {"label": "b", "name": "c", "area": 3}]
    target = tmp_path / "fits.csv"
    fit_example.write_csv(target, rows, ["label", "name", "area"])
    assert "Größe".encode("utf-8") in target.read_bytes()  # whatever the locale is
    with open(target, newline="", encoding="utf-8") as handle:
        assert list(csv.reader(handle))[1] == ["'=1+1", "Größe", "-2.5"]


def test_a_cell_with_a_carriage_return_quotes_every_text_cell_as_gelpys_results_csv_does(fit_example, tmp_path):
    """Python 3.10's csv module leaves a bare CR unquoted: the second layer must hold there."""
    rows = [{"label": "a\rb", "name": "plain", "area": 3}]
    target = tmp_path / "fits.csv"
    fit_example.write_csv(target, rows, ["label", "name", "area"])
    text = target.read_bytes().decode("utf-8")
    lines = text.split("\r\n")  # the csv module's own line ending; a bare CR is inside a quoted cell
    assert lines[1] == '"a\rb","plain",3'
    assert lines[0] == '"label","name","area"'
    assert list(csv.reader(io.StringIO(text, newline="")))[1] == ["a\rb", "plain", "3"]


def test_the_examples_name_an_encoding_for_every_text_file_and_subprocess():
    import ast

    from tests.test_source_rules import _text_io_without_utf8

    offenders = [
        f"{path.name}:{line}  {call}"
        for path in sorted(EXAMPLES.glob("*.py"))
        for line, call in _text_io_without_utf8(ast.parse(path.read_text(encoding="utf-8")))
    ]
    assert not offenders, offenders


# --- a path the run folder recorded -------------------------------------------------------------
@pytest.fixture
def layout(tmp_path, monkeypatch):
    """A run folder two levels below ``tmp_path/data``'s sibling, and a working directory that is none
    of its relatives, so a path resolved from the wrong base points nowhere."""
    run = tmp_path / "analysis" / "run"
    run.mkdir(parents=True)
    (tmp_path / "analysis" / "gel.tif").write_bytes(b"gel")
    elsewhere = tmp_path / "elsewhere" / "deeper"
    elsewhere.mkdir(parents=True)
    monkeypatch.chdir(elsewhere)
    return run, elsewhere


@pytest.mark.parametrize("stored", ["../gel.tif", "gel.tif", "missing.tif"])
def test_the_example_resolves_a_recorded_path_as_gelpy_does(pdf_example, layout, stored):
    run, _ = layout
    assert Path(pdf_example.recorded_path(str(run), stored)).resolve() == RunPaths(run).resolve(stored).resolve()


def test_a_path_recorded_relative_to_the_run_folder_is_found_from_any_directory(pdf_example, layout):
    run, _ = layout
    found = pdf_example.recorded_path(str(run), "../gel.tif")
    assert os.path.exists(found)


def test_a_path_recorded_by_gelpy_1_0_is_still_found_from_its_directory(pdf_example, layout):
    run, elsewhere = layout
    (elsewhere / "gel.tif").write_bytes(b"gel")
    assert Path(pdf_example.recorded_path(str(run), "gel.tif")).resolve() == (elsewhere / "gel.tif").resolve()


def test_the_lane_qc_figure_is_the_lanes_as_set(pdf_example, tmp_path):
    from gelpy.cli import tools

    names = [name for name, _ in pdf_example.STANDARD]
    assert "lane_detect.png" not in names and "non_linear_lanes.png" in names

    import numpy as np
    from skimage import io

    image = tmp_path / "gel.tif"
    io.imsave(str(image), (np.random.default_rng(0).random((120, 160)) * 0.5).astype(np.float32))
    run = str(tmp_path / "run")
    ok, answer = tools.call_tool_safely("init", {"run_dir": run, "image_path": str(image), "n_lanes": 3, "width": 20})
    assert ok, answer
    pdf_example.render_lane_view(run)
    assert (Path(run) / "render" / "non_linear_lanes.png").exists()
