"""results.csv never hands a spreadsheet a formula: a text cell that starts one is text."""

import csv
import io

import pytest

from gelpy.runfolder.results import results_csv_text, spreadsheet_safe
from gelpy.schemas.peaks import BandFit, GaussianPeakParams, LaneFitResult
from gelpy.schemas.results import AnalysisResult


def _rows(result):
    return list(csv.DictReader(io.StringIO(results_csv_text(result))))


def _lanes(labels):
    """One lane per label, none with bands: each lane is exactly one record of the file."""
    return AnalysisResult(
        lanes=[
            LaneFitResult(lane_index=index, label=label, status="empty")
            for index, label in enumerate(labels)
        ]
    )


def _records(text):
    """The records a CSV reader finds, reading the way the csv module documents for a file
    (``newline=""``): line endings reach the reader untranslated, so a bare CR ends a line."""
    return list(csv.reader(io.StringIO(text, newline="")))


def test_text_cells_that_start_a_formula_are_written_as_text():
    lane = LaneFitResult(
        lane_index=0, label='=HYPERLINK("http://example.invalid")', status="failed", error="-1 bands"
    )
    [row] = _rows(AnalysisResult(lanes=[lane]))
    assert row["label"] == "'=HYPERLINK(\"http://example.invalid\")"
    assert row["error"] == "'-1 bands"


@pytest.mark.parametrize(
    ("text", "written"),
    [
        ("+1", "'+1"),
        ("-x", "'-x"),
        ("@sum", "'@sum"),
        ("\tpad", "'\tpad"),
        ("\rcr", "'\rcr"),
        ("plain", "plain"),
        ("1-2", "1-2"),
        ("", ""),
    ],
)
def test_only_formula_prefixes_are_neutralised(text, written):
    assert spreadsheet_safe(text) == written


def test_numbers_are_never_prefixed():
    band = BandFit(
        band_number=0,
        area=1.0,
        area_fraction=1.0,
        maximum_position=-3.5,
        params=GaussianPeakParams(amplitude=1.0, center=-3.5, sigma=2.0),
    )
    [row] = _rows(AnalysisResult(lanes=[LaneFitResult(lane_index=0, label="a", bands=[band])]))
    assert row["maximum_position"] == "-3.5"
    assert spreadsheet_safe(-3.5) == -3.5


def test_a_carriage_return_in_a_cell_cannot_split_a_record():
    """Python 3.10's csv writer leaves a field with a bare CR unquoted (3.11 and later quote it).
    A reader that takes a bare CR for a record break then splits the record, and the text after
    the CR starts a record of its own: possibly a live formula. So the file is written to read back
    the same on every Python: one record per lane, every label as written."""
    labels = ["\r=1", "x\r=1+1", "a,b", 'say "hi"', "line one\n=2"]
    written = ["'\r=1", "x\r=1+1", "a,b", 'say "hi"', "line one\n=2"]  # the first starts with a CR
    header, *records = _records(results_csv_text(_lanes(labels)))
    assert len(records) == len(labels)
    assert all(len(record) == len(header) for record in records)
    assert [record[header.index("label")] for record in records] == written
    assert not [record[0] for record in records if record[0].startswith(("=", "+", "-", "@"))]


def test_numbers_stay_numbers_when_a_carriage_return_forces_quoting():
    band = BandFit(
        band_number=0,
        area=1.0,
        area_fraction=1.0,
        maximum_position=-3.5,
        params=GaussianPeakParams(amplitude=1.0, center=-3.5, sigma=2.0),
    )
    text = results_csv_text(
        AnalysisResult(lanes=[LaneFitResult(lane_index=0, label="x\ry", bands=[band])])
    )
    header, *records = _records(text)
    [record] = records
    assert record[header.index("label")] == "x\ry"
    assert record[header.index("maximum_position")] == "-3.5"
    assert ",-3.5," in text  # not quoted: only text is


def test_text_without_a_carriage_return_is_quoted_only_where_csv_needs_it():
    text = results_csv_text(_lanes(["a", "=1+1", "x,y"]))
    assert text.count('"') == 2  # the comma, nothing else: no blanket quoting
