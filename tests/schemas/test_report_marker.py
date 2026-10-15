"""The report marker lists only names a report build writes: it can never name anything else.

A marker is a plain JSON file that anyone can write into a folder, and ``report html`` and
``init --force`` remove what it lists. So a marker that names a file a build does not write is no
marker at all, whatever else it says.
"""

import pytest
from pydantic import ValidationError

from gelpy.schemas.report import REPORT_MARKER, ReportMarker, is_report_file

#: Every name a build writes, in each of the places it writes them.
WRITTEN = [
    "index.html",
    "report.css",
    "theme.css",
    "img/linear.png",
    "img/non_linear_lanes.png",
    "img/profiles_overview.png",
    "img/fits/lane_000.png",
    "img/profiles_µg_Ω.png",  # a label with letters of any alphabet names a profile plot
]

NOT_WRITTEN = [
    # the run's own files, and what sits around a report
    "recipe.json",
    "fit_config.json",
    "manifest.json",
    "results.csv",
    "report.md",
    "cache/fits/lane_000__abc.json",
    "render/linear.png",
    REPORT_MARKER,  # the marker never lists itself
    # not where a build writes it
    "x.png",
    "img/fits/x.png/y",
    "img/a/b.png",
    "img/fits/a/b.png",
    "img",
    "img/",
    "img/fits",
    "img/fits/",
    "img/x.jpg",
    "img/x.PNG",
    "img/x.png.bak",
    "index.htm",
    "INDEX.HTML",
    "index.html/",
    "report/index.html",
    "theme.css.bak",
    # a name that leads out of the folder, or is not one name
    "",
    "/index.html",
    "./index.html",
    "../recipe.json",
    "img/../recipe.json",
    "img/fits/../x.png",
    "img/..png",
    "img/a..b.png",
    "img/.png",
    "img/.hidden.png",
    "img/fits/.hidden.png",
    "img//x.png",
    "img\\x.png",
    "img/a\\b.png",
    "img/c:x.png",
    "C:/img/x.png",
    "img/a\x00b.png",
    "img/a\nb.png",
    "img/a\x1bb.png",
    "img/a\x7fb.png",
]


@pytest.mark.parametrize("name", WRITTEN)
def test_a_name_a_build_writes_is_accepted(name):
    assert is_report_file(name)
    assert ReportMarker(gelpy_version="1", fingerprint="", files=[name]).files == [name]


def test_a_marker_of_everything_a_build_writes_is_valid():
    assert ReportMarker(gelpy_version="1", fingerprint="f", files=WRITTEN).files == WRITTEN


@pytest.mark.parametrize("name", NOT_WRITTEN)
def test_any_other_name_makes_the_whole_marker_no_marker(name):
    assert not is_report_file(name)
    with pytest.raises(ValidationError):
        ReportMarker(gelpy_version="1", fingerprint="", files=[name])
    with pytest.raises(ValidationError):  # one such name spoils the others
        ReportMarker(gelpy_version="1", fingerprint="", files=["index.html", name, "report.css"])


# --- the length of a name: a file system takes at most 255 bytes of one --------------------------------
#: A name of 256 bytes or more cannot exist anywhere, and a marker that lists one made every rebuild
#: and ``init --force`` fail with "File name too long": such a marker is no marker.
LONG_ENOUGH = [
    pytest.param("img/" + "x" * 251 + ".png", id="a leaf of 255 bytes"),
    pytest.param("img/fits/" + chr(0xB5) * 125 + ".png", id="a leaf of 254 bytes, 125 two-byte letters"),
]
TOO_LONG = [
    pytest.param("img/" + "x" * 252 + ".png", id="a leaf of 256 bytes"),
    pytest.param("img/" + chr(0xB5) * 126 + ".png", id="a leaf of 256 bytes, 126 two-byte letters"),
    pytest.param("img/fits/" + "x" * 5000 + ".png", id="a leaf of 5004 bytes"),
]


@pytest.mark.parametrize("name", LONG_ENOUGH)
def test_a_name_of_up_to_255_bytes_is_accepted(name):
    assert is_report_file(name)
    ReportMarker(gelpy_version="1", fingerprint="", files=[name])


@pytest.mark.parametrize("name", TOO_LONG)
def test_a_name_of_256_bytes_or_more_makes_the_whole_marker_no_marker(name):
    assert not is_report_file(name)
    with pytest.raises(ValidationError):
        ReportMarker(gelpy_version="1", fingerprint="", files=["index.html", name])
