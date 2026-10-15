"""`gelpy report html`: a static, escaped, script-free report folder."""

import json
import os
import re
import shutil
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gelpy import render as render_package
from gelpy.cli.app import app
from gelpy.render.html_report import DEFAULT_CSS

runner = CliRunner()
HOSTILE = ["<script>alert(1)</script>", 'a"b', "x & y", "=SUM(A1)"]
MARKER = ".gelpy-report.json"
PNG = b"\x89PNG\r\n\x1a\n"
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="symlinks need extra privileges")


class _Tags(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags, self.attributes = [], []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        self.attributes.extend(attrs)


def _parse(page):
    parsed = _Tags()
    parsed.feed(page)
    return parsed


def _run(*args):
    result = runner.invoke(app, list(args))
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def _page(run_dir, *extra):
    built = _run("report", "html", run_dir, *extra)
    return (Path(run_dir) / built["index"]).read_text(encoding="utf-8")


@pytest.fixture
def fitted(gel_image, run_dir, tmp_path):
    labels = tmp_path / "labels.txt"
    labels.write_text("\n".join(HOSTILE) + "\n", encoding="utf-8")
    _run("init", run_dir, gel_image, "--n-lanes", "4", "--width", "30")
    _run("lanes", "set", run_dir, "--linspace", "40", "250", "4", "--labels-file", str(labels))
    _run("background", "remove", run_dir, "--model", "plane_ransac")
    _run("fit", "run", run_dir, "--lane", "0", "--lane", "1", "--plot")
    return run_dir


def test_the_report_is_a_self_contained_folder(fitted):
    built = _run("report", "html", fitted)
    assert built["index"] == "report/index.html"
    index = Path(fitted) / built["index"]
    page = index.read_text(encoding="utf-8")
    for name, value in _parse(page).attributes:
        if name in ("src", "href"):
            assert (index.parent / value).is_file(), value
    assert {"index.html", "report.css", "img/fits/lane_000.png", "img/fits/lane_001.png"} <= set(
        built["files"]
    )
    for heading in ("Gel", "Lanes", "Results", "Profiles", "Fits", "Provenance"):
        assert f"<h2>{heading}</h2>" in page


def test_the_page_has_no_scripts_inline_styles_or_outside_references(fitted):
    parsed = _parse(_page(fitted))
    assert not {"script", "style", "iframe", "object", "embed"} & set(parsed.tags)
    assert not [name for name, _ in parsed.attributes if name.startswith("on") or name == "style"]
    for name, value in parsed.attributes:
        if name in ("src", "href"):
            assert not re.match(r"[a-z][a-z0-9+.-]*:|//", value), value


def test_labels_are_escaped_in_the_report(fitted):
    page = _page(fitted)
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert "a&quot;b" in page and "x &amp; y" in page


def test_a_stylesheet_passed_with_css_is_linked_after_the_default(fitted, tmp_path):
    theme = tmp_path / "theme-source.css"
    theme.write_text(":root { --gelpy-accent: #123456; }\n", encoding="utf-8")
    built = _run("report", "html", fitted, "--css", str(theme))
    folder = (Path(fitted) / built["index"]).parent
    page = (folder / "index.html").read_text(encoding="utf-8")
    assert page.index('href="report.css"') < page.index('href="theme.css"')
    assert (folder / "theme.css").read_text(encoding="utf-8") == theme.read_text(encoding="utf-8")


def test_the_default_stylesheet_is_built_on_css_variables():
    defined = set(re.findall(r"(--gelpy-[a-z-]+)\s*:", DEFAULT_CSS))
    used = set(re.findall(r"var\((--gelpy-[a-z-]+)\)", DEFAULT_CSS))
    assert {"--gelpy-bg", "--gelpy-fg", "--gelpy-accent", "--gelpy-font"} <= defined
    assert used <= defined


def test_two_builds_of_one_run_are_identical(fitted):
    assert _page(fitted) == _page(fitted)


def test_only_plots_that_show_another_fit_are_rendered_again(fitted, monkeypatch):
    _run("report", "html", fitted)
    drawn = []
    original = render_package.render_lane_fit
    monkeypatch.setattr(
        render_package, "render_lane_fit", lambda *a, **k: drawn.append(a) or original(*a, **k)
    )
    _run("report", "html", fitted)
    assert drawn == []
    _run("fit", "refit-lane", fitted, "--lane", "1", "--seed", "3")
    _run("report", "html", fitted)
    assert len(drawn) == 1


def test_results_and_report_carry_the_current_labels(fitted, tmp_path):
    relabel = tmp_path / "new.txt"
    relabel.write_text("w\nx\ny\nz\n", encoding="utf-8")
    _run("lanes", "set", fitted, "--labels-file", str(relabel))
    assert [lane["label"] for lane in _run("results", "show", fitted)["lanes"]] == ["w", "x"]
    assert "<td>w</td>" in _page(fitted)


def test_the_report_can_go_outside_the_run_folder(fitted, tmp_path):
    built = _run("report", "html", fitted, "--out", str(tmp_path / "site"))
    assert Path(built["index"]) == tmp_path / "site" / "index.html"
    assert (tmp_path / "site" / "report.css").is_file()


def test_init_force_removes_the_report(fitted, gel_image):
    _run("report", "html", fitted)
    _run("init", fitted, gel_image, "--n-lanes", "4", "--force")
    assert not (Path(fitted) / "report").exists()


# --- labels in plots, runs with nothing to show, hostile names, unreadable stylesheets -----------
def test_a_relabelled_lane_is_plotted_again_with_its_current_label(fitted, tmp_path, monkeypatch):
    """The fit plot is titled with the lane's label, which the fit hash leaves out: a plot that
    shows the right fit under the old label is out of date, and the page captions it with the new
    one."""
    _run("report", "html", fitted)
    relabel = tmp_path / "new.txt"
    relabel.write_text("w\nx\ny\nz\n", encoding="utf-8")
    _run("lanes", "set", fitted, "--labels-file", str(relabel))
    drawn = []
    original = render_package.render_lane_fit
    monkeypatch.setattr(
        render_package,
        "render_lane_fit",
        lambda *a, **k: drawn.append(a[2].label) or original(*a, **k),
    )
    _run("report", "html", fitted)
    assert drawn == ["w", "x"]  # the two fitted lanes, under their new labels
    _run("report", "html", fitted)
    assert drawn == ["w", "x"]  # and once is enough


def test_a_run_without_background_or_fits_still_has_a_report(gel_image, run_dir):
    _run("init", run_dir, gel_image, "--n-lanes", "4", "--width", "30")
    built = _run("report", "html", run_dir)
    page = (Path(run_dir) / built["index"]).read_text(encoding="utf-8")
    assert "No current fits." in page
    assert "<h2>Profiles</h2>" not in page and "<h2>Fits</h2>" not in page
    assert not [name for name in built["files"] if name.startswith("img/fits/")]


def test_a_stale_fit_is_marked_and_not_reported(fitted):
    _run("lanes", "set", fitted, "--width", "20")  # a new strip width: both fits are out of date
    page = _page(fitted)
    assert page.count("<td>stale</td>") == 2
    assert "No current fits." in page
    assert "<h2>Fits</h2>" not in page


def test_the_default_stylesheet_refers_to_nothing_outside_itself():
    assert not re.search(r"url\(|@import|@font-face|https?:", DEFAULT_CSS)


@pytest.mark.skipif(sys.platform == "win32", reason="Windows file names cannot hold these characters")
def test_the_names_of_the_run_folder_and_the_image_are_escaped(gel_image, tmp_path):
    image = tmp_path / "a<b>&.tif"  # a path has no slash to spend on a closing tag
    shutil.copyfile(gel_image, image)
    run = tmp_path / '<img src=x onerror="alert(1)"> & x'
    _run("init", str(run), str(image), "--n-lanes", "4", "--width", "30")
    page = _page(str(run))
    assert "<img src=x" not in page and "<b>" not in page
    assert "&lt;img src=x onerror=&quot;alert(1)&quot;&gt; &amp; x" in page
    assert "a&lt;b&gt;&amp;.tif" in page


@pytest.mark.parametrize("stylesheet", ["missing.css", "latin1.css"])
def test_a_stylesheet_that_cannot_be_read_is_refused_before_anything_is_written(
    fitted, tmp_path, stylesheet
):
    path = tmp_path / stylesheet
    if stylesheet == "latin1.css":
        path.write_bytes(b"/* caf\xe9 */ :root { --gelpy-accent: #123456; }\n")
    result = runner.invoke(app, ["report", "html", fitted, "--css", str(path)])
    assert result.exit_code == 1, result.output
    error = json.loads(result.stderr)
    assert error["code"] == ("not_found" if stylesheet == "missing.css" else "invalid_value")
    if stylesheet == "latin1.css":  # it names the file and what to do, not a codec and a bug
        assert "latin1.css" in error["error"] and "UTF-8" in error["detail"]
        assert "bug" not in error["detail"]
    assert not (Path(fitted) / "report").exists()


@pytest.mark.parametrize("ending", [b"\r", b"\r\n", b"\n"], ids=["CR", "CR LF", "LF"])
def test_the_line_of_a_bad_byte_in_a_stylesheet_counts_every_kind_of_line_ending(fitted, tmp_path, ending):
    path = tmp_path / "legacy.css"
    path.write_bytes(ending.join([b":root { --gelpy-accent: #123456; }", b"/* caf\xe9 */", b"a { color: red; }"]))
    result = runner.invoke(app, ["report", "html", fitted, "--css", str(path)])
    assert result.exit_code == 1, result.output
    assert "line 2" in json.loads(result.stderr)["error"]


# === the report owns its folder, writes atomically, and says when it is out of date ====
def _fail(*args):
    result = runner.invoke(app, list(args))
    assert result.exit_code == 1, result.output
    return json.loads(result.stderr)


def _relabel(run, tmp_path, *labels):
    path = tmp_path / "labels-new.txt"
    path.write_text("\n".join(labels) + "\n", encoding="utf-8")
    _run("lanes", "set", run, "--labels-file", str(path))


def _tree(folder):
    """Every file under ``folder`` (links not followed) as {relative path: bytes}."""
    found = {}
    for here, _, names in os.walk(folder):
        for name in names:
            path = Path(here) / name
            data = b"<link>" if path.is_symlink() else path.read_bytes()
            found[path.relative_to(folder).as_posix()] = data
    return found


def _marker(folder):
    return json.loads((Path(folder) / MARKER).read_text(encoding="utf-8"))


def _cached_fit(run, lane):
    return next(Path(run).glob(f"cache/fits/lane_{lane:03d}__*.json"))


class _Elements(HTMLParser):
    def __init__(self):
        super().__init__()
        self.elements, self.text = [], []

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))

    def handle_data(self, data):
        self.text.append(data)


def _elements(page):
    parsed = _Elements()
    parsed.feed(page)
    return parsed.elements


# --- the marker says what is the report's ------------------------------------------
def test_the_build_marks_its_folder_with_the_files_it_wrote(fitted):
    built = _run("report", "html", fitted)
    marker = _marker(Path(fitted) / "report")
    assert marker["files"] == built["files"] and MARKER not in built["files"]
    assert marker["gelpy_version"] and re.fullmatch(r"[0-9a-f]{16}", marker["fingerprint"])
    assert set(_tree(Path(fitted) / "report")) == {*built["files"], MARKER}


def test_a_folder_that_holds_other_files_is_refused_and_left_alone(fitted, tmp_path):
    other = tmp_path / "mine"
    other.mkdir()
    (other / "index.html").write_text("mine", encoding="utf-8")
    error = _fail("report", "html", fitted, "--out", str(other))
    assert error["code"] == "invalid_value"
    assert "empty or new folder" in error["detail"] and "earlier report html" in error["detail"]
    assert _tree(other) == {"index.html": b"mine"}


@pytest.mark.parametrize("where", ["", "cache", "render"])
def test_the_run_folder_and_its_inner_folders_are_never_a_target(fitted, where):
    run = Path(fitted)
    before = _tree(run)
    target = run / where if where else run
    assert _fail("report", "html", fitted, "--out", str(target))["code"] == "invalid_value"
    assert _tree(run) == before


def test_a_run_whose_own_report_folder_holds_the_users_files_is_refused(gel_image, run_dir):
    """A gelpy 1.0 run folder: its report/ is not gelpy's."""
    _run("init", run_dir, gel_image, "--n-lanes", "4", "--width", "30")
    notes = Path(run_dir) / "report" / "notes.txt"
    notes.parent.mkdir()
    notes.write_text("my notes", encoding="utf-8")
    assert _fail("report", "html", run_dir)["code"] == "invalid_value"
    assert _tree(Path(run_dir) / "report") == {"notes.txt": b"my notes"}


def test_a_new_or_an_empty_folder_is_a_target(fitted, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    for target in (empty, tmp_path / "new" / "deeper"):
        _run("report", "html", fitted, "--out", str(target))
        assert (target / "index.html").is_file() and (target / MARKER).is_file()


def test_a_rebuild_replaces_what_the_earlier_build_wrote_and_nothing_else(fitted, tmp_path):
    theme = tmp_path / "t.css"
    theme.write_text(":root { --gelpy-bg: #fafafa; }\n", encoding="utf-8")
    _run("report", "html", fitted, "--css", str(theme))
    folder = Path(fitted) / "report"
    assert "theme.css" in _marker(folder)["files"]
    (folder / "notes.txt").write_text("mine", encoding="utf-8")
    (folder / "img" / "mine.png").write_bytes(b"mine")
    built = _run("report", "html", fitted)  # no --css this time
    assert "theme.css" not in built["files"] and not (folder / "theme.css").exists()
    assert (folder / "notes.txt").read_text(encoding="utf-8") == "mine"
    assert (folder / "img" / "mine.png").read_bytes() == b"mine"
    assert _marker(folder)["files"] == built["files"]


def test_a_rebuild_removes_the_plots_of_fits_that_went_away_and_their_folder(fitted):
    _run("report", "html", fitted)
    folder = Path(fitted) / "report"
    assert (folder / "img" / "fits" / "lane_000.png").is_file()
    _run("lanes", "set", fitted, "--width", "20")  # both fits are out of date
    built = _run("report", "html", fitted)
    assert not [name for name in built["files"] if name.startswith("img/fits/")]
    assert not (folder / "img" / "fits").exists() and (folder / "img" / "linear.png").is_file()
    assert _marker(folder)["files"] == built["files"]


def test_a_marker_that_does_not_parse_means_the_folder_is_not_the_reports(fitted):
    _run("report", "html", fitted)
    folder = Path(fitted) / "report"
    (folder / MARKER).write_text("{not json", encoding="utf-8")
    before = _tree(folder)
    assert _fail("report", "html", fitted)["code"] == "invalid_value"
    assert _tree(folder) == before


@pytest.mark.parametrize(
    "name", ["../recipe.json", "img/../../recipe.json", "ABSOLUTE", "a\\b", "", "img//x.png", MARKER]
)
def test_a_marker_cannot_name_a_file_outside_its_folder(fitted, tmp_path, name):
    victim = tmp_path / "victim.txt"
    victim.write_text("precious", encoding="utf-8")
    name = str(victim) if name == "ABSOLUTE" else name
    folder = Path(fitted) / "report"
    folder.mkdir()
    (folder / "index.html").write_text("x", encoding="utf-8")
    listed = {"gelpy_version": "1", "fingerprint": "", "files": ["index.html", name]}
    (folder / MARKER).write_text(json.dumps(listed), encoding="utf-8")
    recipe = (Path(fitted) / "recipe.json").read_bytes()
    assert _fail("report", "html", fitted)["code"] == "invalid_value"
    assert (Path(fitted) / "recipe.json").read_bytes() == recipe
    assert victim.read_text(encoding="utf-8") == "precious"


# --- nothing is written or read through a link --------------------------------------------
@posix_only
def test_a_link_at_a_destination_is_replaced_and_its_target_is_untouched(fitted, tmp_path):
    _run("report", "html", fitted)
    victim = tmp_path / "victim.txt"
    victim.write_text("precious", encoding="utf-8")
    dest = Path(fitted) / "report" / "img" / "linear.png"
    dest.unlink()
    dest.symlink_to(victim)
    _run("report", "html", fitted)
    assert victim.read_text(encoding="utf-8") == "precious"
    assert not dest.is_symlink() and dest.read_bytes().startswith(PNG)


@posix_only
def test_a_link_where_the_report_needs_a_folder_is_refused(fitted, tmp_path):
    _run("report", "html", fitted)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "linear.png").write_text("precious", encoding="utf-8")
    folder = Path(fitted) / "report"
    shutil.rmtree(folder / "img")
    (folder / "img").symlink_to(elsewhere, target_is_directory=True)
    error = _fail("report", "html", fitted)
    assert error["code"] == "invalid_value" and "img" in error["error"] and error["detail"]
    assert _tree(elsewhere) == {"linear.png": b"precious"}


@posix_only
def test_a_plot_that_is_a_link_is_refused_not_copied_into_the_report(fitted, tmp_path):
    _run("report", "html", fitted)
    secret = tmp_path / "secret.txt"
    secret.write_text("SECRET (a private file)", encoding="utf-8")
    plot = Path(fitted) / "render" / "fits" / "lane_000.png"
    plot.unlink()
    plot.symlink_to(secret)
    error = _fail("report", "html", fitted)
    assert error["code"] == "invalid_value" and "lane_000.png" in error["error"] and error["detail"]
    assert all(b"SECRET" not in data for data in _tree(Path(fitted) / "report").values())


@posix_only
def test_a_link_in_render_is_replaced_by_the_command_that_draws_it(fitted, tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_text("precious", encoding="utf-8")
    plot = Path(fitted) / "render" / "linear.png"
    plot.unlink(missing_ok=True)
    plot.symlink_to(victim)
    _run("image", "show", fitted, "--kind", "linear")
    assert victim.read_text(encoding="utf-8") == "precious" and not plot.is_symlink()


def test_a_page_that_cannot_be_built_leaves_the_folder_as_it_was(fitted, monkeypatch):
    _run("report", "html", fitted)
    folder = Path(fitted) / "report"
    before = _tree(folder)
    from gelpy.render import html_report

    def refuse(*args, **kwargs):
        raise UnicodeEncodeError("utf-8", "x", 0, 1, "surrogates not allowed")

    monkeypatch.setattr(html_report, "build_page", refuse)
    assert runner.invoke(app, ["report", "html", fitted]).exit_code == 1
    assert _tree(folder) == before


# --- the plot index never claims more than the plots show -----------------------------------------
def test_a_plot_interrupted_while_it_is_drawn_is_drawn_again(fitted, tmp_path, monkeypatch):
    from gelpy.cli import _render

    _relabel(fitted, tmp_path, "X0", "X1", "y", "z")
    original = render_package.render_lane_fit
    calls = []

    def killed_on_the_second(*a, **k):
        calls.append(a[2].label)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return original(*a, **k)

    monkeypatch.setattr(render_package, "render_lane_fit", killed_on_the_second)
    with pytest.raises(KeyboardInterrupt):
        _render.render_fits(fitted, [0, 1])
    index = json.loads((Path(fitted) / "render" / "fits" / "index.json").read_text(encoding="utf-8"))
    assert [(lane, entry["label"]) for lane, entry in sorted(index.items())] == [("0", "X0")]

    _relabel(fitted, tmp_path, *HOSTILE)  # lane 0's plot still says X0
    drawn = []
    monkeypatch.setattr(
        render_package, "render_lane_fit", lambda *a, **k: drawn.append(a[2].label) or original(*a, **k)
    )
    _run("report", "html", fitted)
    assert drawn == [HOSTILE[0], HOSTILE[1]]


def test_relabelling_one_lane_redraws_only_that_lane(fitted, tmp_path, monkeypatch):
    _run("report", "html", fitted)
    _relabel(fitted, tmp_path, HOSTILE[0], "only this one", "y", "z")
    drawn = []
    original = render_package.render_lane_fit
    monkeypatch.setattr(
        render_package, "render_lane_fit", lambda *a, **k: drawn.append(a[2].label) or original(*a, **k)
    )
    _run("report", "html", fitted)
    assert drawn == ["only this one"]


# --- a report older than the run is detectable ------------------------------------------------------
def test_status_says_whether_the_report_is_current(fitted, tmp_path):
    assert _run("status", fitted)["report"] is None
    _run("report", "html", fitted)
    assert _run("status", fitted)["report"] == {"path": "report/index.html", "current": True}

    _relabel(fitted, tmp_path, "w", "x", "y", "z")
    assert _run("status", fitted)["report"]["current"] is False
    _run("report", "html", fitted)
    assert _run("status", fitted)["report"]["current"] is True

    _run("fit", "refit-lane", fitted, "--lane", "1", "--seed", "5")
    assert _run("status", fitted)["report"]["current"] is False
    _run("report", "html", fitted)
    assert _run("status", fitted)["report"]["current"] is True

    recipe_file = Path(fitted) / "recipe.json"  # a hand edit, which no command recorded
    recipe = json.loads(recipe_file.read_text(encoding="utf-8"))
    recipe["lanes"]["lanes"][2]["label"] = "edited by hand"
    recipe_file.write_text(json.dumps(recipe), encoding="utf-8")
    assert _run("status", fitted)["report"]["current"] is False


def test_the_page_shows_the_fingerprint_the_marker_holds(fitted):
    page = _page(fitted)
    fingerprint = _marker(Path(fitted) / "report")["fingerprint"]
    assert f"<dt>report fingerprint</dt><dd>{fingerprint}</dd>" in page


def test_a_report_folder_without_a_marker_is_no_report_to_status(gel_image, run_dir):
    _run("init", run_dir, gel_image, "--n-lanes", "4", "--width", "30")
    (Path(run_dir) / "report").mkdir()
    (Path(run_dir) / "report" / "notes.txt").write_text("mine", encoding="utf-8")
    assert _run("status", run_dir)["report"] is None


def test_a_build_that_stops_half_way_is_not_current_and_is_not_refused_next_time(fitted, monkeypatch):
    from gelpy.render import html_report

    real = html_report.atomic_write

    def killed_before_the_page(path, content):
        if Path(path).name == "index.html":
            raise RuntimeError("killed")
        return real(path, content)

    monkeypatch.setattr(html_report, "atomic_write", killed_before_the_page)
    assert runner.invoke(app, ["report", "html", fitted]).exit_code == 1
    assert _run("status", fitted)["report"] == {"path": "report/index.html", "current": False}
    monkeypatch.setattr(html_report, "atomic_write", real)
    _run("report", "html", fitted)  # the half-built folder is the report's to finish
    assert _run("status", fitted)["report"]["current"] is True


# --- every variable does something -----------------------------------------------------------------------
def _rich_page():
    """A page with every kind of block, built without a run folder."""
    from gelpy.render.html_report import Figure, ReportContent, build_report
    from gelpy.schemas.background import BackgroundStageConfig
    from gelpy.schemas.fit_config import FitConfigStore, PerLaneFitConfig
    from gelpy.schemas.image import ImageRef
    from gelpy.schemas.lanes import LaneDefinition, LaneLayout
    from gelpy.schemas.peaks import BandFit, GaussianPeakParams, LaneFitResult
    from gelpy.schemas.recipe import Recipe
    from gelpy.schemas.results import AnalysisResult
    from gelpy.schemas.run import LaneStatus, RunStatus

    ref = ImageRef(path="gel.tif", sha256="a" * 64, size=1, mtime=1.0, height=250, width=300, dtype="f4")
    lanes = [LaneDefinition(index=i, label=f"lane {i}", x_center_px=40.0 + 70 * i) for i in range(4)]
    recipe = Recipe(
        source_image=ref,
        lanes=LaneLayout(lanes=lanes, profile_width_px=30),
        background=BackgroundStageConfig(model="plane_ransac"),
    )
    band = BandFit(
        band_number=0, component_class="sample", area=1.0, area_fraction=0.9, maximum_position=80.0,
        params=GaussianPeakParams(amplitude=1.0, center=80.0, sigma=5.0),
    )
    ok = LaneFitResult(lane_index=0, label="lane 0", bands=[band], coverage=1.0)
    low = LaneFitResult(
        lane_index=1, label="lane 1", bands=[band], confidence="low", coverage=1.4,
        confidence_reasons=["band 0 overlaps a neighbor (areas correlated)"],
    )
    empty = LaneFitResult(
        lane_index=2, label="lane 2", status="empty", confidence="low",
        confidence_reasons=["no significant signal detected"],
    )
    states = [
        LaneStatus(index=0, label="lane 0", has_fit=True, fit_current=True, status="ok", n_bands=1),
        LaneStatus(index=1, label="lane 1", has_fit=True, fit_current=True, status="ok", n_bands=1),
        LaneStatus(index=2, label="lane 2", has_fit=True, fit_current=True, status="empty"),
        LaneStatus(index=3, label="lane 3", has_fit=False),
    ]
    status = RunStatus(
        run_dir="run", source_image="gel.tif", image_width=300, image_height=250,
        warnings=["image polarity is low-confidence (skew unclear); treating as bright_on_dark."],
        stale_stages=["fit:lane=3"], lanes=states,
    )
    figures = [Figure(src="img/linear.png", caption="Gel", alt="Gel image", width=10, height=10)]
    content = ReportContent(
        run_name="run", gelpy_version="1", recipe=recipe, status=status,
        result=AnalysisResult(lanes=[ok, low, empty]),
        fit_config=FitConfigStore(lanes={0: PerLaneFitConfig()}),
        gel=figures, profiles=figures, fits=figures, fingerprint="0123456789abcdef", damaged=[3],
    )
    return build_report(content)["index.html"].decode("utf-8")


def _css_rules():
    """(selectors, variables used) of every rule of the default stylesheet but :root, which defines."""
    css = re.sub(r"/\*.*?\*/", "", DEFAULT_CSS, flags=re.S)
    rules = []
    for selectors, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
        names = [s.strip() for s in selectors.split(",")]
        if names != [":root"]:
            rules.append((names, set(re.findall(r"var\((--gelpy-[a-z-]+)\)", body))))
    return rules


def _matches(selector, elements):
    """True when a simple selector (``tag``, ``.class``, ``tag.class``) fits an element of the page."""
    match = re.fullmatch(r"([a-z0-9]*)((?:\.[a-z0-9-]+)*)", selector)
    assert match, f"not a simple selector: {selector!r}"
    tag, classes = match.group(1), set(re.findall(r"\.([a-z0-9-]+)", match.group(2)))
    return any(
        (not tag or tag == name) and classes <= set((attrs.get("class") or "").split())
        for name, attrs in elements
    )


def test_every_css_variable_is_used_by_a_rule_that_matches_an_element_of_a_page():
    elements = _elements(_rich_page())
    defined = set(re.findall(r"(--gelpy-[a-z-]+)\s*:", DEFAULT_CSS))
    effective = {
        variable
        for selectors, variables in _css_rules()
        if any(_matches(selector, elements) for selector in selectors)
        for variable in variables
    }
    assert defined - effective == set(), "defined, but no rule that uses it matches the page"


def test_no_rule_of_the_default_stylesheet_matches_nothing():
    elements = _elements(_rich_page())
    dead = [names for names, _ in _css_rules() if not any(_matches(s, elements) for s in names)]
    assert dead == []


# --- bidi and control characters are shown, not acted on ---------------------------------------
# Written as code points: a source file should not itself hold an invisible override.
#: Every bidirectional-formatting character there is: the marks, the embeddings and overrides, the isolates.
BIDI_CONTROLS = [0x061C, 0x200E, 0x200F, *range(0x202A, 0x202F), *range(0x2066, 0x206A)]
#: The C0 and C1 controls but tab and line feed, and DEL.
OTHER_CONTROLS = [*(c for c in range(0x20) if c not in (0x09, 0x0A)), 0x7F, *range(0x80, 0xA0)]
CONTROLS = [chr(c) for c in (*BIDI_CONTROLS, *OTHER_CONTROLS)]


# The figures draw these labels too, and matplotlib warns that its font has no glyph for them.
@pytest.mark.filterwarnings("ignore:Glyph:UserWarning")
@pytest.mark.filterwarnings("ignore:Matplotlib currently does not support Arabic:UserWarning")
def test_control_and_bidi_characters_are_shown_as_text_not_acted_on(gel_image, run_dir, tmp_path):
    assert len(BIDI_CONTROLS) == 12 and len(OTHER_CONTROLS) == 63
    rlo, lri = chr(0x202E), chr(0x2066)
    image = tmp_path / f"gel{rlo}name.tif"
    shutil.copyfile(gel_image, image)
    run = tmp_path / f"run{lri}folder"
    _run("init", str(run), str(image), "--n-lanes", "4", "--width", "30")
    half = len(BIDI_CONTROLS) // 2
    labels = [
        "a" + "".join(map(chr, BIDI_CONTROLS[:half])) + "b",
        "c" + "".join(map(chr, BIDI_CONTROLS[half:])) + "d",
        "e" + "".join(map(chr, OTHER_CONTROLS[:31])) + "f",
        "g" + "".join(map(chr, OTHER_CONTROLS[31:])) + "h",
    ]
    arguments = {"run_dir": str(run), "labels": labels}
    done = runner.invoke(app, ["tool", "call", "lanes_set"], input=json.dumps(arguments))
    assert done.exit_code == 0, done.output
    page = _page(str(run))
    assert [ch for ch in CONTROLS if ch in page] == []
    for code in (*BIDI_CONTROLS, *OTHER_CONTROLS):
        assert f"&lt;U+{code:04X}&gt;" in page, hex(code)


def test_every_element_that_carries_a_user_string_is_isolated_from_its_neighbours(fitted):
    for selector in ("h1", "td", "th", "dd", "figcaption", r"\.user"):
        assert re.search(rf"{selector}\b[^{{}}]*\{{[^{{}}]*unicode-bidi:\s*isolate", DEFAULT_CSS), selector
    page = _page(fitted)
    assert '<span class="user">' in page  # the image name in the meta line
    for wrapped in re.findall(r'<span class="user">(.*?)</span>', page):
        assert "<" not in wrapped  # what is inside is escaped text


# --- --css -------------------------------------------------------------------------------------------------
def test_a_stylesheet_is_copied_byte_for_byte(fitted, tmp_path):
    raw = b"\xef\xbb\xbf:root { --gelpy-bg: #fafafa; }\r\n/* caf\xc3\xa9 */\r\n"
    theme = tmp_path / "theme-source.css"
    theme.write_bytes(raw)
    _run("report", "html", fitted, "--css", str(theme))
    assert (Path(fitted) / "report" / "theme.css").read_bytes() == raw


def test_a_hostile_stylesheet_stays_out_of_the_page(fitted, tmp_path):
    text = (
        "</style><script>alert(1)</script>@import url(http://example.invalid/x.css);\n"
        "body { background: url(javascript:alert(1)) }\n"
    )
    theme = tmp_path / "hostile.css"
    theme.write_text(text, encoding="utf-8")
    _run("report", "html", fitted, "--css", str(theme))
    folder = Path(fitted) / "report"
    page = (folder / "index.html").read_text(encoding="utf-8")
    assert "<script" not in page and "example.invalid" not in page and "javascript" not in page
    assert [attrs["href"] for tag, attrs in _elements(page) if tag == "link"] == ["report.css", "theme.css"]
    assert (folder / "theme.css").read_text(encoding="utf-8") == text


def test_a_stylesheet_of_exactly_one_mebibyte_is_accepted_and_a_byte_more_is_refused(fitted, tmp_path):
    fits = tmp_path / "fits.css"
    fits.write_bytes(b"/*" + b" " * ((1 << 20) - 4) + b"*/")
    assert fits.stat().st_size == 1 << 20
    _run("report", "html", fitted, "--css", str(fits))
    big = tmp_path / "big.css"
    big.write_bytes(fits.read_bytes() + b" ")
    error = _fail("report", "html", fitted, "--css", str(big))
    assert error["code"] == "invalid_value" and "1 MiB" in error["error"] and "big.css" in error["error"]


def test_a_stylesheet_that_is_a_folder_is_refused_by_name(fitted, tmp_path):
    (tmp_path / "styles").mkdir()
    error = _fail("report", "html", fitted, "--css", str(tmp_path / "styles"))
    assert error["code"] == "invalid_value" and "styles" in error["error"] and error["detail"]
    assert not (Path(fitted) / "report").exists()


@posix_only
def test_a_stylesheet_that_is_a_pipe_is_refused_without_waiting_for_a_writer(fitted, tmp_path):
    pipe = tmp_path / "pipe.css"
    os.mkfifo(pipe)
    done = subprocess.run(
        [sys.executable, "-m", "gelpy", "report", "html", fitted, "--css", str(pipe)],
        capture_output=True, text=True, timeout=30,
    )
    assert done.returncode == 1 and json.loads(done.stderr)["code"] == "invalid_value"


# --- an absolute index outside the run ----------------------------------------------------------------
def test_the_index_of_a_report_outside_the_run_is_absolute_from_any_directory(
    fitted, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    built = _run("report", "html", fitted, "--out", "relative-out")
    assert Path(built["index"]).is_absolute()
    assert Path(built["index"]) == tmp_path / "relative-out" / "index.html"
    assert _run("report", "html", fitted)["index"] == "report/index.html"


# --- what the page says ---------------------------------------------------------------------------------
def test_the_page_says_what_its_numbers_are_relative_to_and_where_the_rest_is(fitted):
    page = _page(fitted)
    assert "<th scope=\"col\">position from slice start (px)</th>" in page
    assert "page 1 of 1" in page and "page 0 of" not in page
    assert "full configuration: fit_config.json" in page.lower()


def test_a_low_or_an_empty_lane_says_why_and_a_hostile_cache_text_is_escaped(fitted):
    low, empty = _cached_fit(fitted, 0), _cached_fit(fitted, 1)
    data = json.loads(low.read_text(encoding="utf-8"))
    data.update(
        confidence="low", coverage=0.5,
        confidence_reasons=["band 1 overlaps a neighbor <b>(areas correlated)</b>"],
    )
    low.write_text(json.dumps(data), encoding="utf-8")
    data = json.loads(empty.read_text(encoding="utf-8"))
    data.update(
        status="empty", bands=[], error="<script>alert(1)</script>", confidence="low",
        confidence_reasons=["no significant signal detected"],
    )
    empty.write_text(json.dumps(data), encoding="utf-8")
    page = _page(fitted)
    assert "coverage 0.500" in page
    assert "band 1 overlaps a neighbor &lt;b&gt;(areas correlated)&lt;/b&gt;" in page
    assert "no significant signal detected" in page
    assert "<script>" not in page and "&lt;script&gt;alert(1)&lt;/script&gt;" in page


def test_the_warnings_of_the_run_are_on_the_page(gel_image, run_dir):
    _run("init", run_dir, gel_image, "--n-lanes", "4", "--width", "30")
    recipe_file = Path(run_dir) / "recipe.json"
    recipe = json.loads(recipe_file.read_text(encoding="utf-8"))
    recipe["image"].update(polarity_confidence="low", polarity_reason="skew <i>unclear</i>")
    recipe_file.write_text(json.dumps(recipe), encoding="utf-8")
    page = _page(run_dir)
    assert "Warnings:" in page and "image polarity is low-confidence" in page
    assert "skew &lt;i&gt;unclear&lt;/i&gt;" in page


def test_stale_stages_are_named_in_words(fitted):
    _run("lanes", "set", fitted, "--width", "20")
    note = re.search(r'<p class="note">Stale cached stages.*?</p>', _page(fitted)).group(0)
    assert "lane 0: fit" in note and "lane 1: fit" in note and "fit:lane=" not in note


# --- one bad lane or one odd label does not stop the report ------------------------------------
def test_a_damaged_fit_file_is_listed_as_damaged_and_the_report_still_builds(fitted):
    _cached_fit(fitted, 0).write_text('{"lane_index": 0, "lab', encoding="utf-8")
    page = _page(fitted)
    assert page.count("<td>damaged</td>") == 1 and "<td>current</td>" in page
    assert "Damaged fit file" in page and "lane 0" in page


def test_labels_that_look_like_mathtext_do_not_stop_any_picture(gel_image, run_dir):
    _run("init", run_dir, gel_image, "--n-lanes", "4", "--width", "30")
    arguments = {"run_dir": run_dir, "labels": [r"$\frac$", r"$x$ and $y$", "a$", "$$"]}
    done = runner.invoke(app, ["tool", "call", "lanes_set"], input=json.dumps(arguments))
    assert done.exit_code == 0, done.output
    _run("background", "remove", run_dir, "--model", "plane_ransac")
    _run("image", "show", run_dir, "--kind", "linear", "--lanes")
    _run("profiles", "extract", run_dir, "--plot")
    _run("fit", "run", run_dir, "--lane", "0", "--lane", "1", "--plot")
    page = _page(run_dir)
    assert "$\\frac$" in page


# --- determinism across processes ------------------------------------------------------------------------
def _build_in_a_process(run, out, hash_seed, time_zone):
    env = {**os.environ, "PYTHONHASHSEED": hash_seed, "TZ": time_zone}
    done = subprocess.run(
        [sys.executable, "-m", "gelpy", "report", "html", run, "--out", str(out)],
        env=env, capture_output=True, text=True, timeout=300,
    )
    assert done.returncode == 0, done.stderr


def test_two_builds_in_other_processes_give_the_same_bytes(fitted, tmp_path):
    _build_in_a_process(fitted, tmp_path / "a", "1", "Asia/Tokyo")
    _build_in_a_process(fitted, tmp_path / "b", "999", "America/Los_Angeles")
    assert _tree(tmp_path / "a") == _tree(tmp_path / "b")


# --- markup --------------------------------------------------------------------------------------------------
def test_the_markup_helps_readers_and_screen_readers(fitted):
    page = _page(fitted)
    elements = _elements(page)
    assert all(attrs.get("scope") == "col" for tag, attrs in elements if tag == "th")
    images = [attrs for tag, attrs in elements if tag == "img"]
    assert images and all(
        attrs.get("loading") == "lazy" and attrs["width"].isdigit() and attrs["height"].isdigit()
        for attrs in images
    )
    captions = re.findall(r"<figcaption>(.*?)</figcaption>", page)
    assert len(captions) == len(images)
    assert all(attrs["alt"] and attrs["alt"] != caption for attrs, caption in zip(images, captions))
    assert 'class=""' not in page and 'class="confidence-"' not in page
    assert re.search(r"th, td[^{}]*\{[^{}]*overflow-wrap:\s*anywhere", DEFAULT_CSS)


# === a marker is a plain file anyone can write, so it never widens what a build touches ===
RUN_FILES = ["recipe.json", "fit_config.json", "manifest.json"]
DEBRIS = ".index.html.0123456789ab.tmp"  # what a write killed in the middle of index.html leaves


def _forged(*files):
    return json.dumps({"gelpy_version": "1", "fingerprint": "x", "files": list(files)})


@pytest.fixture
def started(gel_image, run_dir):
    """A run that has been initialised and nothing more: the cheapest run a report can refuse."""
    _run("init", run_dir, gel_image, "--n-lanes", "4", "--width", "30")
    return Path(run_dir)


@pytest.mark.parametrize("listed", [RUN_FILES, ["index.html", "report.css", "img/linear.png"]])
def test_a_marker_planted_in_the_run_folder_does_not_make_it_a_target(started, listed):
    """The marker may name the run's own files, or only names a build writes: either way the run
    folder is no target, and nothing in it is deleted or written."""
    (started / MARKER).write_text(_forged(*listed), encoding="utf-8")
    before = _tree(started)
    error = _fail("report", "html", str(started), "--out", str(started))
    assert error["code"] == "invalid_value" and error["detail"]
    assert _tree(started) == before


@posix_only
@pytest.mark.parametrize("listed", [RUN_FILES, ["index.html", "report.css"]])
def test_a_report_link_to_the_run_folder_with_a_planted_marker_deletes_nothing(started, listed):
    """The hostile run folder: `report -> .` and a marker in the run root, and the
    default command with no flag at all."""
    (started / MARKER).write_text(_forged(*listed), encoding="utf-8")
    (started / "report").symlink_to(".")
    before = _tree(started)
    error = _fail("report", "html", str(started))
    assert error["code"] == "invalid_value" and "link" in error["error"] and "Delete" in error["detail"]
    assert _tree(started) == before and (started / "report").is_symlink()


@posix_only
def test_a_report_link_to_a_folder_outside_the_run_is_refused_and_nothing_is_written_there(started, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (started / "report").symlink_to(outside, target_is_directory=True)
    error = _fail("report", "html", str(started))
    assert error["code"] == "invalid_value" and "link" in error["error"] and "Delete" in error["detail"]
    assert list(outside.iterdir()) == [] and (started / "report").is_symlink()


@posix_only
def test_a_link_the_person_passes_as_out_is_followed_to_a_folder_outside_the_run(fitted, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "link").symlink_to(outside, target_is_directory=True)
    _run("report", "html", fitted, "--out", str(tmp_path / "link"))
    assert (outside / "index.html").is_file() and (outside / MARKER).is_file()


def test_the_run_folder_spelled_another_way_is_still_the_run_folder(started):
    (started / MARKER).write_text(_forged("index.html"), encoding="utf-8")
    before = _tree(started)
    spelled = started / "cache" / ".."
    assert _fail("report", "html", str(started), "--out", str(spelled))["code"] == "invalid_value"
    assert _tree(started) == before


def test_a_marker_that_lists_a_file_a_build_does_not_write_is_no_marker(started, tmp_path):
    folder = tmp_path / "mine"
    folder.mkdir()
    (folder / "index.html").write_text("mine", encoding="utf-8")
    (folder / MARKER).write_text(_forged("index.html", "recipe.json"), encoding="utf-8")
    before = _tree(folder)
    error = _fail("report", "html", str(started), "--out", str(folder))
    assert error["code"] == "invalid_value" and "empty or new folder" in error["detail"]
    assert _tree(folder) == before


@pytest.mark.parametrize("where", ["", "cache", "render", "log", "parent"])
def test_a_planted_marker_does_not_make_the_run_folder_or_one_around_or_inside_it_a_target(started, where):
    target = {"": started, "parent": started.parent}.get(where, started / where)
    target.mkdir(exist_ok=True)
    (target / MARKER).write_text(_forged("index.html", "img/linear.png"), encoding="utf-8")
    before = _tree(started.parent)
    error = _fail("report", "html", str(started), "--out", str(target))
    assert error["code"] == "invalid_value" and error["detail"]
    assert _tree(started.parent) == before


@pytest.mark.parametrize("name", ["recipe.json", "manifest.json"])
def test_a_folder_with_a_run_file_in_it_is_no_target_whatever_marker_it_holds(started, tmp_path, name):
    other = tmp_path / "other"
    other.mkdir()
    (other / name).write_text("precious", encoding="utf-8")
    (other / MARKER).write_text(_forged("index.html"), encoding="utf-8")
    before = _tree(other)
    error = _fail("report", "html", str(started), "--out", str(other))
    assert error["code"] == "invalid_value" and name in error["error"]
    assert _tree(other) == before


def test_the_runs_own_report_folder_may_still_be_named_with_out(fitted):
    _run("report", "html", fitted)
    built = _run("report", "html", fitted, "--out", str(Path(fitted) / "report"))
    assert built["index"] == "report/index.html"


# --- the fingerprint covers the stored config of every lane ----------------------------------------
def test_status_sees_the_stored_config_of_a_lane_that_has_no_current_fit(fitted):
    _run("lanes", "set", fitted, "--width", "20")  # both fits are out of date: no lane has a current fit
    _run("report", "html", fitted)
    assert _run("status", fitted)["report"]["current"] is True
    config_file = Path(fitted) / "fit_config.json"
    config = json.loads(config_file.read_text(encoding="utf-8"))
    config["lanes"]["0"]["fit"]["seed"] += 1  # the page lists the configuration of every lane in the store
    config_file.write_text(json.dumps(config), encoding="utf-8")
    assert _run("status", fitted)["report"]["current"] is False
    _run("report", "html", fitted)
    assert _run("status", fitted)["report"]["current"] is True


# --- what a killed write leaves in a report folder ------------------------------------------------------
def _litter(folder):
    """Leftovers of killed writes in the three places a build writes, and things that only look like them."""
    leftovers = [DEBRIS, f".{MARKER}.0123456789ab.tmp", "img/.linear.png.0123456789ab.tmp",
                 "img/fits/.lane_000.png.0123456789ab.tmp"]
    decoys = {
        "notes.txt": b"mine",
        ".notes.tmp": b"mine",
        ".report.css.0123456789AB.tmp": b"mine",
        "index.html.0123456789ab.tmp": b"mine",
        "img/sub/.x.png.0123456789ab.tmp": b"mine",
    }
    for name in leftovers:
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(b"half")
    for name, data in decoys.items():
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(data)
    return leftovers, decoys


def test_the_next_build_removes_the_leftovers_of_a_killed_one(fitted):
    _run("report", "html", fitted)
    folder = Path(fitted) / "report"
    leftovers, decoys = _litter(folder)
    built = _run("report", "html", fitted)
    for name in leftovers:
        assert not (folder / name).exists(), name
    for name, data in decoys.items():
        assert (folder / name).read_bytes() == data, name
    assert set(_tree(folder)) == {*built["files"], MARKER, *decoys}


def test_a_folder_that_holds_only_the_leftovers_of_a_killed_first_build_is_a_target(fitted, tmp_path):
    folder = tmp_path / "out"
    folder.mkdir()
    (folder / f".{MARKER}.0123456789ab.tmp").write_bytes(b"{")  # killed while the first marker was written
    built = _run("report", "html", fitted, "--out", str(folder))
    assert set(_tree(folder)) == {*built["files"], MARKER}


# --- a build never writes over a file the marker does not list ---------------------------------------
@pytest.mark.parametrize(
    "listed",
    [[], ["index.html"], ["report.css", "img/linear.png"]],
    ids=["nothing", "the page", "a stylesheet and an image"],
)
def test_a_marker_that_lists_only_some_of_the_files_does_not_let_a_build_overwrite_the_others(fitted, listed):
    """"A rebuild replaces only the files its marker lists": a marker anyone can write lists as
    little as it likes, and a file it does not list is not the report's."""
    folder = Path(fitted) / "report"
    names = ["index.html", "report.css", "img/linear.png"]
    for name in names:
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_text("mine", encoding="utf-8")
    (folder / MARKER).write_text(
        json.dumps({"gelpy_version": "1", "fingerprint": "", "files": listed}), encoding="utf-8"
    )
    before = _tree(folder)

    error = _fail("report", "html", fitted)

    assert error["code"] == "invalid_value" and error["detail"]
    unlisted = [name for name in names if name not in listed]
    assert all(name in error["error"] for name in unlisted) and MARKER in error["error"]
    assert not any(name in error["error"] for name in listed)
    assert _tree(folder) == before  # nothing was written, and nothing was removed


def test_a_file_the_marker_lists_is_replaced_as_before(fitted):
    built = _run("report", "html", fitted)
    folder = Path(fitted) / "report"
    (folder / "index.html").write_text("edited", encoding="utf-8")
    rebuilt = _run("report", "html", fitted)
    assert rebuilt["files"] == built["files"]
    assert (folder / "index.html").read_text(encoding="utf-8") != "edited"


# --- --out names the run's own report folder: the same link rule as the default --------------------
@posix_only
def test_out_naming_a_report_folder_that_is_a_link_is_refused_like_the_default(fitted, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (Path(fitted) / "report").symlink_to(outside, target_is_directory=True)
    for args in ((), ("--out", str(Path(fitted) / "report"))):
        error = _fail("report", "html", fitted, *args)
        assert error["code"] == "invalid_value" and "link" in error["error"] and error["detail"]
    assert list(outside.iterdir()) == []
    status = json.loads(runner.invoke(app, ["status", fitted]).stdout)
    assert status["report"] is None  # a folder outside the run is not the run's report


# --- an empty folder is empty although Finder has been in it ------------------------------------------
def test_a_folder_that_holds_only_a_ds_store_is_empty(fitted, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / ".DS_Store").write_bytes(b"finder")
    built = _run("report", "html", fitted, "--out", str(out))
    assert (out / "index.html").is_file() and "index.html" in built["files"]
    assert (out / ".DS_Store").read_bytes() == b"finder"  # left alone, and not listed
    assert ".DS_Store" not in _marker(out)["files"]
    _run("report", "html", fitted, "--out", str(out))  # and a rebuild is a rebuild


def test_a_ds_store_next_to_other_files_is_no_excuse(fitted, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / ".DS_Store").write_bytes(b"finder")
    (out / "notes.txt").write_text("mine", encoding="utf-8")
    assert _fail("report", "html", fitted, "--out", str(out))["code"] == "invalid_value"


# --- a marker with a name too long for a file is no marker ---------------------------------------------
def test_a_marker_that_lists_a_name_too_long_for_a_file_is_no_marker_and_no_os_error(fitted):
    folder = Path(fitted) / "report"
    (folder / "img").mkdir(parents=True)
    (folder / "index.html").write_text("mine", encoding="utf-8")
    listed = {"gelpy_version": "1", "fingerprint": "", "files": ["index.html", "img/" + "x" * 252 + ".png"]}
    (folder / MARKER).write_text(json.dumps(listed), encoding="utf-8")
    before = _tree(folder)
    error = _fail("report", "html", fitted)
    assert error["code"] == "invalid_value"  # not "os_error: File name too long"
    assert _tree(folder) == before


# --- leftovers that cannot be removed do not abort a finished build -------------------------------------
def test_a_leftover_that_cannot_be_removed_does_not_abort_a_finished_build(fitted, monkeypatch):
    import errno

    _run("report", "html", fitted)
    folder = Path(fitted) / "report"
    stuck = folder / ".index.html.0123456789ab.tmp"
    stuck.write_bytes(b"half")
    real = os.unlink

    def immovable(path, *args, **kwargs):
        if os.path.basename(str(path)) == stuck.name:
            raise PermissionError(errno.EPERM, "Operation not permitted", str(path))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", immovable)
    for _ in range(2):  # the first build and every later one
        built = _run("report", "html", fitted)
        assert "index.html" in built["files"]
        assert stuck.exists()
        status = json.loads(runner.invoke(app, ["status", fitted]).stdout)
        assert status["report"]["current"] is True  # the build finished: its marker is the final one
