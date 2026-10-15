"""init refuses an existing run unless forced, and a refused image leaves nothing behind."""

import errno
import json
import os
import shutil
import sys

import pytest

from gelpy.core.errors import ImageDecodeError, RunExistsError
from gelpy.runfolder import commands
from gelpy.runfolder.layout import RunPaths
from gelpy.schemas.requests import (
    FitRunRequest,
    InitRunRequest,
    RemoveBackgroundRequest,
    SetLanesRequest,
)


def _init(run_dir, image_path, n_lanes=4, force=False):
    return commands.init_run(
        InitRunRequest(run_dir=run_dir, image_path=image_path, n_lanes=n_lanes, width=30, force=force)
    )


def test_init_refuses_an_existing_run(gel_image_path, run_dir):
    _init(run_dir, gel_image_path[0])
    with pytest.raises(RunExistsError):
        _init(run_dir, gel_image_path[0])


def test_force_replaces_the_run_and_removes_what_it_produced(gel_image_path, run_dir):
    image_path, centers, _, _ = gel_image_path
    _init(run_dir, image_path)
    commands.set_lanes(SetLanesRequest(run_dir=run_dir, x_centers=[float(c) for c in centers]))
    commands.remove_background(RemoveBackgroundRequest(run_dir=run_dir, model="plane_ransac"))
    commands.fit_run(FitRunRequest(run_dir=run_dir, lanes=[0]))
    paths = RunPaths(run_dir)
    (paths.render / "linear.png").write_bytes(b"png")
    assert paths.results_csv.exists()

    _init(run_dir, image_path, n_lanes=3, force=True)

    recipe = json.loads(paths.recipe.read_text(encoding="utf-8"))
    assert len(recipe["lanes"]["lanes"]) == 3
    assert json.loads(paths.fit_config.read_text(encoding="utf-8"))["lanes"] == {}
    assert not paths.results_csv.exists() and not paths.results_json.exists()
    assert not (paths.render / "linear.png").exists()
    assert any(paths.cache.rglob("*.json"))  # the content-addressed cache is kept
    commands_logged = [
        json.loads(line)["command"]
        for line in paths.history.read_text(encoding="utf-8").splitlines()
    ]
    assert commands_logged.count("init") == 2  # the history is kept


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need extra privileges on Windows")
def test_force_unlinks_a_render_symlink_and_never_follows_it(gel_image_path, run_dir, tmp_path):
    image_path = gel_image_path[0]
    _init(run_dir, image_path)
    paths = RunPaths(run_dir)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "keep.png").write_bytes(b"png")
    shutil.rmtree(paths.render)
    paths.render.symlink_to(elsewhere, target_is_directory=True)

    _init(run_dir, image_path, force=True)

    assert (elsewhere / "keep.png").read_bytes() == b"png"  # what the link pointed at is untouched
    assert not (elsewhere / "fits").exists()  # and nothing was created through the link
    assert paths.render.is_dir() and not paths.render.is_symlink()  # render/ is a real folder again


def test_a_render_folder_that_cannot_be_removed_is_an_error(gel_image_path, run_dir, monkeypatch):
    """rmtree(ignore_errors=True) reported success while the old renders stayed reachable. A
    renders folder that cannot be removed now stops init --force, and the run stays as it was."""
    image_path = gel_image_path[0]
    _init(run_dir, image_path)

    def refusing_rmtree(path, ignore_errors=False, *args, **kwargs):
        if not ignore_errors:  # what shutil.rmtree does: it raises unless asked to ignore errors
            raise PermissionError(errno.EACCES, "Permission denied", str(path))

    monkeypatch.setattr(shutil, "rmtree", refusing_rmtree)
    with pytest.raises(PermissionError):
        _init(run_dir, image_path, n_lanes=3, force=True)
    recipe = json.loads(RunPaths(run_dir).recipe.read_text(encoding="utf-8"))
    assert len(recipe["lanes"]["lanes"]) == 4  # still the old run


MARKER = ".gelpy-report.json"
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="symlinks need extra privileges on Windows")


def _marked_report(paths, files, fingerprint="0123456789abcdef"):
    """A report folder as report html leaves it: the listed files and the marker."""
    folder = paths.html_report
    for name in files:
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_text(name, encoding="utf-8")
    listed = {"gelpy_version": "1.1.0", "fingerprint": fingerprint, "files": list(files)}
    (folder / MARKER).write_text(json.dumps(listed), encoding="utf-8")
    return folder


def _lines(summary):
    return "\n".join(summary.warnings)


@posix_only
def test_force_unlinks_a_report_symlink_and_never_follows_it(gel_image_path, run_dir, tmp_path):
    """A link as report/ is unlinked, like a link as render/; what it points to is not touched."""
    image_path = gel_image_path[0]
    _init(run_dir, image_path)
    paths = RunPaths(run_dir)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "keep.html").write_bytes(b"page")
    paths.html_report.symlink_to(elsewhere, target_is_directory=True)

    summary = _init(run_dir, image_path, force=True)

    assert (elsewhere / "keep.html").read_bytes() == b"page"  # what the link pointed at is untouched
    assert not paths.html_report.exists() and not paths.html_report.is_symlink()
    assert "report" in _lines(summary) and "link" in _lines(summary)


def test_force_removes_the_files_the_marker_lists_then_the_empty_folders_then_the_marker(
    gel_image_path, run_dir
):
    image_path = gel_image_path[0]
    _init(run_dir, image_path)
    paths = RunPaths(run_dir)
    files = ["index.html", "report.css", "img/linear.png", "img/fits/lane_000.png"]
    _marked_report(paths, files)

    summary = _init(run_dir, image_path, force=True)

    assert not paths.html_report.exists()
    assert "Removed" in _lines(summary) and "index.html" in _lines(summary)
    assert "4 files" in _lines(summary)


def test_force_leaves_what_the_marker_does_not_list_and_says_so(gel_image_path, run_dir):
    image_path = gel_image_path[0]
    _init(run_dir, image_path)
    paths = RunPaths(run_dir)
    folder = _marked_report(paths, ["index.html", "img/linear.png"])
    (folder / "notes.txt").write_text("mine", encoding="utf-8")
    (folder / "img" / "mine.png").write_bytes(b"mine")

    summary = _init(run_dir, image_path, force=True)

    assert (folder / "notes.txt").read_text(encoding="utf-8") == "mine"
    assert (folder / "img" / "mine.png").read_bytes() == b"mine"
    for name in ("index.html", "img/linear.png", MARKER):
        assert not (folder / name).exists(), name
    assert "Removed" in _lines(summary)
    assert "Left" in _lines(summary) and "notes.txt" in _lines(summary) and "img/mine.png" in _lines(summary)


def test_force_leaves_a_report_folder_without_a_marker_alone(gel_image_path, run_dir):
    """A 1.0 run folder's report/ is the user's: gelpy never wrote there."""
    image_path = gel_image_path[0]
    _init(run_dir, image_path)
    paths = RunPaths(run_dir)
    (paths.html_report / "figures").mkdir(parents=True)
    (paths.html_report / "figures" / "fig1.svg").write_text("<svg/>", encoding="utf-8")
    (paths.html_report / "notes.txt").write_text("mine", encoding="utf-8")

    summary = _init(run_dir, image_path, force=True)

    assert (paths.html_report / "notes.txt").read_text(encoding="utf-8") == "mine"
    assert (paths.html_report / "figures" / "fig1.svg").read_text(encoding="utf-8") == "<svg/>"
    assert "Left" in _lines(summary) and MARKER in _lines(summary)


@pytest.mark.parametrize("name", ["../recipe.json", "img/../../recipe.json", "a\\b", "img//x.png"])
def test_force_does_not_trust_a_marker_that_names_a_path_it_should_not(gel_image_path, run_dir, name):
    image_path = gel_image_path[0]
    _init(run_dir, image_path)
    paths = RunPaths(run_dir)
    folder = _marked_report(paths, ["index.html"])
    listed = json.loads((folder / MARKER).read_text(encoding="utf-8"))
    listed["files"].append(name)
    (folder / MARKER).write_text(json.dumps(listed), encoding="utf-8")

    summary = _init(run_dir, image_path, force=True)

    assert (folder / "index.html").exists()  # an unreliable marker is no marker: nothing of it is removed
    assert "Left" in _lines(summary)
    recipe = json.loads(paths.recipe.read_text(encoding="utf-8"))
    assert len(recipe["lanes"]["lanes"]) == 4  # the init itself went through


@posix_only
def test_force_unlinks_a_listed_file_that_is_a_link_and_never_follows_it(gel_image_path, run_dir, tmp_path):
    image_path = gel_image_path[0]
    _init(run_dir, image_path)
    paths = RunPaths(run_dir)
    victim = tmp_path / "victim.txt"
    victim.write_text("precious", encoding="utf-8")
    folder = _marked_report(paths, ["index.html", "img/linked.png"])
    (folder / "img" / "linked.png").unlink()
    (folder / "img" / "linked.png").symlink_to(victim)

    _init(run_dir, image_path, force=True)

    assert victim.read_text(encoding="utf-8") == "precious"
    assert not (folder / "img" / "linked.png").exists() and not (folder / "img" / "linked.png").is_symlink()


@posix_only
def test_force_never_follows_a_link_that_stands_where_the_marker_lists_a_folder(
    gel_image_path, run_dir, tmp_path
):
    image_path = gel_image_path[0]
    _init(run_dir, image_path)
    paths = RunPaths(run_dir)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "linear.png").write_text("precious", encoding="utf-8")
    folder = _marked_report(paths, ["index.html", "img/linear.png"])
    shutil.rmtree(folder / "img")
    (folder / "img").symlink_to(elsewhere, target_is_directory=True)

    summary = _init(run_dir, image_path, force=True)

    assert (elsewhere / "linear.png").read_text(encoding="utf-8") == "precious"
    assert (folder / "img").is_symlink() and "img" in _lines(summary)
    assert not (folder / "index.html").exists()


DEBRIS = ".index.html.0123456789ab.tmp"  # what a write killed in the middle of index.html leaves


def test_force_removes_the_leftovers_of_a_killed_build_with_the_listed_files(gel_image_path, run_dir):
    image_path = gel_image_path[0]
    _init(run_dir, image_path)
    paths = RunPaths(run_dir)
    folder = _marked_report(paths, ["index.html", "img/linear.png"])
    for name in (DEBRIS, "img/.linear.png.0123456789ab.tmp", f".{MARKER}.0123456789ab.tmp"):
        (folder / name).write_bytes(b"half")
    (folder / "notes.txt").write_text("mine", encoding="utf-8")

    summary = _init(run_dir, image_path, force=True)

    assert sorted(p.name for p in folder.iterdir()) == ["notes.txt"]  # img/ held nothing else
    assert "Removed" in _lines(summary) and DEBRIS in _lines(summary)


def test_force_removes_a_report_folder_that_holds_only_leftovers(gel_image_path, run_dir):
    """A first build killed while its marker was written: gelpy's temp file is all there is."""
    image_path = gel_image_path[0]
    _init(run_dir, image_path)
    paths = RunPaths(run_dir)
    paths.html_report.mkdir()
    (paths.html_report / f".{MARKER}.0123456789ab.tmp").write_bytes(b"{")

    summary = _init(run_dir, image_path, force=True)

    assert not paths.html_report.exists()
    assert "Removed" in _lines(summary)


def test_force_leaves_the_leftovers_next_to_the_users_files_when_there_is_no_marker(gel_image_path, run_dir):
    image_path = gel_image_path[0]
    _init(run_dir, image_path)
    paths = RunPaths(run_dir)
    paths.html_report.mkdir()
    (paths.html_report / DEBRIS).write_bytes(b"half")
    (paths.html_report / "notes.txt").write_text("mine", encoding="utf-8")

    summary = _init(run_dir, image_path, force=True)

    assert sorted(p.name for p in paths.html_report.iterdir()) == sorted([DEBRIS, "notes.txt"])
    assert "Left" in _lines(summary)


def test_a_listed_file_that_cannot_be_removed_is_an_error(gel_image_path, run_dir, monkeypatch):
    """Like the renders: a report that stayed reachable must not be reported as removed."""
    image_path = gel_image_path[0]
    _init(run_dir, image_path)
    paths = RunPaths(run_dir)
    _marked_report(paths, ["index.html", "img/linear.png"])
    shutil.rmtree(paths.render)  # the report is all that is left to remove
    real = os.unlink

    def refusing(path, *args, **kwargs):
        if str(path).startswith(str(paths.html_report) + os.sep):
            raise PermissionError(errno.EACCES, "Permission denied", str(path))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", refusing)
    with pytest.raises(PermissionError):
        _init(run_dir, image_path, n_lanes=3, force=True)
    recipe = json.loads(paths.recipe.read_text(encoding="utf-8"))
    assert len(recipe["lanes"]["lanes"]) == 4  # still the old run


def test_a_folder_without_a_recipe_is_not_a_run(gel_image_path, run_dir):
    """A crashed 1.0 init left cache/, log/ and render/ but no recipe.json: init takes the
    folder without --force, because only a recipe.json makes a run."""
    paths = RunPaths(run_dir)
    paths.ensure_dirs()
    assert not paths.exists()
    _init(run_dir, gel_image_path[0])
    assert paths.exists()


def test_a_refused_image_leaves_no_run_folder_behind(tmp_path):
    text = tmp_path / "notes.tif"
    text.write_text("not an image", encoding="utf-8")
    run = tmp_path / "run"
    with pytest.raises(ImageDecodeError):
        _init(str(run), str(text), n_lanes=3)
    assert not run.exists()


def test_force_leaves_a_report_whose_marker_lists_a_name_too_long_for_a_file(gel_image_path, run_dir):
    """Such a marker is no marker (a name of 256 bytes or more cannot exist): ``init --force`` used
    to fail on it with "File name too long", and so did every later one."""
    image_path = gel_image_path[0]
    _init(run_dir, image_path)
    paths = RunPaths(run_dir)
    folder = _marked_report(paths, ["index.html"])
    listed = json.loads((folder / MARKER).read_text(encoding="utf-8"))
    listed["files"].append("img/" + "x" * 252 + ".png")
    (folder / MARKER).write_text(json.dumps(listed), encoding="utf-8")

    summary = _init(run_dir, image_path, n_lanes=3, force=True)

    assert (folder / "index.html").exists()  # an unreliable marker is no marker: nothing of it is removed
    assert "Left" in _lines(summary)
    assert len(json.loads(paths.recipe.read_text(encoding="utf-8"))["lanes"]["lanes"]) == 3


def test_force_goes_through_with_a_leftover_that_cannot_be_removed(gel_image_path, run_dir, monkeypatch):
    image_path = gel_image_path[0]
    _init(run_dir, image_path)
    paths = RunPaths(run_dir)
    folder = _marked_report(paths, ["index.html", "img/linear.png"])
    (folder / DEBRIS).write_bytes(b"half")
    real = os.unlink

    def immovable(path, *args, **kwargs):
        if os.path.basename(str(path)) == DEBRIS:
            raise PermissionError(errno.EPERM, "Operation not permitted", str(path))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", immovable)

    summary = _init(run_dir, image_path, n_lanes=3, force=True)

    assert not (folder / "index.html").exists() and not (folder / MARKER).exists()
    assert (folder / DEBRIS).exists()  # still there, and said so
    assert "Left" in _lines(summary) and DEBRIS in _lines(summary)
    assert len(json.loads(paths.recipe.read_text(encoding="utf-8"))["lanes"]["lanes"]) == 3


def test_force_does_not_warn_about_a_report_folder_that_holds_only_a_ds_store(gel_image_path, run_dir):
    image_path = gel_image_path[0]
    _init(run_dir, image_path)
    paths = RunPaths(run_dir)
    paths.html_report.mkdir()
    (paths.html_report / ".DS_Store").write_bytes(b"finder")

    summary = _init(run_dir, image_path, force=True)

    assert (paths.html_report / ".DS_Store").read_bytes() == b"finder"
    assert "Left" not in _lines(summary)  # nothing the report owns, and nothing the person put there
