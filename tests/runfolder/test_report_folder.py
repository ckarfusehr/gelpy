"""Which folder a report may be written into, and what is removed from it.

The marker is a plain JSON file that anyone can write into a folder, so it is never enough on its
own: the run folder, the folders inside it (but the run's own ``report/``) and the folders above
it are no target whatever they hold, and neither is a folder that holds a ``recipe.json`` or a
``manifest.json``. A folder is the report's by its marker, and what a build removes is what that
marker lists and nothing else.
"""

import errno
import os
import sys

import pytest

from gelpy.core.errors import InvalidValueError
from gelpy.runfolder.layout import RunPaths
from gelpy.runfolder.report_folder import claim_folder, clear_report, sweep_debris, write_marker
from gelpy.schemas.report import REPORT_MARKER, ReportMarker

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="symlinks need extra privileges")

DEBRIS = ".index.html.0123456789ab.tmp"  # what a write killed in the middle of index.html leaves


def _marker(*files):
    return ReportMarker(gelpy_version="1", fingerprint="f", files=list(files))


def _plant(folder, *files):
    """The folder as ``report html`` leaves it: a valid marker, and the files it lists."""
    folder.mkdir(parents=True, exist_ok=True)
    for name in files:
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_text(name, encoding="utf-8")
    write_marker(folder, _marker(*files))
    return folder


def _files(folder):
    return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file() or p.is_symlink())


@pytest.fixture
def run(tmp_path):
    root = tmp_path / "run"
    for inner in ("cache", "render", "log"):
        (root / inner).mkdir(parents=True)
    (root / "recipe.json").write_text("{}", encoding="utf-8")
    return RunPaths(root)


# --- what makes a folder no target, marker or not ---------------------------------------------------
def test_the_run_folder_is_never_a_target_whatever_marker_it_holds(run):
    _plant(run.root, "index.html")
    with pytest.raises(InvalidValueError) as excinfo:
        claim_folder(run.root, run)
    assert "run folder" in str(excinfo.value) and excinfo.value.detail


@pytest.mark.parametrize("inner", ["cache", "render", "log", "report/sub", "elsewhere/new"])
def test_a_folder_inside_the_run_other_than_its_own_report_is_never_a_target(run, inner):
    target = run.root / inner
    if target.exists():
        _plant(target, "index.html")
    with pytest.raises(InvalidValueError) as excinfo:
        claim_folder(target, run)
    assert "inside the run folder" in str(excinfo.value) and excinfo.value.detail


def test_the_runs_own_report_folder_is_a_target(run):
    assert claim_folder(run.html_report, run) is None  # not there yet
    assert claim_folder(run.html_report, run, chosen=False) is None
    _plant(run.html_report, "index.html", "img/linear.png")
    for chosen in (True, False):
        assert claim_folder(run.html_report, run, chosen=chosen).files == ["index.html", "img/linear.png"]


def test_a_folder_above_the_run_is_never_a_target(run, tmp_path):
    _plant(tmp_path, "index.html")
    with pytest.raises(InvalidValueError) as excinfo:
        claim_folder(tmp_path, run)
    assert "holds the run folder" in str(excinfo.value) and excinfo.value.detail


@pytest.mark.parametrize("name", ["recipe.json", "manifest.json"])
def test_a_folder_that_holds_a_recipe_or_a_manifest_is_never_a_target(run, tmp_path, name):
    other = _plant(tmp_path / "other", "index.html")
    (other / name).write_text("precious", encoding="utf-8")
    with pytest.raises(InvalidValueError) as excinfo:
        claim_folder(other, run)
    assert name in str(excinfo.value) and excinfo.value.detail
    assert (other / name).read_text(encoding="utf-8") == "precious"


@posix_only
def test_a_link_to_the_run_folder_is_the_run_folder(run, tmp_path):
    (tmp_path / "link").symlink_to(run.root, target_is_directory=True)
    with pytest.raises(InvalidValueError):
        claim_folder(tmp_path / "link", run)


@posix_only
def test_a_link_to_a_folder_inside_the_run_is_that_folder(run, tmp_path):
    (tmp_path / "link").symlink_to(run.root / "cache", target_is_directory=True)
    with pytest.raises(InvalidValueError):
        claim_folder(tmp_path / "link", run)
    run.html_report.symlink_to(run.root / "cache", target_is_directory=True)
    with pytest.raises(InvalidValueError):  # `--out RUN/report`, where report is a link inside the run
        claim_folder(run.html_report, run, chosen=True)


@posix_only
def test_a_link_the_person_passes_to_a_folder_outside_the_run_is_followed(run, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "link").symlink_to(outside, target_is_directory=True)
    assert claim_folder(tmp_path / "link", run, chosen=True) is None
    _plant(outside, "index.html")
    assert claim_folder(tmp_path / "link", run, chosen=True).files == ["index.html"]


@posix_only
def test_the_default_report_folder_is_a_real_folder_or_absent(run, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    run.html_report.symlink_to(outside, target_is_directory=True)
    with pytest.raises(InvalidValueError) as excinfo:
        claim_folder(run.html_report, run, chosen=False)
    assert "link" in str(excinfo.value) and "Delete" in excinfo.value.detail
    assert list(outside.iterdir()) == []  # and nothing was written there


@posix_only
def test_a_default_report_link_to_the_run_folder_is_refused_with_a_planted_marker(run):
    _plant(run.root, "index.html")
    run.html_report.symlink_to(".")
    with pytest.raises(InvalidValueError):
        claim_folder(run.html_report, run, chosen=False)


def _tells_case_apart(folder):
    probe = folder / "CaseProbe"
    probe.write_text("x", encoding="utf-8")
    try:
        return not (folder / "caseprobe").exists()
    finally:
        probe.unlink()


@pytest.mark.parametrize("where", ["inside", "above"])
def test_folders_are_compared_by_what_they_are_not_by_the_case_they_are_typed_in(run, tmp_path, where):
    """On a file system that does not tell names apart by case (the default of macOS and Windows) a
    folder typed in another case is the same folder."""
    if _tells_case_apart(tmp_path):
        pytest.skip("this file system tells names apart by case")
    if where == "inside":
        target = run.root.parent / run.root.name.upper() / "cache"
    else:
        target = tmp_path.parent / tmp_path.name.upper()
    _plant(target, "index.html")
    with pytest.raises(InvalidValueError):
        claim_folder(target, run)


# --- leftovers of a killed write ---------------------------------------------------------------------
def test_a_folder_that_holds_only_the_leftovers_of_a_killed_write_is_free(run, tmp_path):
    folder = tmp_path / "out"
    folder.mkdir()
    (folder / DEBRIS).write_bytes(b"half")
    (folder / f".{REPORT_MARKER}.0123456789ab.tmp").write_bytes(b"{")
    assert claim_folder(folder, run) is None
    (folder / "notes.txt").write_text("mine", encoding="utf-8")
    with pytest.raises(InvalidValueError):
        claim_folder(folder, run)


@posix_only
def test_a_folder_or_a_link_named_like_a_leftover_is_content(run, tmp_path):
    folder = tmp_path / "out"
    (folder / DEBRIS).mkdir(parents=True)
    with pytest.raises(InvalidValueError):
        claim_folder(folder, run)
    other = tmp_path / "other"
    other.mkdir()
    (other / DEBRIS).symlink_to(tmp_path)
    with pytest.raises(InvalidValueError):
        claim_folder(other, run)


def _litter(folder):
    """Leftovers in the three places a build writes, and things that only look like them."""
    leftovers = [DEBRIS, f".{REPORT_MARKER}.0123456789ab.tmp", "img/.linear.png.0123456789ab.tmp",
                 "img/fits/.lane_000.png.0123456789ab.tmp"]
    for name in leftovers:
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(b"half")
    decoys = {
        "notes.txt": b"mine",
        ".notes.tmp": b"mine",
        ".report.css.0123456789AB.tmp": b"mine",  # upper case: not what a write makes
        ".index.html.0123456789a.tmp": b"mine",  # too short
        "index.html.0123456789ab.tmp": b"mine",  # not a dotfile
        "img/sub/.x.png.0123456789ab.tmp": b"mine",  # below the folders a build writes in
    }
    for name, data in decoys.items():
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(data)
    return sorted(leftovers), decoys


def test_sweeping_removes_the_leftovers_and_nothing_that_only_looks_like_one(tmp_path):
    folder = tmp_path / "report"
    leftovers, decoys = _litter(folder)
    (folder / "img" / "fits" / ".gone.png.0123456789ab.tmp").write_bytes(b"x")  # the last file of a folder
    removed = sweep_debris(folder)
    assert removed == sorted([*leftovers, "img/fits/.gone.png.0123456789ab.tmp"])
    assert _files(folder) == sorted(decoys)
    assert not (folder / "img" / "fits").exists()  # it held nothing else


@posix_only
def test_sweeping_unlinks_regular_files_only_and_never_enters_a_link(tmp_path):
    folder = tmp_path / "report"
    folder.mkdir()
    (folder / DEBRIS.replace("index.html", "dir")).mkdir()  # a folder named like a leftover
    victim = tmp_path / "victim.txt"
    victim.write_text("precious", encoding="utf-8")
    (folder / DEBRIS.replace("index.html", "link")).symlink_to(victim)  # and a link
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / ".x.png.0123456789ab.tmp").write_bytes(b"theirs")
    (folder / "img").symlink_to(elsewhere, target_is_directory=True)  # img/ is a link: not entered
    assert sweep_debris(folder) == []
    assert victim.read_text(encoding="utf-8") == "precious"
    assert (elsewhere / ".x.png.0123456789ab.tmp").read_bytes() == b"theirs"
    assert (folder / DEBRIS.replace("index.html", "link")).is_symlink()
    assert (folder / DEBRIS.replace("index.html", "dir")).is_dir()


# --- init --force removes them too ----------------------------------------------------------------------
def test_clearing_a_marked_folder_removes_the_leftovers_with_the_listed_files(tmp_path):
    folder = _plant(tmp_path / "report", "index.html", "img/linear.png", "img/fits/lane_000.png")
    leftovers, _ = _litter(folder)
    cleared = clear_report(folder)
    assert set(leftovers) <= set(cleared.removed)
    assert {"index.html", "img/linear.png", "img/fits/lane_000.png"} <= set(cleared.removed)
    assert "notes.txt" in cleared.left and not (folder / REPORT_MARKER).exists()
    assert not (folder / "img" / "fits").exists() and (folder / "img" / "sub").is_dir()


def test_clearing_a_folder_that_holds_only_leftovers_removes_it(tmp_path):
    folder = tmp_path / "report"
    folder.mkdir()
    (folder / DEBRIS).write_bytes(b"half")
    cleared = clear_report(folder)
    assert cleared.removed == [DEBRIS] and cleared.left == []
    assert not folder.exists()


def test_a_folder_without_a_marker_that_holds_anything_else_is_left_alone(tmp_path):
    folder = tmp_path / "report"
    folder.mkdir()
    (folder / DEBRIS).write_bytes(b"half")
    (folder / "notes.txt").write_text("mine", encoding="utf-8")
    assert clear_report(folder) is None
    assert _files(folder) == sorted([DEBRIS, "notes.txt"])


# --- a link typed as a path inside the run is never followed -------------------------------------------
@posix_only
def test_a_link_typed_as_a_path_inside_the_run_is_refused_whichever_way_the_folder_is_chosen(run, tmp_path):
    """The default command refuses a report/ that is a link, and so must ``--out RUN/report``: the
    folder it names lies inside the run, whatever the link points to, and ``status`` reads the run's
    own report/ as the run's report."""
    outside = tmp_path / "outside"
    outside.mkdir()
    run.html_report.symlink_to(outside, target_is_directory=True)
    for chosen in (True, False):
        with pytest.raises(InvalidValueError) as excinfo:
            claim_folder(run.html_report, run, chosen=chosen)
        assert "link" in str(excinfo.value) and "Delete" in excinfo.value.detail
    assert list(outside.iterdir()) == []


@posix_only
@pytest.mark.parametrize("spelling", ["dotted", "through a link to the run", "relative to the working directory"])
def test_a_link_inside_the_run_is_refused_however_its_path_is_spelled(run, tmp_path, monkeypatch, spelling):
    outside = tmp_path / "outside"
    outside.mkdir()
    (run.root / "keep").symlink_to(outside, target_is_directory=True)
    if spelling == "dotted":
        typed = run.root / "." / "x" / ".." / "keep"
        (run.root / "x").mkdir()
    elif spelling == "through a link to the run":
        (tmp_path / "alias").symlink_to(run.root, target_is_directory=True)
        typed = tmp_path / "alias" / "keep"
    else:
        monkeypatch.chdir(tmp_path)
        typed = "run/keep"
    with pytest.raises(InvalidValueError) as excinfo:
        claim_folder(typed, run, chosen=True)
    assert "link" in str(excinfo.value)
    assert list(outside.iterdir()) == []


@posix_only
def test_a_link_outside_the_run_that_the_person_passes_is_still_followed(run, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "linked-report").symlink_to(outside, target_is_directory=True)
    assert claim_folder(tmp_path / "linked-report", run, chosen=True) is None


# --- an empty folder is empty although Finder has been in it -------------------------------------------
def test_a_ds_store_does_not_make_a_folder_non_empty(run, tmp_path):
    folder = tmp_path / "out"
    folder.mkdir()
    (folder / ".DS_Store").write_bytes(b"finder")
    assert claim_folder(folder, run) is None
    (folder / DEBRIS).write_bytes(b"half")  # and it sits next to the leftovers of a killed write, too
    assert claim_folder(folder, run) is None
    (folder / "notes.txt").write_text("mine", encoding="utf-8")
    with pytest.raises(InvalidValueError):
        claim_folder(folder, run)


@posix_only
def test_a_folder_or_a_link_named_ds_store_is_content(run, tmp_path):
    folder = tmp_path / "out"
    (folder / ".DS_Store").mkdir(parents=True)
    with pytest.raises(InvalidValueError):
        claim_folder(folder, run)
    other = tmp_path / "other"
    other.mkdir()
    (other / ".DS_Store").symlink_to(tmp_path)
    with pytest.raises(InvalidValueError):
        claim_folder(other, run)


def test_a_ds_store_is_not_removed_with_the_report(tmp_path):
    folder = _plant(tmp_path / "report", "index.html")
    (folder / ".DS_Store").write_bytes(b"finder")
    cleared = clear_report(folder)
    assert cleared.removed == ["index.html"] and cleared.left == [".DS_Store"]
    assert (folder / ".DS_Store").read_bytes() == b"finder"


# --- the files a marker does not list are never written over -----------------------------------------
def test_a_name_the_marker_does_not_list_must_not_exist(tmp_path):
    from gelpy.runfolder.report_folder import check_overwrites

    folder = _plant(tmp_path / "report", "index.html", "img/linear.png")
    earlier = _marker("index.html", "img/linear.png")
    check_overwrites(folder, ["index.html", "report.css", "img/linear.png", "img/fits/lane_000.png"], earlier)
    (folder / "report.css").write_text("mine", encoding="utf-8")
    with pytest.raises(InvalidValueError) as excinfo:
        check_overwrites(folder, ["index.html", "report.css", "img/linear.png"], earlier)
    assert "report.css" in str(excinfo.value) and REPORT_MARKER in str(excinfo.value) and excinfo.value.detail
    assert (folder / "report.css").read_text(encoding="utf-8") == "mine"


def test_a_marker_that_lists_nothing_lists_nothing(tmp_path):
    from gelpy.runfolder.report_folder import check_overwrites

    folder = tmp_path / "report"
    folder.mkdir()
    (folder / "index.html").write_text("mine", encoding="utf-8")
    with pytest.raises(InvalidValueError):
        check_overwrites(folder, ["index.html"], _marker())
    check_overwrites(folder, ["report.css"], _marker())  # a name that does not exist is the build's to write


def test_a_listed_name_may_exist_or_be_gone(tmp_path):
    from gelpy.runfolder.report_folder import check_overwrites

    folder = _plant(tmp_path / "report", "index.html")
    check_overwrites(folder, ["index.html", "img/gone.png"], _marker("index.html", "img/gone.png"))


def test_an_unlisted_name_that_is_a_listed_file_under_another_spelling_is_listed(tmp_path):
    """A file system that does not tell names apart by case has one file under ``img/LINEAR.png``
    and ``img/linear.png``: the marker lists it, so the build may write it."""
    from gelpy.runfolder.report_folder import check_overwrites

    folder = _plant(tmp_path / "report", "img/linear.png")
    try:
        os.link(folder / "img" / "linear.png", folder / "img" / "alias.png")  # one file, two names
    except (OSError, NotImplementedError):
        pytest.skip("this file system has no hard links")
    check_overwrites(folder, ["img/alias.png"], _marker("img/linear.png"))


# --- pruning compares files, not names ---------------------------------------------------------------
def test_a_stale_name_that_is_the_file_just_written_is_not_removed(tmp_path):
    """On a file system that does not tell names apart by case, the marker of an earlier build may list
    ``img/LINEAR.png`` for the file this build writes as ``img/linear.png``: pruning the old name would
    delete the new image. Two names of one file, here a hard link, stand for that case on any system."""
    from gelpy.runfolder.report_folder import drop_stale

    folder = _plant(tmp_path / "report", "img/linear.png", "img/old.png")
    try:
        os.link(folder / "img" / "linear.png", folder / "img" / "LINEAR.png")
    except (OSError, NotImplementedError):
        pytest.skip("this file system has no hard links")
    earlier = _marker("img/LINEAR.png", "img/old.png")
    removed = drop_stale(folder, earlier, keep=["index.html", "img/linear.png"])
    assert removed == ["img/old.png"]
    assert (folder / "img" / "linear.png").read_text(encoding="utf-8") == "img/linear.png"


def test_a_marker_naming_a_fresh_file_in_another_case_does_not_cost_the_new_image(tmp_path):
    """On a file system that does not tell names apart by case (the default of macOS and Windows)."""
    from gelpy.runfolder.report_folder import drop_stale

    if _tells_case_apart(tmp_path):
        pytest.skip("this file system tells names apart by case")
    folder = tmp_path / "report"
    (folder / "img").mkdir(parents=True)
    (folder / "img" / "linear.png").write_bytes(b"new")
    assert drop_stale(folder, _marker("img/LINEAR.png"), keep=["img/linear.png"]) == []
    assert (folder / "img" / "linear.png").read_bytes() == b"new"


# --- a name too long for a file ------------------------------------------------------------------------
def test_removing_a_name_the_file_system_finds_too_long_is_not_an_error(tmp_path, monkeypatch):
    """A marker may list a name of up to 255 bytes, which some file systems refuse (ENAMETOOLONG): the
    name cannot exist there, so there is nothing to remove."""
    from gelpy.runfolder.report_folder import remove_files

    folder = _plant(tmp_path / "report", "index.html")
    (folder / "img").mkdir()
    real = os.lstat

    def too_long(path, *args, **kwargs):
        if str(path).endswith("long.png"):
            raise OSError(errno.ENAMETOOLONG, "File name too long", str(path))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(os, "lstat", too_long)
    result = remove_files(folder, ["img/long.png", "index.html"])
    assert result.removed == ["index.html"] and result.left == []


# --- leftovers that cannot be removed -------------------------------------------------------------------
@pytest.fixture
def immovable(monkeypatch):
    """Names of files that cannot be unlinked (an immutable flag, a read-only mount): ``refuse(name)``."""
    refused = set()
    real = os.unlink

    def unlink(path, *args, **kwargs):
        if os.path.basename(str(path)) in refused:
            raise PermissionError(errno.EPERM, "Operation not permitted", str(path))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", unlink)
    return refused.add


def test_sweeping_leftovers_that_cannot_be_removed_is_best_effort(tmp_path, immovable):
    folder = tmp_path / "report"
    folder.mkdir()
    stuck = ".index.html.0123456789ab.tmp"
    (folder / stuck).write_bytes(b"half")
    (folder / ".report.css.ffffffffffff.tmp").write_bytes(b"half")
    immovable(stuck)
    removed = sweep_debris(folder)  # must not raise
    assert removed == [".report.css.ffffffffffff.tmp"]
    assert (folder / stuck).exists()


def test_clearing_a_report_whose_leftovers_cannot_be_removed_still_removes_the_rest(tmp_path, immovable):
    folder = _plant(tmp_path / "report", "index.html", "img/linear.png")
    stuck = ".index.html.0123456789ab.tmp"
    (folder / stuck).write_bytes(b"half")
    immovable(stuck)
    cleared = clear_report(folder)
    assert set(cleared.removed) == {"index.html", "img/linear.png"}
    assert cleared.left == [stuck]
    assert not (folder / REPORT_MARKER).exists()


def test_a_listed_file_that_cannot_be_removed_is_still_an_error(tmp_path, immovable):
    """Only leftovers are best-effort: a report that stayed reachable must not be reported as gone."""
    folder = _plant(tmp_path / "report", "index.html")
    immovable("index.html")
    with pytest.raises(PermissionError):
        clear_report(folder)


def test_a_folder_that_holds_only_leftovers_that_cannot_be_removed_is_left_as_it_is(tmp_path, immovable):
    folder = tmp_path / "report"
    folder.mkdir()
    (folder / DEBRIS).write_bytes(b"half")
    immovable(DEBRIS)
    cleared = clear_report(folder)
    assert cleared.removed == [] and cleared.left == [DEBRIS]
    assert folder.is_dir()
