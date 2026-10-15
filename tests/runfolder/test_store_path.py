"""A stored path opens its target from the run folder, whatever symlinks lie on the way.

The operating system follows ``..`` from the physical directory, so a relative path worked out from
the names alone is wrong when the run folder, or a directory above it, is a link; and one directory
under two names (``/tmp`` and ``/private/tmp``) shares nothing by name."""

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from gelpy.runfolder import commands
from gelpy.runfolder.layout import RunPaths
from gelpy.runfolder.recipe_io import load_recipe
from gelpy.schemas.requests import (
    FitRunRequest,
    InitRunRequest,
    RemoveBackgroundRequest,
    SetImageRequest,
    SetLanesRequest,
    StatusRequest,
)


def _opens(run_dir, stored, image) -> bool:
    """Whether ``stored``, taken from ``run_dir`` as the operating system follows it, is ``image``."""
    try:
        return os.path.samefile(os.path.join(run_dir, stored), image)
    except OSError:
        return False


def _symlink(target: Path, link: Path) -> None:
    try:
        os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError):  # Windows without the privilege, or no links on the volume
        pytest.skip("this system cannot create symbolic links")


@pytest.fixture
def tree(tmp_path):
    """Directories with links at several depths and an image in each data directory::

        home/data/gel.tif       home/analysis  -> disk/analysis
        plain/data/gel.tif      home/ssd       -> disk/ssd_data
        disk/ssd_data/gel.tif   home/work/link -> disk/analysis
        disk/analysis/gel.tif
    """
    base = tmp_path.resolve()
    for directory in ("home/data", "home/work", "disk/analysis", "disk/ssd_data", "plain/data", "plain/runs"):
        (base / directory).mkdir(parents=True)
    for image in ("home/data/gel.tif", "plain/data/gel.tif", "disk/ssd_data/gel.tif", "disk/analysis/gel.tif"):
        (base / image).write_bytes(b"an image is not decoded here")
    _symlink(base / "disk" / "analysis", base / "home" / "analysis")
    _symlink(base / "disk" / "ssd_data", base / "home" / "ssd")
    _symlink(base / "disk" / "analysis", base / "home" / "work" / "link")
    return base


#: (run folder, image, the record when nothing needs working out beyond the names, else None)
LAYOUTS = {
    "no links": ("plain/runs/run", "plain/data/gel.tif", "../../data/gel.tif"),
    "image in a linked directory": ("home/run3", "home/ssd/gel.tif", "../ssd/gel.tif"),
    "run folder below a linked directory": ("home/analysis/run1", "home/data/gel.tif", None),
    "run folder below a link two levels down": ("home/work/link/run2", "home/data/gel.tif", None),
    "run folder and image both through links": ("home/analysis/run4", "home/ssd/gel.tif", None),
    "one directory under two names": ("home/work/link/run5", "disk/analysis/gel.tif", "../gel.tif"),
}


@pytest.mark.parametrize("layout", LAYOUTS, ids=list(LAYOUTS))
def test_a_stored_path_opens_the_target_from_the_run_folder(tree, layout):
    run_part, image_part, tidy = LAYOUTS[layout]
    run_dir, image = tree / run_part, tree / image_part
    run_dir.mkdir(parents=True)
    stored = RunPaths(run_dir).store_path(image)
    assert not Path(stored).is_absolute(), stored  # the directories are shared: no reason for an absolute path
    assert _opens(run_dir, stored, image), f"{stored!r} does not lead from {run_dir} to {image}"
    if tidy is not None:
        assert stored == tidy


def test_a_relative_run_dir_is_recorded_from_its_physical_place(tree, monkeypatch):
    monkeypatch.chdir(tree / "home")
    (tree / "home" / "analysis" / "run6").mkdir()
    stored = RunPaths("analysis/run6").store_path("data/gel.tif")
    assert _opens("analysis/run6", stored, "data/gel.tif"), stored


def test_a_file_that_does_not_exist_yet_gets_the_plain_relative_path(tree):
    run_dir = tree / "plain" / "runs" / "run"
    run_dir.mkdir()
    assert RunPaths(run_dir).store_path(tree / "plain" / "data" / "later.tif") == "../../data/later.tif"


def test_nothing_shared_below_the_filesystem_root_is_stored_absolute(tree):
    run_dir = tree / "plain" / "runs" / "run"
    run_dir.mkdir()
    elsewhere = Path(tree.anchor) / "elsewhere" / "gel.tif"
    assert RunPaths(run_dir).store_path(elsewhere) == str(elsewhere)


def test_a_run_folder_reached_through_a_symlink_works(gel_image_path, tmp_path, monkeypatch):
    """The 1.0 behaviour, and the regression of the first 1.1 build: the recorded image path was
    worked out from the names, so it led nowhere once ``..`` was followed physically."""
    _, centers, _, _ = gel_image_path
    real = tmp_path / "disk" / "analysis"
    real.mkdir(parents=True)
    _symlink(real, tmp_path / "analysis")
    monkeypatch.chdir(tmp_path)
    commands.init_run(
        InitRunRequest(run_dir="analysis/run", image_path="synthetic_gel.tif", n_lanes=4, width=30)
    )
    commands.set_lanes(SetLanesRequest(run_dir="analysis/run", x_centers=[float(c) for c in centers]))
    commands.remove_background(RemoveBackgroundRequest(run_dir="analysis/run", model="plane_ransac"))
    summary = commands.fit_run(FitRunRequest(run_dir="analysis/run", lanes=[0]))
    assert "fit:lane=0" in summary.recomputed
    status = commands.status(StatusRequest(run_dir=str(real / "run")))
    assert (status.n_fit, status.n_stale, status.stale_stages) == (1, 0, [])


@pytest.mark.skipif(os.path.realpath("/tmp") == "/tmp", reason="/tmp is not a symlink on this system")
def test_one_directory_spelled_with_and_without_its_link_is_not_stored_absolute():
    """macOS: /tmp is a link to /private/tmp, so a run under /tmp and an image under /private/tmp
    share no name below the root although they are siblings."""
    base = tempfile.mkdtemp(dir="/tmp")
    try:
        real = os.path.realpath(base)
        (Path(real) / "gel.tif").write_bytes(b"an image is not decoded here")
        (Path(base) / "run").mkdir()
        assert RunPaths(Path(base) / "run").store_path(Path(real) / "gel.tif") == "../gel.tif"
    finally:
        shutil.rmtree(base)


def test_a_path_with_dot_dot_after_a_link_is_recorded_for_the_file_the_system_opens(tree):
    """``home/analysis`` is a link to ``disk/analysis``, so the system takes ``home/analysis/../data``
    to ``disk/data``: ``..`` leaves the directory the link points to, not the one it stands in. The
    names alone collapse it to ``home/data``, which here holds another file."""
    (tree / "disk" / "data").mkdir()
    real = tree / "disk" / "data" / "gel.tif"
    real.write_bytes(b"the file the operating system opens")
    run_dir = tree / "plain" / "runs" / "run"
    run_dir.mkdir()
    typed = os.path.join(str(tree), "home", "analysis", "..", "data", "gel.tif")

    stored = RunPaths(run_dir).store_path(typed)

    assert _opens(run_dir, stored, real), f"{stored!r} does not lead from {run_dir} to {real}"
    assert not _opens(run_dir, stored, tree / "home" / "data" / "gel.tif")


def test_init_records_the_file_it_read_when_the_path_has_dot_dot_after_a_link(
    tree, gel_image_path, monkeypatch
):
    real = tree / "disk" / "data" / "gel.tif"
    real.parent.mkdir()
    shutil.copy(gel_image_path[0], real)
    monkeypatch.chdir(tree / "home")  # home/data/gel.tif is the decoy the names collapse to

    commands.init_run(
        InitRunRequest(run_dir="run", image_path="analysis/../data/gel.tif", n_lanes=4, width=30)
    )

    paths = RunPaths("run")
    assert os.path.samefile(paths.resolve(load_recipe(paths).source_image.path), real)
    commands.set_image(SetImageRequest(run_dir="run", invert=False))  # reads the recorded file again
