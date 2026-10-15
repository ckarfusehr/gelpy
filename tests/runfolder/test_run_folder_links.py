"""gelpy never writes through a link inside a run folder.

A run folder can come from anyone: a name in it may have been pointed at a file or a folder
outside it. Where gelpy writes a file, a link is replaced by the new file (the atomic writer, which
never writes through it). Where it appends (log/history.jsonl) or locks (.gelpy.lock), a link is
refused. A folder of the run that is a link is refused before anything is written. The run folder
itself may be a link: the person passes it.
"""

import json
import os
import shutil
import sys
import typing
from pathlib import Path

import numpy as np
import pytest
from skimage import io
from typer.testing import CliRunner

from gelpy.cli import _render, tools
from gelpy.cli.app import app
from gelpy.core.errors import InvalidValueError, to_error_response
from gelpy.runfolder import commands
from gelpy.runfolder.layout import RunPaths
from gelpy.schemas.profiles import LineProfileConfig
from gelpy.schemas.requests import (
    DetectLanesRequest,
    ExtractProfilesRequest,
    FitRunRequest,
    GetResultsRequest,
    InitRunRequest,
    RemoveBackgroundRequest,
    SetContrastRequest,
    SetLanesRequest,
    StatusRequest,
)

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="symlinks need extra privileges")

#: The folders of a run that gelpy writes into, shallowest first.
FOLDERS = [
    "cache",
    "cache/image",
    "cache/background",
    "cache/profiles",
    "cache/fits",
    "render",
    "render/fits",
    "log",
]
PRECIOUS = b"precious: not for gelpy to touch"


def _gaussian(x, amp, mean, sd):
    return amp * np.exp(-((x - mean) ** 2) / (2 * sd**2))


def _write_gel(path: Path) -> None:
    height, width = 250, 300
    band = _gaussian(np.arange(height), 1.0, 80, 6) + _gaussian(np.arange(height), 0.7, 160, 6)
    image = np.zeros((height, width))
    for centre in (40, 110, 180, 250):
        image[:, centre - 15 : centre + 16] += band[:, None]
    gx, gy = np.meshgrid(np.arange(width), np.arange(height))
    image += 0.02 * gx + 0.01 * gy + 5.0
    io.imsave(str(path), (image / image.max()).astype(np.float32))


@pytest.fixture(scope="module")
def fitted_workspace(tmp_path_factory):
    """A folder holding a gel and its run, with every file gelpy writes into a run present."""
    work = tmp_path_factory.mktemp("links")
    _write_gel(work / "gel.tif")
    run = str(work / "run")
    commands.init_run(InitRunRequest(run_dir=run, image_path=str(work / "gel.tif"), n_lanes=4, width=30))
    commands.set_lanes(SetLanesRequest(run_dir=run, x_centers=[40.0, 110.0, 180.0, 250.0]))
    commands.remove_background(RemoveBackgroundRequest(run_dir=run, model="plane_ransac"))
    commands.fit_run(FitRunRequest(run_dir=run, lanes=[0, 1]))
    _render.write_report(run)
    for kind in ("raw", "corrected"):
        _render.render_image(run, kind=kind)
    _render.render_profiles(run, "all", LineProfileConfig())
    _render.render_fits(run, [0, 1])
    _render.render_background(run)
    tools.run_tool("lanes_detect", DetectLanesRequest(run_dir=run, n_lanes=4))
    return work


@pytest.fixture
def fitted(fitted_workspace, tmp_path):
    """A private copy of the fitted run (with its gel, so the recorded image path still opens)."""
    work = tmp_path / "work"
    shutil.copytree(fitted_workspace, work, symlinks=True)
    return work / "run"


@pytest.fixture
def started(tmp_path):
    """A freshly initialised run (the folders are there, empty) and the folder a link will point at."""
    _write_gel(tmp_path / "gel.tif")
    run = tmp_path / "run"
    commands.init_run(
        InitRunRequest(run_dir=str(run), image_path=str(tmp_path / "gel.tif"), n_lanes=4, width=30)
    )
    commands.remove_background(RemoveBackgroundRequest(run_dir=str(run), model="plane_ransac"))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    return run, elsewhere


def _link_folder(run: Path, folder: str, elsewhere: Path) -> None:
    """Make ``folder`` of the run a link to ``elsewhere``."""
    shutil.rmtree(run / folder)
    (run / folder).symlink_to(elsewhere, target_is_directory=True)


def _tree(folder: Path) -> dict[str, bytes]:
    return {
        p.relative_to(folder).as_posix(): p.read_bytes() for p in sorted(folder.rglob("*")) if p.is_file()
    }


# --- a link where a file is written is replaced ----------------------------------------------------
def _set_contrast(run):
    commands.set_contrast(SetContrastRequest(run_dir=str(run), gamma=0.3))


def _fit_lane_0(run):
    commands.fit_run(FitRunRequest(run_dir=str(run), lanes=[0]))


#: (file as a glob below the run, what the link points at, the call that writes the file). "copy": a
#: copy of the file, for a file the call reads before it writes; "garbage": a file of other content,
#: for a cache file (a read that fails is a miss: it is written again) or a file only written;
#: "dangling": a name that does not exist, which writing through the link would create.
LEAVES = [
    pytest.param("recipe.json", "copy", _set_contrast, id="recipe.json"),
    pytest.param("manifest.json", "garbage", _set_contrast, id="manifest.json"),
    pytest.param("fit_config.json", "copy", _fit_lane_0, id="fit_config.json"),
    pytest.param("results.csv", "garbage", _fit_lane_0, id="results.csv"),
    pytest.param("results.csv", "dangling", _fit_lane_0, id="results.csv dangling"),
    pytest.param("results.json", "garbage", _fit_lane_0, id="results.json"),
    pytest.param("report.md", "garbage", lambda run: _render.write_report(str(run)), id="report.md"),
    pytest.param(
        "cache/image/*.npy", "garbage", lambda run: _render.render_image(str(run), kind="raw"), id="image cache"
    ),
    pytest.param(
        "cache/background/*.npz",
        "garbage",
        lambda run: _render.render_image(str(run), kind="corrected"),
        id="background cache",
    ),
    pytest.param(
        "cache/profiles/lane_000__*.json",
        "garbage",
        lambda run: commands.extract_profiles(ExtractProfilesRequest(run_dir=str(run), lanes=[0])),
        id="profile cache",
    ),
    pytest.param("cache/fits/lane_000__*.json", "garbage", _fit_lane_0, id="fit cache"),
    pytest.param(
        "render/raw.png", "garbage", lambda run: _render.render_image(str(run), kind="raw"), id="raw.png"
    ),
    pytest.param(
        "render/raw.png", "dangling", lambda run: _render.render_image(str(run), kind="raw"), id="raw.png dangling"
    ),
    pytest.param(
        "render/profiles_overview.png",
        "garbage",
        lambda run: _render.render_profiles(str(run), "all", LineProfileConfig()),
        id="profiles_overview.png",
    ),
    pytest.param(
        "render/fits/lane_000.png", "garbage", lambda run: _render.render_fits(str(run), [0]), id="fit plot"
    ),
    pytest.param(
        "render/fits/index.json", "garbage", lambda run: _render.render_fits(str(run), [0]), id="fit plot index"
    ),
    pytest.param(
        "render/background_diagnostic.png",
        "garbage",
        lambda run: _render.render_background(str(run)),
        id="background diagnostic",
    ),
    pytest.param(
        "render/lane_detect.png",
        "garbage",
        lambda run: tools.run_tool("lanes_detect", DetectLanesRequest(run_dir=str(run), n_lanes=4)),
        id="lane overlay",
    ),
]


@pytest.mark.parametrize(("pattern", "mode", "write"), LEAVES)
def test_a_link_where_a_file_is_written_is_replaced_and_its_target_is_untouched(
    fitted, tmp_path, pattern, mode, write
):
    leaf = sorted(fitted.glob(pattern))[0]
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    victim = elsewhere / "victim"
    if mode == "copy":
        victim.write_bytes(leaf.read_bytes())
    elif mode == "garbage":
        victim.write_bytes(PRECIOUS)
    planted = victim.read_bytes() if victim.exists() else None
    leaf.unlink()
    leaf.symlink_to(victim)

    write(fitted)

    assert (victim.read_bytes() if victim.exists() else None) == planted
    assert leaf.exists() and not leaf.is_symlink()  # the link is gone: the file gelpy wrote is there


# --- the append log and the lock file are refused when they are links -------------------------------
def test_the_history_log_is_not_appended_through_a_link(fitted, tmp_path):
    history = RunPaths(fitted).history
    victim = tmp_path / "victim.jsonl"
    victim.write_bytes(history.read_bytes())
    history.unlink()
    history.symlink_to(victim)
    before = victim.read_bytes()
    recipe = (fitted / "recipe.json").read_bytes()

    with pytest.raises(InvalidValueError) as refused:
        _set_contrast(fitted)

    assert victim.read_bytes() == before  # not appended to
    assert (fitted / "recipe.json").read_bytes() == recipe  # and nothing else was written first
    response = to_error_response(refused.value)
    assert response.code == "invalid_value" and "history.jsonl" in response.error and "link" in response.error
    assert "Delete" in response.detail and "repeat the command" in response.detail


def test_the_history_log_is_not_created_through_a_dangling_link(fitted, tmp_path):
    history = RunPaths(fitted).history
    history.unlink()
    victim = tmp_path / "created-by-a-write-through-the-link.jsonl"
    history.symlink_to(victim)

    with pytest.raises(InvalidValueError):
        _set_contrast(fitted)

    assert not victim.exists()


def test_the_lock_file_is_not_opened_through_a_link(fitted, tmp_path):
    lock = RunPaths(fitted).lock
    victim = tmp_path / "created-by-opening-the-link.lock"
    lock.unlink()
    lock.symlink_to(victim)  # dangling: opening it with O_CREAT would make the file outside the run

    with pytest.raises(InvalidValueError) as refused:
        _set_contrast(fitted)

    assert not victim.exists()
    assert ".gelpy.lock" in str(refused.value) and "link" in str(refused.value)


def test_a_lock_file_that_is_a_link_to_a_file_is_refused_too(fitted, tmp_path):
    victim = tmp_path / "victim.lock"
    victim.write_bytes(PRECIOUS)
    RunPaths(fitted).lock.unlink()
    RunPaths(fitted).lock.symlink_to(victim)
    with pytest.raises(InvalidValueError):
        _set_contrast(fitted)
    assert victim.read_bytes() == PRECIOUS


# --- a link where a folder belongs is refused before anything is written ----------------------------
READS = {"get_results", "status"}
NOT_WRITERS = {"image_warnings"}
#: Request fields besides run_dir that have no default.
EXTRA_FIELDS = {"refit_lane": {"lane": 0}}


def _public_functions(module):
    return {
        name: func
        for name, func in vars(module).items()
        if callable(func)
        and getattr(func, "__module__", None) == module.__name__
        and not name.startswith("_")
        and not isinstance(func, type)
    }


#: Every writer but init (which makes the folders, and has tests of its own below).
WRITERS = [
    pytest.param(commands, name, id=f"commands.{name}")
    for name in sorted(set(_public_functions(commands)) - READS - NOT_WRITERS - {"init_run"})
] + [pytest.param(_render, name, id=f"_render.{name}") for name in sorted(_public_functions(_render))]


#: What each renderer takes after the run folder.
RENDER_ARGS = {
    "render_profiles": ("all", LineProfileConfig()),
    "render_fits": ([0],),
    "ensure_fit_plots": ([0],),
    "render_lane_overlay": (None,),  # the outcome of a detection: the refusal comes before it is read
}


def _call(owner, name, run):
    func = getattr(owner, name)
    if owner is _render:
        return func(str(run), *RENDER_ARGS.get(name, ()))
    model = typing.get_type_hints(func)["request"]
    return func(model(run_dir=str(run), **EXTRA_FIELDS.get(name, {})))


@pytest.mark.parametrize("folder", FOLDERS)
@pytest.mark.parametrize(("owner", "name"), WRITERS)
def test_every_writer_refuses_a_linked_folder_before_it_writes(started, owner, name, folder):
    run, elsewhere = started
    _link_folder(run, folder, elsewhere)
    before = _tree(run)

    with pytest.raises(InvalidValueError) as refused:
        _call(owner, name, run)

    assert list(elsewhere.iterdir()) == []  # nothing was created through the link
    assert _tree(run) == before  # and nothing was written anywhere else
    response = to_error_response(refused.value)
    assert response.code == "invalid_value"
    assert f"{folder}/" in response.error and "link" in response.error  # it names the folder
    assert "Delete" in response.detail and "repeat the command" in response.detail  # and the next step


def test_the_writers_are_the_ones_the_run_lock_pins():
    """A writer added later is listed here, so that it is tested for a linked folder too."""
    assert {param.id for param in WRITERS} == {
        "commands.detect_lanes",
        "commands.extract_profiles",
        "commands.fit_run",
        "commands.refit_lane",
        "commands.remove_background",
        "commands.run_lane_detection",
        "commands.set_contrast",
        "commands.set_image",
        "commands.set_lanes",
        "_render.ensure_fit_plots",
        "_render.render_background",
        "_render.render_fits",
        "_render.render_image",
        "_render.render_lane_overlay",
        "_render.render_profiles",
        "_render.write_html_report",
        "_render.write_report",
    }


@pytest.mark.parametrize("folder", FOLDERS)
def test_a_linked_folder_is_refused_in_a_run_that_has_been_fitted(fitted, tmp_path, folder):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    _link_folder(fitted, folder, elsewhere)
    with pytest.raises(InvalidValueError):
        _set_contrast(fitted)
    assert list(elsewhere.iterdir()) == []


def test_a_file_where_a_folder_belongs_is_refused_too(started):
    run, _ = started
    shutil.rmtree(run / "render")
    (run / "render").write_bytes(b"not a folder")
    with pytest.raises(InvalidValueError) as refused:
        _call(_render, "render_image", run)
    assert "render/" in str(refused.value) and "not a folder" in str(refused.value)


def test_a_folder_that_is_absent_is_made_as_a_real_folder(started):
    run, _ = started
    shutil.rmtree(run / "render")
    _render.render_image(str(run), kind="raw")
    assert (run / "render").is_dir() and not (run / "render").is_symlink()
    assert (run / "render" / "raw.png").is_file()


def test_reads_are_not_refused_by_a_linked_folder(started):
    """Only writing through a link is refused: status and results answer as they always did."""
    run, elsewhere = started
    _link_folder(run, "render", elsewhere)
    _link_folder(run, "log", elsewhere)
    assert commands.status(StatusRequest(run_dir=str(run))).n_lanes == 4
    assert commands.get_results(GetResultsRequest(run_dir=str(run))).lanes == []


def test_a_run_folder_that_is_itself_a_link_is_written_as_usual(fitted, tmp_path):
    """The person passes the run folder: it may be a link, and a link above the folders is fine."""
    alias = tmp_path / "alias"
    alias.symlink_to(fitted, target_is_directory=True)
    commands.set_contrast(SetContrastRequest(run_dir=str(alias), gamma=0.4))
    _render.render_image(str(alias), kind="raw")
    assert json.loads((fitted / "recipe.json").read_text(encoding="utf-8"))["contrast"]["gamma"]["gamma"] == 0.4
    assert (fitted / "render" / "raw.png").is_file()


# --- init -------------------------------------------------------------------------------------------
@pytest.mark.parametrize("folder", [f for f in FOLDERS if not f.startswith("render")])
def test_init_force_refuses_a_linked_folder_and_leaves_the_old_run_as_it_was(fitted, tmp_path, folder):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    _link_folder(fitted, folder, elsewhere)
    results = (fitted / "results.csv").read_bytes()
    recipe = (fitted / "recipe.json").read_bytes()

    with pytest.raises(InvalidValueError):
        commands.init_run(
            InitRunRequest(
                run_dir=str(fitted),
                image_path=str(fitted.parent / "gel.tif"),
                n_lanes=3,
                force=True,
            )
        )

    assert list(elsewhere.iterdir()) == []
    assert (fitted / "results.csv").read_bytes() == results  # nothing was removed first
    assert (fitted / "recipe.json").read_bytes() == recipe


def test_init_force_replaces_a_linked_render_folder_whole_and_never_follows_it(fitted, tmp_path):
    """``init --force`` removes the renders: a link where they were is removed with them, and what it
    pointed at is not touched."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "keep.png").write_bytes(b"png")
    (elsewhere / "fits").mkdir()
    shutil.rmtree(fitted / "render")
    (fitted / "render").symlink_to(elsewhere, target_is_directory=True)

    commands.init_run(
        InitRunRequest(run_dir=str(fitted), image_path=str(fitted.parent / "gel.tif"), n_lanes=3, force=True)
    )

    assert sorted(p.name for p in elsewhere.iterdir()) == ["fits", "keep.png"]
    assert (fitted / "render").is_dir() and not (fitted / "render").is_symlink()


@pytest.mark.parametrize("folder", FOLDERS)
def test_init_refuses_a_link_where_it_must_make_a_folder(tmp_path, folder):
    _write_gel(tmp_path / "gel.tif")
    run = tmp_path / "run"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    parent, _, leaf = folder.rpartition("/")
    (run / parent).mkdir(parents=True)
    (run / folder).symlink_to(elsewhere, target_is_directory=True)

    with pytest.raises(InvalidValueError):
        commands.init_run(InitRunRequest(run_dir=str(run), image_path=str(tmp_path / "gel.tif"), n_lanes=3))

    assert list(elsewhere.iterdir()) == []
    assert not (run / "recipe.json").exists()


def test_init_does_not_append_to_a_history_log_that_is_a_link(tmp_path):
    _write_gel(tmp_path / "gel.tif")
    run = tmp_path / "run"
    (run / "log").mkdir(parents=True)
    victim = tmp_path / "victim.jsonl"
    victim.write_bytes(PRECIOUS)
    (run / "log" / "history.jsonl").symlink_to(victim)

    with pytest.raises(InvalidValueError):
        commands.init_run(InitRunRequest(run_dir=str(run), image_path=str(tmp_path / "gel.tif"), n_lanes=3))

    assert victim.read_bytes() == PRECIOUS
    assert not (run / "recipe.json").exists()


# --- through the command line ------------------------------------------------------------------------
def test_the_command_line_answers_a_linked_folder_with_one_structured_error(started):
    run, elsewhere = started
    _link_folder(run, "render", elsewhere)

    result = CliRunner().invoke(app, ["image", "show", str(run), "--kind", "raw"])

    assert result.exit_code == 1 and result.stdout == ""
    error = json.loads(result.stderr)
    assert error["code"] == "invalid_value"
    assert "render/" in error["error"] and "link" in error["error"]
    assert "repeat the command" in error["detail"]
    assert list(elsewhere.iterdir()) == []
