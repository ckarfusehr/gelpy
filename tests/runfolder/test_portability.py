"""A run folder works from any working directory and can move together with its image."""

import json
import shutil
from pathlib import Path

from gelpy.runfolder import commands
from gelpy.runfolder.layout import RunPaths
from gelpy.schemas.requests import (
    FitRunRequest,
    GetResultsRequest,
    InitRunRequest,
    RemoveBackgroundRequest,
    SetImageRequest,
    SetLanesRequest,
    StatusRequest,
)


def _init(run, image, **extra):
    commands.init_run(InitRunRequest(run_dir=run, image_path=image, n_lanes=4, width=30, **extra))


def _fit(run, centers, lanes=(0,)):
    commands.set_lanes(SetLanesRequest(run_dir=run, x_centers=[float(c) for c in centers]))
    commands.remove_background(RemoveBackgroundRequest(run_dir=run, model="plane_ransac"))
    return commands.fit_run(FitRunRequest(run_dir=run, lanes=list(lanes)))


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_the_image_path_is_stored_relative_to_the_run(gel_image_path, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _init("analysis/run", "synthetic_gel.tif")
    assert _json("analysis/run/recipe.json")["source_image"]["path"] == "../../synthetic_gel.tif"


def test_a_run_works_from_any_working_directory(gel_image_path, tmp_path, monkeypatch):
    _, centers, _, _ = gel_image_path
    monkeypatch.chdir(tmp_path)
    _init("run", "synthetic_gel.tif")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert "fit:lane=0" in _fit("../run", centers).recomputed
    status = commands.status(StatusRequest(run_dir=str(tmp_path / "run")))
    assert (status.n_fit, status.n_stale, status.stale_stages) == (1, 0, [])
    assert len(commands.get_results(GetResultsRequest(run_dir=str(tmp_path / "run"))).lanes) == 1


def test_a_fit_without_a_stored_config_is_found_from_another_directory(
    gel_image_path, tmp_path, monkeypatch
):
    """With no entry in fit_config.json a fit's file is found through its recorded path alone."""
    _, centers, _, _ = gel_image_path
    monkeypatch.chdir(tmp_path)
    _init("run", "synthetic_gel.tif")
    _fit("run", centers)
    RunPaths("run").fit_config.unlink()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    status = commands.status(StatusRequest(run_dir=str(tmp_path / "run")))
    assert (status.n_fit, status.n_stale, status.stale_stages) == (1, 0, [])


def test_cache_paths_are_recorded_relative_to_the_run(gel_image_path, run_dir):
    image_path, centers, _, _ = gel_image_path
    _init(run_dir, image_path)
    _fit(run_dir, centers)
    for record in _json(RunPaths(run_dir).manifest)["stages"].values():
        assert record["cache_path"].startswith("cache/"), record["cache_path"]


def test_a_run_moved_together_with_its_image_keeps_working(gel_image_path, tmp_path):
    image_path, centers, _, _ = gel_image_path
    data = tmp_path / "data"
    data.mkdir()
    shutil.copy(image_path, data / "gel.tif")
    _init(str(data / "run"), str(data / "gel.tif"))
    _fit(str(data / "run"), centers)
    moved = tmp_path / "moved"
    data.rename(moved)
    run = str(moved / "run")
    status = commands.status(StatusRequest(run_dir=run))
    assert (status.n_fit, status.stale_stages) == (1, [])
    commands.set_image(SetImageRequest(run_dir=run, invert=False))  # reads the moved image
    assert "fit:lane=1" in commands.fit_run(FitRunRequest(run_dir=run, lanes=[1])).recomputed


def test_an_image_on_another_branch_of_the_filesystem_is_stored_absolute(tmp_path):
    outside = Path(tmp_path.anchor) / "elsewhere" / "gel.tif"
    assert RunPaths(tmp_path / "run").store_path(outside) == str(outside)


def test_a_1_0_run_folder_still_opens(gel_image_path, tmp_path, monkeypatch):
    """gelpy 1.0 recorded the image and cache paths relative to the working directory."""
    _, centers, _, _ = gel_image_path
    monkeypatch.chdir(tmp_path)
    _init("run", "synthetic_gel.tif")
    _fit("run", centers)
    paths = RunPaths("run")
    recipe = _json(paths.recipe)
    recipe["source_image"]["path"] = "synthetic_gel.tif"
    paths.recipe.write_text(json.dumps(recipe), encoding="utf-8")
    manifest = _json(paths.manifest)
    for record in manifest["stages"].values():
        record["cache_path"] = "run/" + record["cache_path"]
    paths.manifest.write_text(json.dumps(manifest), encoding="utf-8")

    status = commands.status(StatusRequest(run_dir="run"))
    assert (status.n_fit, status.stale_stages) == (1, [])
    commands.set_image(SetImageRequest(run_dir="run", invert=False))  # reads the image


def test_history_entries_do_not_repeat_the_run_directory(gel_image_path, run_dir):
    image_path, centers, _, _ = gel_image_path
    _init(run_dir, image_path)
    _fit(run_dir, centers)
    lines = RunPaths(run_dir).history.read_text(encoding="utf-8").splitlines()
    for entry in map(json.loads, lines):
        assert "run_dir" not in entry["payload"]


def test_non_ascii_labels_round_trip_through_every_file(gel_image_path, run_dir):
    image_path, centers, _, _ = gel_image_path
    labels = ["Größe", "µ-Probe", "Ünïcödé", "日本"]
    _init(run_dir, image_path, labels=labels)
    _fit(run_dir, centers, lanes=(0, 1))
    paths = RunPaths(run_dir)
    assert [lane["label"] for lane in _json(paths.recipe)["lanes"]["lanes"]] == labels
    text = paths.results_csv.read_text(encoding="utf-8")
    assert "Größe" in text and "µ-Probe" in text
