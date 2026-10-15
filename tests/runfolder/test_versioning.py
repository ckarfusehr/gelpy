"""The gelpy version is recorded in every file that outlives a call, and keys the cache."""

import json

import gelpy
from gelpy.runfolder import commands, stages
from gelpy.runfolder.fit_config_io import save_fit_config
from gelpy.runfolder.layout import RunPaths
from gelpy.runfolder.recipe_io import load_recipe
from gelpy.schemas.fit_config import FitConfig, FitConfigStore
from gelpy.schemas.requests import (
    FitRunRequest,
    GetResultsRequest,
    InitRunRequest,
    RemoveBackgroundRequest,
    SetLanesRequest,
    StatusRequest,
)


def _fitted(gel_image_path, run_dir):
    image_path, centers, _, _ = gel_image_path
    commands.init_run(InitRunRequest(run_dir=run_dir, image_path=image_path, n_lanes=4, width=30))
    commands.set_lanes(
        SetLanesRequest(run_dir=run_dir, x_centers=[float(c) for c in centers], width=30)
    )
    commands.remove_background(RemoveBackgroundRequest(run_dir=run_dir, model="plane_ransac"))
    commands.fit_run(FitRunRequest(run_dir=run_dir, lanes=[0], fit=FitConfig()))
    return run_dir


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_manifest_results_and_history_carry_the_version(gel_image_path, run_dir):
    paths = RunPaths(_fitted(gel_image_path, run_dir))
    assert _json(paths.manifest)["gelpy_version"] == gelpy.__version__
    assert _json(paths.results_json)["gelpy_version"] == gelpy.__version__
    entries = [json.loads(line) for line in paths.history.read_text(encoding="utf-8").splitlines()]
    assert entries
    assert all(entry["gelpy_version"] == gelpy.__version__ for entry in entries)


def test_status_reports_the_running_version(gel_image_path, run_dir):
    _fitted(gel_image_path, run_dir)
    assert commands.status(StatusRequest(run_dir=run_dir)).gelpy_version == gelpy.__version__


def test_a_new_version_never_reuses_an_old_cache(gel_image_path, run_dir, monkeypatch):
    _fitted(gel_image_path, run_dir)
    recipe = load_recipe(RunPaths(run_dir))
    before = stages.image_hash(recipe)

    monkeypatch.setattr(gelpy, "__version__", "99.0.0")

    assert stages.image_hash(recipe) != before
    status = commands.status(StatusRequest(run_dir=run_dir))
    assert (status.n_fit, status.n_stale) == (0, 1)
    assert commands.get_results(GetResultsRequest(run_dir=run_dir)).lanes == []


def test_status_does_not_read_the_cache_file_of_a_stale_fit(gel_image_path, run_dir, monkeypatch):
    """After an upgrade an older release's cache file is the normal state. A release that changed
    LaneFitResult incompatibly must not make `status` fail on every old run folder, so the file
    of a stale fit is never parsed."""
    _fitted(gel_image_path, run_dir)
    fit_file = next((RunPaths(run_dir).cache / "fits").glob("*"))
    fit_file.write_text('{"written": "by an older release"}', encoding="utf-8")
    monkeypatch.setattr(gelpy, "__version__", "99.0.0")

    status = commands.status(StatusRequest(run_dir=run_dir))

    lane = status.lanes[0]
    assert (lane.has_fit, lane.fit_current) == (True, False)
    assert (status.n_fit, status.n_stale) == (0, 1)
    assert (lane.status, lane.n_bands) == (None, 0)  # the file was not read: nothing is claimed
    assert "fit:lane=0" in status.stale_stages


def test_a_fit_without_a_config_entry_is_stale_under_a_new_version(
    gel_image_path, run_dir, monkeypatch
):
    _fitted(gel_image_path, run_dir)
    save_fit_config(RunPaths(run_dir), FitConfigStore())  # the lane keeps its fit, not its intent
    before = commands.status(StatusRequest(run_dir=run_dir))
    assert (before.n_fit, before.n_stale) == (1, 0)  # nothing to check it against: kept

    monkeypatch.setattr(gelpy, "__version__", "99.0.0")
    after = commands.status(StatusRequest(run_dir=run_dir))

    assert (after.n_fit, after.n_stale) == (0, 1)
    assert "fit:lane=0" in after.stale_stages
    assert commands.get_results(GetResultsRequest(run_dir=run_dir)).lanes == []


def test_a_fit_an_older_release_recorded_is_stale_without_a_config_entry(gel_image_path, run_dir):
    """Releases before 1.1 recorded no version on a stage. Without a config entry the version is
    all there is to judge a fit by, so such a record counts as written by another version."""
    paths = RunPaths(_fitted(gel_image_path, run_dir))
    save_fit_config(paths, FitConfigStore())
    manifest = _json(paths.manifest)
    for record in manifest["stages"].values():
        record.pop("gelpy_version", None)  # a 1.0 manifest holds no version on a stage
    paths.manifest.write_text(json.dumps(manifest), encoding="utf-8")

    status = commands.status(StatusRequest(run_dir=run_dir))

    assert (status.n_fit, status.n_stale) == (0, 1)
