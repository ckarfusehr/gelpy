"""stale_stages lists exactly the stages the next command that needs them would recompute.

A stage is stale when the ledger records it, it belongs to a lane the recipe still has (or is not a
lane stage), and the cache holds no artifact for the current recipe and running gelpy version. The
ledger lags the cache (an undone edit makes older artifacts current again), so its hashes are never
what decides.
"""

import json

import pytest

import gelpy
from gelpy.runfolder import commands, stages
from gelpy.runfolder.layout import RunPaths
from gelpy.runfolder.manifest import load_manifest
from gelpy.runfolder.recipe_io import load_recipe
from gelpy.schemas.fit_config import FitConfig
from gelpy.schemas.requests import (
    ExtractProfilesRequest,
    FitRunRequest,
    InitRunRequest,
    RemoveBackgroundRequest,
    SetLanesRequest,
    StatusRequest,
)


@pytest.fixture
def fitted_run(gel_image_path, run_dir):
    """Four lanes, background removed, every lane fitted."""
    image_path, centers, _, _ = gel_image_path
    commands.init_run(InitRunRequest(run_dir=run_dir, image_path=image_path, n_lanes=4, width=30))
    commands.set_lanes(SetLanesRequest(run_dir=run_dir, x_centers=[float(c) for c in centers]))
    commands.remove_background(RemoveBackgroundRequest(run_dir=run_dir, model="plane_ransac"))
    _fit_all(run_dir)
    return run_dir


def _fit_all(run):
    return commands.fit_run(FitRunRequest(run_dir=run, lanes="all", fit=FitConfig()))


def _status(run):
    return commands.status(StatusRequest(run_dir=run))


def _fit_file(run, lane=0):
    paths = RunPaths(run)
    return paths.resolve(load_manifest(paths).stages[f"fit:lane={lane}"].cache_path)


def test_a_fitted_run_has_nothing_to_recompute(fitted_run):
    assert _status(fitted_run).stale_stages == []


def _edit_the_lane_width(run, monkeypatch):
    commands.set_lanes(SetLanesRequest(run_dir=run, width=40))


def _install_another_version(run, monkeypatch):
    monkeypatch.setattr(gelpy, "__version__", "99.0.0")


@pytest.mark.parametrize(
    "change",
    [_edit_the_lane_width, _install_another_version],
    ids=["a lane width edit", "another gelpy version"],
)
def test_the_listed_stages_are_exactly_what_the_next_fit_command_recomputes(
    fitted_run, monkeypatch, change
):
    change(fitted_run, monkeypatch)

    listed = _status(fitted_run).stale_stages

    assert listed  # the change made something stale
    assert set(_fit_all(fitted_run).recomputed) == set(listed)
    assert _status(fitted_run).stale_stages == []


def test_undoing_an_edit_leaves_nothing_to_recompute(fitted_run):
    commands.set_lanes(SetLanesRequest(run_dir=fitted_run, width=40))
    _fit_all(fitted_run)  # the ledger now names the width-40 artifacts
    commands.set_lanes(SetLanesRequest(run_dir=fitted_run, width=30))  # undone: width 30 is cached

    assert _status(fitted_run).stale_stages == []
    assert _fit_all(fitted_run).recomputed == []
    status = _status(fitted_run)
    assert status.stale_stages == []
    assert (status.n_fit, status.n_stale) == (4, 0)


def test_returning_to_a_version_whose_caches_exist_leaves_nothing_to_recompute(
    fitted_run, monkeypatch
):
    version = gelpy.__version__
    monkeypatch.setattr(gelpy, "__version__", "99.0.0")
    _fit_all(fitted_run)  # computes everything again; the ledger now names those artifacts
    monkeypatch.setattr(gelpy, "__version__", version)

    assert _status(fitted_run).stale_stages == []
    assert _fit_all(fitted_run).recomputed == []
    assert _status(fitted_run).stale_stages == []


def test_a_fit_cache_hit_records_the_upstream_stages_it_reused(fitted_run, monkeypatch):
    version = gelpy.__version__
    monkeypatch.setattr(gelpy, "__version__", "99.0.0")
    _fit_all(fitted_run)
    monkeypatch.setattr(gelpy, "__version__", version)

    assert _fit_all(fitted_run).recomputed == []

    paths = RunPaths(fitted_run)
    recipe, ledger = load_recipe(paths), load_manifest(paths).stages
    assert ledger["image"].hash == stages.image_hash(recipe)
    assert ledger["background"].hash == stages.background_hash(recipe)
    for lane in recipe.lanes.lanes:
        assert ledger[f"profile:lane={lane.index}"].hash == stages.profile_hash(recipe, lane)
        assert ledger[f"profile:lane={lane.index}"].gelpy_version == version


def test_the_stages_of_a_dropped_lane_are_not_stale(fitted_run):
    commands.set_lanes(SetLanesRequest(run_dir=fitted_run, x_centers=[40.0, 110.0, 180.0]))

    assert _status(fitted_run).stale_stages == []
    _fit_all(fitted_run)
    status = _status(fitted_run)
    assert status.stale_stages == []
    assert (status.n_fit, status.n_stale) == (3, 0)


@pytest.mark.parametrize(
    "record",
    [{"stage": "report", "key": "report"}, {"stage": "fit", "key": "fit"}],
    ids=["a kind this release does not know", "a lane stage without a lane"],
)
def test_a_ledger_record_that_cannot_be_placed_is_ignored(fitted_run, record):
    paths = RunPaths(fitted_run)
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    manifest["stages"][record["key"]] = {**record, "hash": "x", "cache_path": "elsewhere"}
    paths.manifest.write_text(json.dumps(manifest), encoding="utf-8")

    assert _status(fitted_run).stale_stages == []


def test_a_truncated_fit_file_counts_as_missing(fitted_run):
    _fit_file(fitted_run).write_text('{"lane_index": 0, "label"', encoding="utf-8")

    status = _status(fitted_run)  # must not raise

    lane = status.lanes[0]
    assert (lane.has_fit, lane.fit_current, lane.status, lane.n_bands) == (False, False, None, 0)
    assert (status.n_fit, status.n_stale) == (3, 1)
    assert status.stale_stages == ["fit:lane=0"]
    # the next command recovers it, as Pipeline.ensure_fit does for any damaged cache file
    assert _fit_all(fitted_run).recomputed == ["fit:lane=0"]
    assert _status(fitted_run).stale_stages == []


def test_a_missing_fit_file_is_counted_stale_by_n_stale_and_stale_stages(fitted_run):
    _fit_file(fitted_run, lane=2).unlink()

    status = _status(fitted_run)

    assert not status.lanes[2].has_fit
    assert (status.n_fit, status.n_stale) == (3, 1)
    assert status.stale_stages == ["fit:lane=2"]
    assert _fit_all(fitted_run).recomputed == ["fit:lane=2"]
    status = _status(fitted_run)
    assert (status.n_fit, status.n_stale, status.stale_stages) == (4, 0, [])


def test_lane_stages_are_listed_in_numeric_order(gel_image_path, run_dir):
    commands.init_run(
        InitRunRequest(run_dir=run_dir, image_path=gel_image_path[0], n_lanes=12, width=10)
    )
    commands.set_lanes(SetLanesRequest(run_dir=run_dir, linspace=(15.0, 285.0, 12)))
    commands.remove_background(RemoveBackgroundRequest(run_dir=run_dir, model="plane_ransac"))
    commands.extract_profiles(ExtractProfilesRequest(run_dir=run_dir))
    commands.set_lanes(SetLanesRequest(run_dir=run_dir, width=12))

    assert _status(run_dir).stale_stages == [f"profile:lane={i}" for i in range(12)]


# --- a deleted upstream artifact is listed until a command that needs it runs --------------------------
@pytest.mark.parametrize(
    ("stage", "pattern", "needs"),
    [
        ("image", "image/*.npy", lambda run: _render_image(run, "raw")),
        ("background", "background/*.npz", lambda run: _render_image(run, "corrected")),
        (
            "profile:lane=0",
            "profiles/lane_000__*.json",
            lambda run: commands.extract_profiles(ExtractProfilesRequest(run_dir=run, lanes=[0])),
        ),
    ],
)
def test_a_deleted_upstream_artifact_stays_listed_until_a_command_that_needs_it_runs(
    fitted_run, stage, pattern, needs
):
    """A fit that is cached reads one small file and pulls on nothing upstream, so ``fit run`` leaves a
    deleted image, background or profile listed (and says it recomputed nothing); the commands that
    work from the artifact (the renders, the profile extraction) compute it again."""
    next(RunPaths(fitted_run).cache.glob(pattern)).unlink()
    assert _status(fitted_run).stale_stages == [stage]

    assert _fit_all(fitted_run).recomputed == []
    assert _status(fitted_run).stale_stages == [stage]

    needs(fitted_run)
    assert _status(fitted_run).stale_stages == []


def _render_image(run, kind):
    from gelpy.cli import _render

    return _render.render_image(run, kind=kind)
