"""Acceptance tests for the hard requirement: cheap targeted edits.

Proved via call-count spies + file-mtime checks, not wall-clock timing.
"""

import numpy as np
import pytest

from gelpy.runfolder import commands, pipeline
from gelpy.runfolder.layout import RunPaths
from gelpy.schemas.fit_config import FitConfig
from gelpy.schemas.profiles import LineProfileConfig
from gelpy.schemas.requests import (
    FitRunRequest,
    GetResultsRequest,
    InitRunRequest,
    FitConfigOverrides,
    RefitLaneRequest,
    RemoveBackgroundRequest,
    SetLanesRequest,
)


@pytest.fixture
def prepared_run(gel_image_path, run_dir):
    """A run folder with 4 lanes and background removal configured."""
    image_path, centers, w, h = gel_image_path
    commands.init_run(InitRunRequest(run_dir=run_dir, image_path=image_path, n_lanes=4, width=30))
    commands.set_lanes(
        SetLanesRequest(run_dir=run_dir, x_centers=[float(c) for c in centers], width=30)
    )
    commands.remove_background(RemoveBackgroundRequest(run_dir=run_dir, model="plane_ransac"))
    return run_dir


def _spy(monkeypatch, target_module, name):
    calls = {"n": 0}
    original = getattr(target_module, name)

    def wrapper(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(target_module, name, wrapper)
    return calls


def test_two_lanes_one_background_computation(prepared_run, monkeypatch):
    """(a) Two lanes fit back-to-back with unchanged background => background computed once,
    and a second fit pass reuses the cached background (zero further computations)."""
    from gelpy.core.background import plane_ransac

    bg_calls = _spy(monkeypatch, plane_ransac, "fit_plane")

    commands.fit_run(FitRunRequest(run_dir=prepared_run, lanes=[0, 1]))
    assert bg_calls["n"] == 1  # one compute shared across both lanes

    bg_calls["n"] = 0
    commands.fit_run(FitRunRequest(run_dir=prepared_run, lanes=[2, 3]))
    assert bg_calls["n"] == 0  # background cache hit: never recomputed


def test_refit_one_lane_touches_only_that_fit(prepared_run):
    """(b) Changing one lane's fit config recomputes only that lane's fit and leaves
    every other cache file — and the image/background caches — untouched."""
    # Broad pass across all lanes.
    commands.fit_run(
        FitRunRequest(
            run_dir=prepared_run,
            lanes="all",
            profile=LineProfileConfig(slice_start=0, slice_end=None),
            fit=FitConfig(),
        )
    )
    paths = RunPaths(prepared_run)

    def mtimes(directory):
        return {p.name: p.stat().st_mtime_ns for p in directory.glob("*")}

    image_before = mtimes(paths.cache / "image")
    bg_before = mtimes(paths.cache / "background")
    profiles_before = mtimes(paths.cache / "profiles")
    fits_before = mtimes(paths.cache / "fits")

    # Targeted refit of lane 1 only, changing the fit seed (forces a new fit_hash).
    summary = commands.refit_lane(
        RefitLaneRequest(
            run_dir=prepared_run,
            lane=1,
            fit_overrides=FitConfigOverrides(seed=1),
        )
    )

    assert summary.recomputed == ["fit:lane=1"]
    assert "image" in summary.reused
    assert "background" in summary.reused
    assert "profile:lane=1" in summary.reused

    # image + background caches completely untouched.
    assert mtimes(paths.cache / "image") == image_before
    assert mtimes(paths.cache / "background") == bg_before
    # profiles untouched (same width/slice as broad pass).
    assert mtimes(paths.cache / "profiles") == profiles_before
    # exactly one new fit file added (lane 1's new hash); the old files remain untouched.
    fits_after = mtimes(paths.cache / "fits")
    assert set(fits_before).issubset(set(fits_after))
    for name, mtime in fits_before.items():
        assert fits_after[name] == mtime  # pre-existing fit files unchanged
    assert len(fits_after) == len(fits_before) + 1


def test_identical_fit_recomputes_nothing(prepared_run):
    """Rerunning an identical fit reuses the fit cache — nothing recomputes."""
    req = FitRunRequest(run_dir=prepared_run, lanes=[0], fit=FitConfig())
    commands.fit_run(req)
    summary = commands.fit_run(req)
    assert summary.recomputed == []
    assert "fit:lane=0" in summary.reused


def test_corrupt_fit_cache_recovers(prepared_run):
    """(c) A corrupted fit cache file is detected and transparently recomputed."""
    req = FitRunRequest(run_dir=prepared_run, lanes=[0], fit=FitConfig())
    commands.fit_run(req)
    paths = RunPaths(prepared_run)
    fit_file = next((paths.cache / "fits").glob("*"))
    fit_file.write_text("{ this is not valid json")

    summary = commands.fit_run(req)  # must not raise
    assert "fit:lane=0" in summary.recomputed
    # results still assemble
    result = commands.get_results(GetResultsRequest(run_dir=prepared_run))
    assert any(lane.lane_index == 0 for lane in result.lanes)


def test_results_csv_columns(prepared_run):
    commands.fit_run(
        FitRunRequest(run_dir=prepared_run, lanes="all", fit=FitConfig())
    )
    paths = RunPaths(prepared_run)
    header = paths.results_csv.read_text().splitlines()[0]
    for col in ["lane_index", "label", "band_number", "component_class", "area_fraction", "maximum_position"]:
        assert col in header
