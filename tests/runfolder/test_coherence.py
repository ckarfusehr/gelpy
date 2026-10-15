"""Regression tests for recipe/results coherence and cache-key robustness.

These cover the paths the original acceptance tests never exercised: editing a
recipe *after* fitting, a failed fit flowing to results, source-image drift, and
hashing hand-editable background params. See the review that motivated them.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from skimage import io

from gelpy.core.errors import SourceImageChangedError, to_error_response
from gelpy.runfolder import commands, stages
from gelpy.runfolder.layout import RunPaths
from gelpy.runfolder.manifest import load_manifest
from gelpy.runfolder.pipeline import Pipeline
from gelpy.runfolder.recipe_io import load_recipe
from gelpy.schemas.image import ImageRef
from gelpy.schemas.lanes import LaneDefinition, LaneLayout
from gelpy.schemas.background import BackgroundStageConfig
from gelpy.schemas.fit_config import FitConfig
from gelpy.schemas.recipe import Recipe
from gelpy.schemas.requests import (
    FitRunRequest,
    GetResultsRequest,
    InitRunRequest,
    RemoveBackgroundRequest,
    SetImageRequest,
    SetLanesRequest,
    StatusRequest,
)
from gelpy.schemas.results import AnalysisResult
from gelpy.schemas.peaks import LaneFitResult


@pytest.fixture
def prepared_run(gel_image_path, run_dir):
    image_path, centers, w, h = gel_image_path
    commands.init_run(InitRunRequest(run_dir=run_dir, image_path=image_path, n_lanes=4, width=30))
    commands.set_lanes(
        SetLanesRequest(run_dir=run_dir, x_centers=[float(c) for c in centers], width=30)
    )
    commands.remove_background(RemoveBackgroundRequest(run_dir=run_dir, model="plane_ransac"))
    return run_dir


# --- A2: stale results after a recipe edit -----------------------------------
def test_recipe_edit_invalidates_stale_results(prepared_run):
    """Editing lane width after fitting changes fit_hash; get_results must not report
    the now-stale fits until the lanes are re-fit."""
    commands.fit_run(
        FitRunRequest(run_dir=prepared_run, lanes="all", fit=FitConfig())
    )
    before = commands.get_results(GetResultsRequest(run_dir=prepared_run))
    assert len(before.lanes) == 4

    # Change the profile width -> every lane's profile_hash (and fit_hash) changes.
    commands.set_lanes(SetLanesRequest(run_dir=prepared_run, width=40))
    after = commands.get_results(GetResultsRequest(run_dir=prepared_run))
    assert after.lanes == []  # stale fits excluded, nothing silently reported

    # Re-fitting under the new recipe restores them.
    commands.fit_run(
        FitRunRequest(run_dir=prepared_run, lanes="all", fit=FitConfig())
    )
    restored = commands.get_results(GetResultsRequest(run_dir=prepared_run))
    assert len(restored.lanes) == 4


def test_status_reports_stale_after_recipe_edit(prepared_run):
    """`status` uses the same per-lane store as results: a recipe edit that changes
    every lane's fit_hash must flip fits from current to stale (never silently 'ok')."""
    commands.fit_run(
        FitRunRequest(run_dir=prepared_run, lanes="all", fit=FitConfig())
    )
    before = commands.status(StatusRequest(run_dir=prepared_run))
    assert before.n_lanes == 4 and before.n_fit == 4 and before.n_stale == 0
    assert all(lane.has_fit and lane.fit_current for lane in before.lanes)

    commands.set_lanes(SetLanesRequest(run_dir=prepared_run, width=40))
    after = commands.status(StatusRequest(run_dir=prepared_run))
    assert after.n_fit == 0 and after.n_stale == 4
    assert all(lane.has_fit and not lane.fit_current for lane in after.lanes)


def test_dropped_lane_not_reported(prepared_run):
    """A lane removed from the recipe must not linger in results from an old fit."""
    commands.fit_run(
        FitRunRequest(run_dir=prepared_run, lanes="all", fit=FitConfig())
    )
    # Reduce to 2 lanes; lanes 2 and 3 no longer exist in the recipe.
    commands.set_lanes(SetLanesRequest(run_dir=prepared_run, x_centers=[40.0, 110.0], width=30))
    result = commands.get_results(GetResultsRequest(run_dir=prepared_run))
    assert all(lane.lane_index < 2 for lane in result.lanes)


# --- A1: failed lanes stay visible in results --------------------------------
def test_failed_lane_appears_in_rows():
    result = AnalysisResult(
        lanes=[
            LaneFitResult(lane_index=0, label="ok", status="failed", error="no peaks"),
        ]
    )
    rows = result.to_rows()
    assert len(rows) == 1
    assert rows[0]["status"] == "failed"
    assert rows[0]["lane_index"] == 0


# --- A3: source-image drift is surfaced, not silently mis-cached -------------
def test_source_drift_raises(prepared_run, gel_image_path):
    """When the source file's bytes change, any stage that actually needs to decode
    the image refuses rather than caching the new pixels under the recipe's stale
    hash. (Fully-cached stages are correctly unaffected — nothing recomputes.)"""
    image_path, centers, w, h = gel_image_path
    # Replace the source file with genuinely different bytes (new sha + mtime).
    io.imsave(image_path, (np.random.default_rng(1).random((h, w)) * 0.5).astype(np.float32))

    paths = RunPaths(prepared_run)
    recipe = load_recipe(paths)
    pipeline = Pipeline(paths, recipe, load_manifest(paths))
    with pytest.raises(SourceImageChangedError):
        pipeline.ensure_image()


def _pipeline(run):
    paths = RunPaths(run)
    return Pipeline(paths, load_recipe(paths), load_manifest(paths))


def test_the_changed_image_message_names_where_it_looked_and_how_the_recipe_records_it(
    prepared_run, gel_image_path
):
    image_path, _, w, h = gel_image_path
    io.imsave(image_path, (np.random.default_rng(1).random((h, w)) * 0.5).astype(np.float32))
    with pytest.raises(SourceImageChangedError) as caught:
        _pipeline(prepared_run).ensure_image()
    message = str(caught.value)
    assert f"Source image {image_path!r} (recorded as '../synthetic_gel.tif') has changed" in message


def test_a_missing_source_image_says_where_to_put_it_back(prepared_run, gel_image_path):
    image_path = gel_image_path[0]
    Path(image_path).unlink()
    with pytest.raises(FileNotFoundError) as caught:
        _pipeline(prepared_run).ensure_image()
    error = to_error_response(caught.value)
    assert error.code == "not_found"  # as for any missing file: callers that branch on it keep working
    assert repr(image_path) in error.error and "recorded as '../synthetic_gel.tif'" in error.error
    assert repr(image_path) in error.detail and "recipe.json" in error.detail


def test_set_image_refuses_a_file_that_is_not_the_recorded_image(gel_image_path, tmp_path, monkeypatch):
    """``RunPaths.resolve`` honours a path that exists only relative to the working directory, as
    gelpy 1.0 recorded paths. A different file of that name found there is not the image the
    recipe records, and its pixels must not be interpreted in the recipe's name."""
    _, _, w, h = gel_image_path
    monkeypatch.chdir(tmp_path)
    commands.init_run(
        InitRunRequest(run_dir="run", image_path="synthetic_gel.tif", n_lanes=4, width=30)
    )
    paths = RunPaths("run")
    recipe = json.loads(paths.recipe.read_text(encoding="utf-8"))
    recipe["source_image"]["path"] = "other.tif"  # relative to the working directory, as 1.0 wrote it
    paths.recipe.write_text(json.dumps(recipe), encoding="utf-8")
    io.imsave("other.tif", (np.random.default_rng(1).random((h, w)) * 0.5).astype(np.float32))
    before = load_recipe(paths).image

    with pytest.raises(SourceImageChangedError):
        commands.set_image(SetImageRequest(run_dir="run", invert=True))
    assert load_recipe(paths).image == before  # nothing was re-interpreted


# --- A4: background_hash hashes validated (canonical) params -----------------
def _bare_recipe(background):
    ref = ImageRef(
        path="/tmp/x.tif", sha256="a" * 64, size=1, mtime=0.0, height=10, width=10, dtype="float32"
    )
    layout = LaneLayout(
        lanes=[LaneDefinition(index=0, label="1", x_center_px=5.0)], profile_width_px=3
    )
    return Recipe(source_image=ref, lanes=layout, background=background)


def test_background_hash_canonicalizes_params():
    """An empty params dict and an explicit ``stripes=None`` are the same request and
    must hash identically — proving the hash runs on the validated model."""
    a = _bare_recipe(BackgroundStageConfig(model="plane_ransac", params={}))
    b = _bare_recipe(BackgroundStageConfig(model="plane_ransac", params={"stripes": None}))
    assert stages.background_hash(a) == stages.background_hash(b)


def test_results_csv_format_matches_the_live_assembly(tmp_path):
    """`results show --format csv` must agree with `--format json`, even when the file is stale.

    results.csv is written by a fit pass. If the CSV format read that file back, then any
    edit that stales a lane would make one command emit two disagreeing answers: the file
    still lists the stale rows, while the assembled result correctly drops them. Rendering
    both from the same assembly is what keeps them consistent.
    """
    from gelpy.runfolder.results import results_csv_text
    from gelpy.schemas.peaks import BandFit, GaussianPeakParams, LaneFitResult
    from gelpy.schemas.results import AnalysisResult

    def _lane(index: int, label: str) -> LaneFitResult:
        return LaneFitResult(
            lane_index=index,
            label=label,
            coverage=1.0,
            signal_area=10.0,
            bands=[
                BandFit(
                    band_number=0,
                    component_class="sample",
                    area=10.0,
                    area_fraction=1.0,
                    maximum_position=50.0,
                    params=GaussianPeakParams(amplitude=1.0, center=50.0, sigma=5.0),
                )
            ],
        )

    two_lanes = AnalysisResult(lanes=[_lane(0, "a"), _lane(1, "b")])
    one_lane = AnalysisResult(lanes=[_lane(0, "a")])

    stale = results_csv_text(two_lanes)
    current = results_csv_text(one_lane)

    # Same header, and the row count follows the assembly it was rendered from.
    assert stale.splitlines()[0] == current.splitlines()[0]
    assert len(stale.splitlines()) == 3
    assert len(current.splitlines()) == 2

    import csv as _csv
    import io as _io

    labels = lambda text: [r["label"] for r in _csv.DictReader(_io.StringIO(text))]
    assert labels(stale) == ["a", "b"]
    assert labels(current) == ["a"], "a dropped lane must not survive into the CSV rendering"
