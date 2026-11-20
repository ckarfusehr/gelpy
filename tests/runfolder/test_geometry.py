"""Geometry in a run folder: recorded in recipe.json as part of the image interpretation,
replayed by ``apply_image_meta``, folded into the hash chain, and harmless to the analysis.

The example-gel tests turn, mirror, pad or tilt a copy of a repository example gel and let gelpy
undo it: the quarter-turn path must reproduce the original analysis exactly; the fine-rotation
path (a copy tilted by scipy, straightened by gelpy: two bilinear passes) within 1 px for lane
centres and for band positions, against a lane pitch of about 176 px and bands several pixels
wide. Lane detection fits an even grid; the two passes change the fitted pitch by about 0.1 px
(measured), which moves the outer lanes by about half a pixel, with no offset in the middle.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
from scipy import ndimage
from scipy.signal import find_peaks

from gelpy.core.errors import InvalidValueError
from gelpy.core.geometry import apply_geometry, rotated_shape
from gelpy.core.image_io import apply_image_meta
from gelpy.runfolder import commands, stages
from gelpy.runfolder.layout import RunPaths
from gelpy.runfolder.manifest import load_manifest
from gelpy.runfolder.pipeline import Pipeline
from gelpy.runfolder.recipe_io import load_recipe
from gelpy.schemas.image import ImageGeometry
from gelpy.schemas.requests import (
    CropImageRequest,
    DetectLanesRequest,
    ExtractProfilesRequest,
    InitRunRequest,
    RotateImageRequest,
    SetImageRequest,
    StatusRequest,
)

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "data"


def _init(image_path, run_dir, n_lanes=4, **kwargs):
    commands.init_run(InitRunRequest(run_dir=run_dir, image_path=str(image_path), n_lanes=n_lanes, **kwargs))


def _pipeline(run_dir):
    paths = RunPaths(run_dir)
    return Pipeline(paths, load_recipe(paths), load_manifest(paths))


def _rotate(run_dir, **kwargs):
    return commands.rotate_image(RotateImageRequest(run_dir=run_dir, **kwargs))


def _crop(run_dir, box):
    return commands.crop_image(CropImageRequest(run_dir=run_dir, box=box))


# --- the record and its replay ------------------------------------------------------------
def test_replaying_the_recorded_meta_gives_exactly_the_array_the_stages_analysed(gel_image_path, run_dir):
    path, *_ = gel_image_path  # 300 x 250 px
    _init(path, run_dir)
    _rotate(run_dir, turn=90, flip=True, angle=2.5)
    _crop(run_dir, (10, 12, 240, 300))

    on_disk = json.loads((Path(run_dir) / "recipe.json").read_text())
    assert on_disk["image"]["geometry"] == {"turn": 90, "flip": True, "angle": 2.5, "crop": [10, 12, 240, 300]}

    staged = _pipeline(run_dir).ensure_image()
    recipe = load_recipe(RunPaths(run_dir))
    cached = np.load(RunPaths(run_dir).image_cache(stages.image_hash(recipe)))
    replayed = apply_image_meta(path, recipe.image)  # one call, from the recorded meta alone
    assert replayed.shape == (288, 230)
    assert np.array_equal(replayed, staged) and np.array_equal(replayed, cached)

    # the order is decode, interpret, then turn, flip, angle, crop
    unoriented = apply_image_meta(path, recipe.image.model_copy(update={"geometry": ImageGeometry()}))
    assert np.array_equal(replayed, apply_geometry(unoriented, recipe.image.geometry))


def test_the_geometry_is_logged_and_shown_by_status(gel_image_path, run_dir):
    path, *_ = gel_image_path
    _init(path, run_dir)
    summary = _rotate(run_dir, turn=-90)
    assert summary.image["geometry"]["turn"] == 270
    assert (summary.image["oriented_width"], summary.image["oriented_height"]) == (250, 300)
    _crop(run_dir, (5, 6, 205, 286))
    state = commands.status(StatusRequest(run_dir=run_dir))
    assert state.geometry == ImageGeometry(turn=270, crop=(5, 6, 205, 286))
    assert (state.analysis_width, state.analysis_height) == (200, 280)
    assert (state.image_width, state.image_height) == (300, 250)  # the source's own size
    history = [json.loads(line) for line in (Path(run_dir) / "log" / "history.jsonl").read_text().splitlines()]
    assert [entry["command"] for entry in history][-2:] == ["image_rotate", "image_crop"]


def test_a_geometry_change_recomputes_and_its_undo_reuses_the_cache(gel_image_path, run_dir):
    path, *_ = gel_image_path
    _init(path, run_dir)
    extract = ExtractProfilesRequest(run_dir=run_dir, allow_raw=True)
    assert "image" in commands.extract_profiles(extract).recomputed
    _rotate(run_dir, turn=180)
    assert "image" in commands.extract_profiles(extract).recomputed
    _rotate(run_dir, turn=0)
    again = commands.extract_profiles(extract)  # the first geometry again: all cached
    assert again.recomputed == [] and "profile:lane=0" in again.reused


def test_a_new_orientation_removes_a_crop_chosen_on_the_old_one(gel_image_path, run_dir):
    path, *_ = gel_image_path
    _init(path, run_dir)
    _rotate(run_dir, turn=90)
    _crop(run_dir, (0, 0, 200, 200))
    same = _rotate(run_dir, turn=90)
    assert load_recipe(RunPaths(run_dir)).image.geometry.crop == (0, 0, 200, 200)
    assert same.warnings == []
    other = _rotate(run_dir, turn=90, angle=1.0)
    assert load_recipe(RunPaths(run_dir)).image.geometry.crop is None
    assert any("crop was removed" in warning for warning in other.warnings)
    _crop(run_dir, None)  # removing an absent crop is no change
    assert load_recipe(RunPaths(run_dir)).image.geometry == ImageGeometry(turn=90, angle=1.0)


def test_a_new_frame_spreads_the_lanes_evenly_over_its_width(gel_image_path, run_dir):
    path, *_ = gel_image_path
    _init(path, run_dir, labels=["a", "b", "c", "d"])
    summary = _rotate(run_dir, turn=90)  # the width becomes 250 px
    lanes = load_recipe(RunPaths(run_dir)).lanes
    assert [lane.x_center_px for lane in lanes.lanes] == pytest.approx([12.5, 87.5, 162.5, 237.5])
    assert lanes.labels() == ["a", "b", "c", "d"]
    assert any("lane centres were spread evenly" in warning for warning in summary.warnings)


def test_image_set_keeps_the_geometry(gel_image_path, run_dir):
    path, *_ = gel_image_path
    _init(path, run_dir)
    _rotate(run_dir, turn=90, angle=-3.0)
    _crop(run_dir, (4, 4, 200, 250))
    commands.set_image(SetImageRequest(run_dir=run_dir, invert=True))
    meta = load_recipe(RunPaths(run_dir)).image
    assert meta.inverted and meta.geometry == ImageGeometry(turn=90, angle=-3.0, crop=(4, 4, 200, 250))


def test_an_out_of_range_crop_is_refused_and_changes_nothing(gel_image_path, run_dir):
    path, *_ = gel_image_path
    _init(path, run_dir)
    recipe = Path(run_dir) / "recipe.json"
    before = recipe.read_bytes()
    with pytest.raises(InvalidValueError) as caught:
        _crop(run_dir, (0, 0, 301, 250))
    assert "x1 = 301 is beyond the width 300" in str(caught.value)
    assert "300 x 250 px" in str(caught.value)
    assert recipe.read_bytes() == before


# --- the repository's example gels ----------------------------------------------------------
def _write_copy(raw: np.ndarray, path: Path) -> Path:
    tifffile.imwrite(path, raw)
    return path


def _detect(run_dir, n_lanes):
    return commands.detect_lanes(DetectLanesRequest(run_dir=run_dir, n_lanes=n_lanes, apply=True))


def _band_positions(profile: np.ndarray, prominence: float) -> np.ndarray:
    """Peak positions of a profile, prominence given as a fraction of its range."""
    peaks, _ = find_peaks(profile, prominence=prominence * float(np.ptp(profile)), distance=8)
    return peaks


@pytest.mark.skipif(not (EXAMPLES / "example_gel_1.tif").exists(), reason="example gel not available")
def test_a_turned_mirrored_padded_copy_is_restored_exactly(tmp_path):
    original = EXAMPLES / "example_gel_1.tif"
    raw = tifffile.imread(original)
    margin = 37
    # the copy is the inverse of turn=90 + flip, padded with a dark margin: gelpy must undo it
    turned = np.rot90(raw[:, ::-1], -1)
    copy = _write_copy(np.pad(turned, margin, constant_values=int(raw.min())), tmp_path / "copy.tif")

    run_a, run_b = str(tmp_path / "a"), str(tmp_path / "b")
    _init(original, run_a, n_lanes=20)
    inverted = load_recipe(RunPaths(run_a)).image.inverted
    _init(copy, run_b, n_lanes=20, invert=inverted)
    _rotate(run_b, turn=90, flip=True)
    height, width = raw.shape
    _crop(run_b, (margin, margin, margin + width, margin + height))

    assert np.array_equal(_pipeline(run_a).ensure_image(), _pipeline(run_b).ensure_image())
    lanes_a, lanes_b = _detect(run_a, 20), _detect(run_b, 20)
    assert lanes_b.centers == lanes_a.centers  # count, order and positions: identical
    pipe_a, pipe_b = _pipeline(run_a), _pipeline(run_b)
    for lane_a, lane_b in zip(pipe_a.recipe.lanes.lanes, pipe_b.recipe.lanes.lanes):
        assert np.array_equal(pipe_a.ensure_profile(lane_a), pipe_b.ensure_profile(lane_b))


@pytest.mark.skipif(not (EXAMPLES / "example_gel_2.tif").exists(), reason="example gel not available")
def test_a_tilted_copy_is_straightened_within_tolerance(tmp_path):
    original = EXAMPLES / "example_gel_2.tif"
    raw = tifffile.imread(original)
    height, width = raw.shape
    margin, tilt = 60, 2.0
    # tilted clockwise by 2 degrees about its centre by scipy (an independent implementation)
    padded = np.pad(raw.astype(np.float64), margin, constant_values=float(raw.min()))
    tilted = ndimage.rotate(padded, -tilt, reshape=False, order=1, mode="constant", cval=float(raw.min()))
    copy = _write_copy(np.rint(tilted).astype(raw.dtype), tmp_path / "tilted.tif")

    run_a, run_b = str(tmp_path / "a"), str(tmp_path / "b")
    _init(original, run_a, n_lanes=11)
    inverted = load_recipe(RunPaths(run_a)).image.inverted
    _init(copy, run_b, n_lanes=11, invert=inverted)
    _rotate(run_b, angle=tilt)  # positive: counter-clockwise, undoing the clockwise tilt
    canvas = rotated_shape(*padded.shape, tilt)
    top = (canvas[0] - padded.shape[0]) // 2 + margin
    left = (canvas[1] - padded.shape[1]) // 2 + margin
    _crop(run_b, (left, top, left + width, top + height))

    lanes_a, lanes_b = _detect(run_a, 11), _detect(run_b, 11)
    assert len(lanes_b.centers) == len(lanes_a.centers) == 11
    assert np.all(np.diff(lanes_b.centers) > 0)  # left to right, the same order
    assert np.abs(np.subtract(lanes_b.centers, lanes_a.centers)).max() < 1.0

    # the same strips on both (run a's centres), so the profiles compare like for like: every
    # clear band of the original has a band of the straightened copy within 1 px
    pipe_a, pipe_b = _pipeline(run_a), _pipeline(run_b)
    compared = 0
    for lane_a in pipe_a.recipe.lanes.lanes:
        lane_b = pipe_b.recipe.lanes.lanes[lane_a.index].model_copy(update={"x_center_px": lane_a.x_center_px})
        bands_a = _band_positions(pipe_a.ensure_profile(lane_a), 0.2)
        bands_b = _band_positions(pipe_b.ensure_profile(lane_b), 0.05)
        for band in bands_a:
            assert np.abs(bands_b - band).min() <= 1, (lane_a.index, band, bands_b)
        compared += len(bands_a)
    assert compared >= 11
