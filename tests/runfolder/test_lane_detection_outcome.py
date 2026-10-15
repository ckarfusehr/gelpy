"""``run_lane_detection`` returns the detection together with what its overlay is drawn from;
``detect_lanes`` stays the plain result of the same call."""

import dataclasses

import numpy as np
import pytest

from gelpy.runfolder import commands
from gelpy.schemas.requests import DetectLanesRequest, InitRunRequest, RemoveBackgroundRequest


@pytest.fixture
def run(gel_image_path, run_dir):
    commands.init_run(
        InitRunRequest(run_dir=run_dir, image_path=gel_image_path[0], n_lanes=4, width=30)
    )
    return run_dir


def test_the_outcome_carries_the_result_the_image_and_the_detection(run, gel_image_path):
    _, _, width, height = gel_image_path
    outcome = commands.run_lane_detection(DetectLanesRequest(run_dir=run, n_lanes=4))

    assert isinstance(outcome, commands.LaneDetectionOutcome)
    assert outcome.image.shape == (height, width)
    assert outcome.detection.projection.shape == (width,)  # one value per image column
    assert outcome.result.centers == outcome.detection.centers  # the result is the detection's
    assert outcome.result.applied is False
    with pytest.raises(dataclasses.FrozenInstanceError):
        outcome.result = None


def test_detect_lanes_is_the_result_of_run_lane_detection(run):
    request = DetectLanesRequest(run_dir=run, n_lanes=4)
    assert commands.detect_lanes(request) == commands.run_lane_detection(request).result


def test_the_outcome_carries_the_row_bounds_the_detection_was_given(run):
    whole = commands.run_lane_detection(DetectLanesRequest(run_dir=run, n_lanes=4))
    assert (whole.y_start, whole.y_end) == (None, None)
    band = commands.run_lane_detection(
        DetectLanesRequest(run_dir=run, n_lanes=4, y_start=30, y_end=200)
    )
    assert (band.y_start, band.y_end) == (30, 200)


def test_an_outcome_compares_and_hashes_by_identity(run):
    """Its fields are arrays: a generated __eq__ raises on two different outcomes, a generated
    __hash__ on any."""
    request = DetectLanesRequest(run_dir=run, n_lanes=4)
    one, other = commands.run_lane_detection(request), commands.run_lane_detection(request)
    assert one == one and one != other
    assert len({one, other}) == 2


def test_the_outcome_carries_the_image_the_detection_ran_on(run):
    commands.remove_background(RemoveBackgroundRequest(run_dir=run, model="plane_ransac"))
    raw = commands.run_lane_detection(DetectLanesRequest(run_dir=run, n_lanes=4)).image
    corrected = commands.run_lane_detection(
        DetectLanesRequest(run_dir=run, n_lanes=4, use_corrected=True)
    ).image
    assert raw.shape == corrected.shape
    assert not np.array_equal(raw, corrected)  # use_corrected picks the image that is carried
