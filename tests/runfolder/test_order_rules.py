"""Profiles and fits need background removal unless raw is asked for; status names stale stages."""

import json

import pytest

from gelpy.core.errors import BackgroundRequiredError, LaneSelectionError
from gelpy.runfolder import commands
from gelpy.runfolder.layout import RunPaths
from gelpy.schemas.fit_config import FitConfig
from gelpy.schemas.requests import (
    ExtractProfilesRequest,
    FitRunRequest,
    InitRunRequest,
    RefitLaneRequest,
    RemoveBackgroundRequest,
    SetLanesRequest,
    StatusRequest,
)

CALLS = [
    ("profiles", commands.extract_profiles, lambda run: ExtractProfilesRequest(run_dir=run)),
    ("fit", commands.fit_run, lambda run: FitRunRequest(run_dir=run, lanes=[0])),
    ("refit", commands.refit_lane, lambda run: RefitLaneRequest(run_dir=run, lane=0)),
]


@pytest.fixture
def raw_run(gel_image_path, run_dir):
    image_path, centers, _, _ = gel_image_path
    commands.init_run(InitRunRequest(run_dir=run_dir, image_path=image_path, n_lanes=4, width=30))
    commands.set_lanes(SetLanesRequest(run_dir=run_dir, x_centers=[float(c) for c in centers]))
    return run_dir


@pytest.mark.parametrize(("name", "call", "make"), CALLS, ids=[c[0] for c in CALLS])
def test_profiles_and_fits_need_background_removal(raw_run, name, call, make):
    with pytest.raises(BackgroundRequiredError):
        call(make(raw_run))


def test_allow_raw_measures_the_raw_image_on_purpose(raw_run):
    summary = commands.fit_run(FitRunRequest(run_dir=raw_run, lanes=[0], allow_raw=True))
    assert "fit:lane=0" in summary.recomputed


@pytest.mark.parametrize(("name", "call", "make"), CALLS, ids=[c[0] for c in CALLS])
def test_allow_raw_lifts_the_refusal_of_every_command(raw_run, name, call, make):
    summary = call(make(raw_run).model_copy(update={"allow_raw": True}))
    assert any(key.startswith(("profile:", "fit:")) for key in summary.recomputed)


def test_extract_profiles_logs_allow_raw_like_the_fits_do(raw_run):
    commands.extract_profiles(ExtractProfilesRequest(run_dir=raw_run, allow_raw=True))

    last = json.loads(RunPaths(raw_run).history.read_text(encoding="utf-8").splitlines()[-1])
    assert last["command"] == "extract_profiles"
    assert last["payload"]["allow_raw"] is True


def test_a_bad_selector_is_reported_before_the_missing_background(raw_run):
    with pytest.raises(LaneSelectionError):
        commands.fit_run(FitRunRequest(run_dir=raw_run, lanes=["nonexistent"]))


def test_status_names_the_stale_stages(raw_run):
    commands.remove_background(RemoveBackgroundRequest(run_dir=raw_run, model="plane_ransac"))
    commands.fit_run(FitRunRequest(run_dir=raw_run, lanes=[0], fit=FitConfig()))
    assert commands.status(StatusRequest(run_dir=raw_run)).stale_stages == []

    commands.set_lanes(SetLanesRequest(run_dir=raw_run, width=40))
    assert commands.status(StatusRequest(run_dir=raw_run)).stale_stages == [
        "fit:lane=0",
        "profile:lane=0",
    ]

    commands.remove_background(RemoveBackgroundRequest(run_dir=raw_run, model="robust_plane"))
    assert "background" in commands.status(StatusRequest(run_dir=raw_run)).stale_stages


def test_status_names_every_stage_a_new_version_made_stale(raw_run, monkeypatch):
    import gelpy

    commands.remove_background(RemoveBackgroundRequest(run_dir=raw_run, model="plane_ransac"))
    commands.fit_run(FitRunRequest(run_dir=raw_run, lanes=[0], fit=FitConfig()))
    assert commands.status(StatusRequest(run_dir=raw_run)).stale_stages == []

    monkeypatch.setattr(gelpy, "__version__", "99.0.0")
    assert commands.status(StatusRequest(run_dir=raw_run)).stale_stages == [
        "background",
        "fit:lane=0",
        "image",
        "profile:lane=0",
    ]
