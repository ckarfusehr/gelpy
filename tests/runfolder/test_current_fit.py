"""Which fit of a lane is current: the one the recipe, the stored configuration and the gelpy version
call for, when it is in the cache.

The cache keeps every fit it computed, keyed by what it was computed from, so an edit that is undone
makes an older fit current again. The ledger only says which artifact a command last recorded, so
``status``, ``results show``, the plots and the report judge a fit by its expected artifact and leave
the ledger's record out of it, the way ``stale_stages`` always did. A fit file that does not validate
counts as missing, as it does for ``status``, and an error about a cache file names the file and never
what is in it.
"""

import json

import pytest

import gelpy
from gelpy.cli import _render
from gelpy.core.errors import InvalidValueError, to_error_response
from gelpy.runfolder import commands
from gelpy.runfolder.layout import RunPaths
from gelpy.runfolder.manifest import load_manifest
from gelpy.runfolder import results as results_module
from gelpy.schemas.fit_config import FitConfig
from gelpy.schemas.requests import (
    FitRunRequest,
    GetResultsRequest,
    InitRunRequest,
    RemoveBackgroundRequest,
    SetLanesRequest,
    StatusRequest,
)

SECRET = "SECRET-CONTENT-OF-ANOTHER-FILE"


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


def _results(run):
    return commands.get_results(GetResultsRequest(run_dir=run))


def _fit_file(run, lane=0):
    paths = RunPaths(run)
    return paths.resolve(load_manifest(paths).stages[f"fit:lane={lane}"].cache_path)


def _edit_manifest(run, change):
    paths = RunPaths(run)
    data = json.loads(paths.manifest.read_text(encoding="utf-8"))
    change(data)
    paths.manifest.write_text(json.dumps(data), encoding="utf-8")


# --- an edit that is undone makes the fits current again, without a fit run -----------------------------
def test_results_are_back_right_after_an_edit_is_undone_without_a_fit_run(fitted_run):
    first = _results(fitted_run)
    assert [lane.lane_index for lane in first.lanes] == [0, 1, 2, 3]

    commands.set_lanes(SetLanesRequest(run_dir=fitted_run, width=40))
    assert _results(fitted_run).lanes == []  # stale: nothing is claimed that no longer reflects the recipe
    _fit_all(fitted_run)  # the ledger now names the width-40 fits
    commands.set_lanes(SetLanesRequest(run_dir=fitted_run, width=30))  # undone: no fit run

    again = _results(fitted_run)

    assert [lane.lane_index for lane in again.lanes] == [0, 1, 2, 3]
    assert [lane.model_dump(mode="json") for lane in again.lanes] == [
        lane.model_dump(mode="json") for lane in first.lanes
    ]  # the very fits of the first pass


def test_status_agrees_with_stale_stages_right_after_an_edit_is_undone(fitted_run):
    commands.set_lanes(SetLanesRequest(run_dir=fitted_run, width=40))
    _fit_all(fitted_run)
    commands.set_lanes(SetLanesRequest(run_dir=fitted_run, width=30))

    status = _status(fitted_run)

    assert status.stale_stages == []
    assert (status.n_fit, status.n_stale) == (4, 0)
    assert all(lane.has_fit and lane.fit_current and lane.n_bands for lane in status.lanes)


def test_the_page_and_the_plots_follow_the_same_fits_after_an_edit_is_undone(fitted_run):
    commands.set_lanes(SetLanesRequest(run_dir=fitted_run, width=40))
    _fit_all(fitted_run)
    commands.set_lanes(SetLanesRequest(run_dir=fitted_run, width=30))

    built = _render.write_html_report(fitted_run)

    assert {f"img/fits/lane_{i:03d}.png" for i in range(4)} <= set(built.files)
    assert _status(fitted_run).report.current is True


def test_an_edit_that_is_not_undone_still_makes_the_fits_stale(fitted_run):
    commands.set_lanes(SetLanesRequest(run_dir=fitted_run, width=40))
    status = _status(fitted_run)
    assert (status.n_fit, status.n_stale) == (0, 4)
    assert not any(lane.fit_current for lane in status.lanes)
    assert _results(fitted_run).lanes == []


def test_another_gelpy_version_makes_the_fits_stale_and_going_back_makes_them_current(fitted_run, monkeypatch):
    version = gelpy.__version__
    monkeypatch.setattr(gelpy, "__version__", "99.0.0")
    assert (_status(fitted_run).n_fit, _results(fitted_run).lanes) == (0, [])
    _fit_all(fitted_run)
    assert _status(fitted_run).n_fit == 4
    monkeypatch.setattr(gelpy, "__version__", version)
    assert (_status(fitted_run).n_fit, _status(fitted_run).n_stale) == (4, 0)
    assert len(_results(fitted_run).lanes) == 4


def test_a_lane_whose_expected_fit_is_not_cached_stays_stale_while_the_others_are_current(fitted_run):
    commands.set_lanes(SetLanesRequest(run_dir=fitted_run, x_centers=[40.0, 110.0, 180.0, 249.0]))
    status = _status(fitted_run)
    assert [lane.fit_current for lane in status.lanes] == [True, True, True, False]
    assert [lane.lane_index for lane in _results(fitted_run).lanes] == [0, 1, 2]


def test_a_wiped_ledger_is_still_a_ledger_to_rebuild(fitted_run):
    """Only a lane the ledger records has a fit to judge: after a wipe nothing is current until a
    command records the lane again (the documented behaviour of a disposable ledger)."""
    RunPaths(fitted_run).manifest.unlink()
    assert (_status(fitted_run).n_fit, _results(fitted_run).lanes) == (0, [])
    assert _fit_all(fitted_run).recomputed == []
    assert _status(fitted_run).n_fit == 4


# --- a fit file that does not validate counts as missing, in results as in status ---------------------
def test_results_leave_out_a_damaged_current_fit_the_way_status_does(fitted_run):
    _fit_file(fitted_run, lane=0).write_text('{"lane_index": 0, "label"', encoding="utf-8")

    results = _results(fitted_run)  # must not raise

    assert [lane.lane_index for lane in results.lanes] == [1, 2, 3]
    status = _status(fitted_run)
    assert (status.n_fit, status.n_stale) == (3, 1)  # the same lanes: results and status agree
    assert _fit_all(fitted_run).recomputed == ["fit:lane=0"]  # and the next fit mends it
    assert [lane.lane_index for lane in _results(fitted_run).lanes] == [0, 1, 2, 3]


def test_a_lane_left_out_for_a_damaged_fit_is_named_with_the_command_that_mends_it(fitted_run):
    """The omission is never silent: `results show` says which lane and what to run next."""
    from typer.testing import CliRunner

    from gelpy.cli.app import app

    assert _results(fitted_run).warnings == []
    _fit_file(fitted_run, lane=2).write_text("{", encoding="utf-8")

    (warning,) = _results(fitted_run).warnings
    assert "Lane 2" in warning and "fit refit-lane" in warning and "--lane 2" in warning
    for fmt, in_stdout in (("json", True), ("csv", False), ("table", False)):
        shown = CliRunner().invoke(app, ["results", "show", fitted_run, "--format", fmt])
        assert shown.exit_code == 0
        assert ("refit-lane" in shown.stdout) is in_stdout
        assert ("refit-lane" in shown.stderr) is (not in_stdout)


def test_a_fit_file_that_is_not_utf8_is_left_out_too(fitted_run):
    _fit_file(fitted_run, lane=1).write_bytes(b'{"label": "Gr\xf6\xdfe"}')
    assert [lane.lane_index for lane in _results(fitted_run).lanes] == [0, 2, 3]


# --- a fit record without a lane is a structured error ---------------------------------------------------
@pytest.mark.parametrize("key", ["fit", "fit:lane", "fit:lane=x"], ids=["no lane", "no number", "not a number"])
def test_a_fit_record_that_names_no_lane_is_a_structured_error_not_an_index_error(fitted_run, key):
    def add(data):
        data["stages"][key] = {"stage": "fit", "key": key, "hash": "x", "cache_path": "elsewhere"}

    _edit_manifest(fitted_run, add)

    with pytest.raises(InvalidValueError) as raised:
        _results(fitted_run)

    response = to_error_response(raised.value)
    assert response.code == "invalid_value"
    assert "manifest.json" in response.error and repr(key) in response.error  # which record
    assert "manifest.json" in response.detail and "fit run" in response.detail  # the step that gets the lanes back
    assert _status(fitted_run).n_lanes == 4  # status has always ignored such a record


# --- an error about a cache file names the file and never what is in it --------------------------------
def test_reading_a_fit_file_that_does_not_validate_names_the_file_and_not_its_content(tmp_path):
    foreign = tmp_path / "notes.txt"
    foreign.write_text(
        json.dumps({"lane_index": SECRET, "bands": SECRET, SECRET + "-KEY": SECRET}), encoding="utf-8"
    )

    with pytest.raises(InvalidValueError) as raised:
        results_module.read_fit_result(foreign)

    response = to_error_response(raised.value)
    assert response.code == "invalid_value"
    assert "notes.txt" in response.error  # the file
    assert any(word in response.error for word in ("extra_forbidden", "lane_index", "bands"))  # and why
    assert "SECRET" not in response.error and "SECRET" not in (response.detail or "")  # no value, no key
    assert "fit" in response.detail  # the next step: run the fit again


def test_a_file_that_is_not_json_or_not_utf8_is_named_too(tmp_path):
    for content in (b"not json at all " + SECRET.encode(), b"\xff\xfe" + SECRET.encode()):
        path = tmp_path / "fit.json"
        path.write_bytes(content)
        with pytest.raises(InvalidValueError) as raised:
            results_module.read_fit_result(path)
        assert "fit.json" in str(raised.value) and SECRET not in str(raised.value)


def test_a_record_that_points_at_another_file_never_echoes_it(fitted_run, tmp_path):
    """A manifest is a plain file: its cache_path may name any file the process can read."""
    foreign = tmp_path / "private-notes.json"
    foreign.write_text(json.dumps({"lane_index": 0, "label": SECRET, "bands": SECRET}), encoding="utf-8")
    real = _fit_file(fitted_run, lane=0)
    real.unlink()  # the expected fit is not cached, so nothing can heal the record
    _edit_manifest(fitted_run, lambda data: data["stages"]["fit:lane=0"].update(cache_path=str(foreign)))

    # results and status do not read a fit that is not current at all
    assert [lane.lane_index for lane in _results(fitted_run).lanes] == [1, 2, 3]
    # a renderer that does read the recorded file reports it by name
    with pytest.raises(InvalidValueError) as raised:
        _render.render_fits(fitted_run, [0])
    response = to_error_response(raised.value)
    assert "private-notes.json" in response.error
    assert SECRET not in response.error and SECRET not in (response.detail or "")


def test_a_record_that_points_at_another_file_is_healed_when_the_expected_fit_is_cached(fitted_run, tmp_path):
    foreign = tmp_path / "private-notes.json"
    foreign.write_text(SECRET, encoding="utf-8")
    _edit_manifest(fitted_run, lambda data: data["stages"]["fit:lane=0"].update(cache_path=str(foreign)))

    assert [lane.lane_index for lane in _results(fitted_run).lanes] == [0, 1, 2, 3]
    assert _status(fitted_run).n_fit == 4
    plots = _render.render_fits(fitted_run, [0])  # reads the expected file, not the recorded path
    assert [p.name for p in plots] == ["lane_000.png"]
