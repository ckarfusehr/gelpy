"""End-to-end replay of the real notebook workflow via the CLI.

This is the primary verification: reproduce the broad -> refit -> tight sequence
against the actual example_gel_1.tif and confirm the targeted refit recomputes
exactly one fit while leaving the image/background caches byte-for-byte untouched.
"""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gelpy.cli.app import app

runner = CliRunner()
IMAGE = Path(__file__).resolve().parents[2] / "examples" / "data" / "example_gel_1.tif"


def _run(*args):
    result = runner.invoke(app, list(args))
    assert result.exit_code == 0, f"args={args}\n{result.output}"
    return result


@pytest.mark.skipif(not IMAGE.exists(), reason="real gel image not available")
def test_real_workflow_replay(tmp_path):
    run_dir = str(tmp_path / "run")

    _run("init", run_dir, str(IMAGE), "--n-lanes", "20", "--width", "180")
    _run("lanes", "set", run_dir, "--linspace", "150", "4485", "20", "--width", "180",
         "--label-rotation", "90")
    _run("background", "remove", run_dir, "--model", "plane_ransac")

    # broad pass over all lanes (grow_select needs no manual peak-detection tuning)
    out = _run("fit", "run", run_dir, "--lane", "re:.*", "--slice", "50", "600",
               "--method", "grow_select")
    summary = json.loads(out.output)
    # cold run: image + background are each computed exactly once (not per-lane),
    # alongside one profile and one fit per lane.
    assert "image" in summary["recomputed"]
    assert "background" in summary["recomputed"]
    assert len([k for k in summary["recomputed"] if k.startswith("fit:")]) == 20
    assert len([k for k in summary["recomputed"] if k.startswith("profile:")]) == 20

    # snapshot upstream cache mtimes
    def cache_mtimes(sub):
        return {p.name: p.stat().st_mtime_ns for p in (Path(run_dir) / "cache" / sub).glob("*")}

    image_before = cache_mtimes("image")
    bg_before = cache_mtimes("background")

    # targeted refit of a single lane (a seed change forces a new fit_hash for that lane only)
    out = _run("fit", "refit-lane", run_dir, "--lane", "12", "--seed", "1")
    summary = json.loads(out.output)
    assert summary["recomputed"] == ["fit:lane=12"]
    assert set(summary["reused"]) >= {"image", "background", "profile:lane=12"}

    # image + background caches untouched by the refit
    assert cache_mtimes("image") == image_before
    assert cache_mtimes("background") == bg_before

    # tight final pass, then an identical rerun that must recompute nothing
    tight = ("fit", "run", run_dir, "--lane", "re:.*", "--slice", "60", "600",
             "--method", "grow_select", "--no-plot")
    _run(*tight)
    out = _run(*tight)
    assert json.loads(out.output)["recomputed"] == []

    # browsable-without-python artifacts exist
    assert (Path(run_dir) / "results.csv").exists()
    assert (Path(run_dir) / "results.json").exists()
    assert (Path(run_dir) / "report.md").exists()
    header = (Path(run_dir) / "results.csv").read_text().splitlines()[0]
    for col in ["lane_index", "label", "band_number", "component_class", "area_fraction", "maximum_position"]:
        assert col in header

    # results read back from cache without recompute
    result = json.loads(_run("results", "show", run_dir).output)
    assert len(result["lanes"]) == 20
