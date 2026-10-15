"""Text files are UTF-8 whatever the machine's default encoding is.

``tests/test_source_rules.py`` reads the code for a text file opened without an encoding. These tests
run the real thing in a process whose default encoding is ASCII, so they fail on any machine, UTF-8 or
not, if one slips through."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gelpy.cli.app import app

LABELS = ["Größe", "µ-Probe", "Ünïcödé", "日本"]
HERE = Path(__file__).resolve()
PORTABILITY = HERE.parents[1] / "runfolder" / "test_portability.py"
ROUND_TRIPS = (
    f"{PORTABILITY}::test_non_ascii_labels_round_trip_through_every_file",
    f"{HERE}::test_non_ascii_labels_round_trip_through_the_cli",
)


def test_non_ascii_labels_round_trip_through_the_cli(gel_image, run_dir, tmp_path):
    labels_file = tmp_path / "labels.txt"
    labels_file.write_bytes("\n".join(LABELS).encode("utf-8"))
    runner = CliRunner()
    labels = str(labels_file)
    steps = (
        ("init", run_dir, gel_image, "--n-lanes", "4", "--width", "30", "--labels-file", labels),
        ("lanes", "set", run_dir, "--linspace", "40", "250", "4", "--labels-file", labels),
        ("background", "remove", run_dir, "--model", "plane_ransac"),
        ("fit", "run", run_dir, "--lane", "0", "--lane", "1"),
        ("results", "show", run_dir, "--format", "csv"),
    )
    for step in steps:
        result = runner.invoke(app, list(step))
        assert result.exit_code == 0, f"{step}\n{result.output}"
    run = Path(run_dir)
    recipe = json.loads((run / "recipe.json").read_text(encoding="utf-8"))
    assert [lane["label"] for lane in recipe["lanes"]["lanes"]] == LABELS
    assert "Größe" in (run / "results.csv").read_text(encoding="utf-8")
    assert "Größe" in (run / "report.md").read_text(encoding="utf-8")
    assert "Größe" in result.output  # what the CLI printed


def test_the_round_trips_hold_in_a_process_whose_default_encoding_is_ascii():
    env = {key: value for key, value in os.environ.items() if key != "LC_CTYPE"}
    env.update(PYTHONUTF8="0", LC_ALL="C", LANG="C", PYTHONCOERCECLOCALE="0")
    default = subprocess.run(
        [sys.executable, "-c", "import locale; print(locale.getpreferredencoding(False))"],
        env=env, capture_output=True, text=True,
    ).stdout.strip()
    if "utf" in default.lower():
        pytest.skip(f"this machine cannot be given a default text encoding other than {default}")
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *ROUND_TRIPS],
        env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    assert done.returncode == 0, (
        f"under a {default} default encoding:\n{done.stdout[-3000:]}{done.stderr[-1000:]}"
    )
