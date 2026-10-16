"""Shared fixtures for the documentation verification suite."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
EXAMPLE_IMAGE = REPO / "examples" / "data" / "example_gel_1.tif"

sys.path.insert(0, str(REPO))


@pytest.fixture(scope="session")
def repo() -> Path:
    return REPO


@pytest.fixture(scope="session")
def example_image() -> Path:
    if not EXAMPLE_IMAGE.exists():
        pytest.skip(f"{EXAMPLE_IMAGE.relative_to(REPO)} is absent")
    return EXAMPLE_IMAGE


@pytest.fixture(scope="session")
def gelpy_cli() -> list[str]:
    """How to invoke the CLI under the interpreter running the tests.

    ``python -m gelpy`` rather than the ``gelpy`` entry point, so the tests exercise the
    checkout being tested even when a different gelpy is first on PATH.
    """
    return [sys.executable, "-m", "gelpy"]


def run_cli(argv: list[str], cwd: Path) -> subprocess.CompletedProcess:
    """Run one gelpy invocation, returning the completed process (never raising)."""
    return subprocess.run(
        [sys.executable, "-m", "gelpy", *argv],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=900,
    )
