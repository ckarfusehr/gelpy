"""Packaging invariants: one version, one skill, no accidental sdist payload.

These are cheap guards against the class of mistake that is invisible until someone else
installs the package.
"""

from __future__ import annotations

import importlib.metadata
import json
import tomllib
from pathlib import Path

import pytest

import gelpy

REPO = Path(__file__).resolve().parents[1]


def _pyproject() -> dict:
    return tomllib.loads((REPO / "pyproject.toml").read_bytes().decode())


def test_version_is_single_sourced():
    """The installed distribution's version must come from ``gelpy.__version__``.

    Two hand-maintained copies of a version number diverge; hatch reads it out of
    ``__init__.py`` so there is exactly one place to change.
    """
    assert importlib.metadata.version("gelpy") == gelpy.__version__


def test_pyproject_declares_the_version_dynamically():
    project = _pyproject()["project"]
    assert "version" not in project, "version must not be duplicated in pyproject.toml"
    assert project.get("dynamic") == ["version"]


def test_version_is_a_released_major():
    """1.0 is deliberately incompatible with the published 0.0.4 notebook API."""
    major = int(gelpy.__version__.split(".")[0])
    assert major >= 1, "the agent-native rewrite is 1.x; 0.x was the pre-rewrite Gel API"


def test_changelog_documents_the_current_version():
    changelog = (REPO / "CHANGELOG.md").read_text()
    assert f"## [{gelpy.__version__}]" in changelog, (
        f"CHANGELOG.md has no entry for {gelpy.__version__}"
    )


def test_sdist_does_not_ship_the_example_tiffs():
    """The TIFFs are ~17 MB and exist for the test suite, not for every `pip install`.

    hatchling does not read MANIFEST.in; the sdist rule lives in pyproject.toml.
    """
    sdist = _pyproject()["tool"]["hatch"]["build"]["targets"]["sdist"]
    assert "/examples/data" in sdist["exclude"]


@pytest.mark.parametrize("pattern", ["*.csv", "*.png", "*.jpg", "*.svg"])
def test_gitignore_has_no_blanket_asset_ignore(pattern):
    """A blanket ignore silently swallows committed content.

    The generated walkthrough commits PNGs and a results CSV; a repo-wide `*.csv` (which
    this project had) makes `git add` quietly do nothing, which is very hard to notice.
    """
    lines = {
        line.strip()
        for line in (REPO / ".gitignore").read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    assert pattern not in lines


def test_exactly_one_copy_of_the_analysis_skill():
    """Two copies of SKILL.md in one repo is a drift vector; keep exactly one."""
    copies = sorted(
        p for p in REPO.rglob("SKILL.md") if ".git" not in p.parts and "node_modules" not in p.parts
    )
    assert len(copies) == 1, f"expected one SKILL.md, found: {[str(p) for p in copies]}"


def test_plugin_manifest_matches_the_package():
    """The plugin is the delivery vehicle for the skill; its identity must not drift."""
    manifest = json.loads((REPO / "plugin" / ".claude-plugin" / "plugin.json").read_text())
    assert manifest["name"] == "gelpy"
    assert manifest["description"].strip()
    # No `version` on purpose: a versioned plugin is only re-delivered to users when
    # that field is bumped, so the skill would silently stop updating on push. Without it the
    # plugin resolves to the commit and tracks the branch.
    assert "version" not in manifest


def test_marketplace_entry_matches_the_plugin_manifest():
    """`plugin/marketplace-entry.json` is a paste-me snippet; keep it honest."""
    manifest = json.loads((REPO / "plugin" / ".claude-plugin" / "plugin.json").read_text())
    entry = json.loads((REPO / "plugin" / "marketplace-entry.json").read_text())
    for form in ("vendored", "referenced"):
        assert entry[form]["name"] == manifest["name"]
        assert entry[form]["description"] == manifest["description"]
