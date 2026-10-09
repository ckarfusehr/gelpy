"""The distributable plugin must stay consistent with the package and the marketplace snippet."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PLUGIN = REPO / "plugin"
MANIFEST = PLUGIN / ".claude-plugin" / "plugin.json"
SKILL = PLUGIN / "skills" / "gelpy-analyze" / "SKILL.md"
EXAMPLES = PLUGIN / "skills" / "gelpy-analyze" / "examples"


def test_plugin_layout_is_canonical():
    """`.claude-plugin/plugin.json` at the root, skills beside it -- not inside it."""
    assert MANIFEST.exists()
    assert SKILL.exists()
    assert not (PLUGIN / ".claude-plugin" / "skills").exists(), (
        "component directories must sit next to .claude-plugin/, never inside it"
    )


def test_manifest_is_valid_json_with_a_kebab_case_name():
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["name"] == "gelpy"
    assert manifest["name"] == manifest["name"].lower().replace("_", "-")


def test_skill_has_frontmatter_with_name_and_description():
    """Claude Code reads these two fields to decide whether the skill applies."""
    text = SKILL.read_text()
    assert text.startswith("---\n"), "SKILL.md needs YAML front matter"
    head = text.split("---")[1]
    assert "name:" in head and "description:" in head


def test_skill_description_says_when_to_use_it():
    """A description that does not say *when* to use the skill will not be selected."""
    text = SKILL.read_text().split("---")[1].lower()
    assert "use when" in text or "use this" in text


def test_sync_plugin_check_runs_and_reports_cleanly_against_a_fresh_target(tmp_path):
    """`--check` must exit non-zero on a marketplace that has never received the plugin."""
    marketplace = tmp_path / "marketplace"
    (marketplace / ".claude-plugin").mkdir(parents=True)
    (marketplace / ".claude-plugin" / "marketplace.json").write_text(
        json.dumps({"name": "test", "owner": {"name": "t"}, "plugins": []}, indent=2)
    )
    result = subprocess.run(
        [sys.executable, str(REPO / "tools" / "sync_plugin.py"), "--to", str(marketplace), "--check"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "OUT OF SYNC" in result.stdout


def test_sync_plugin_vendors_the_plugin_and_upserts_the_entry(tmp_path):
    """The end-to-end delivery step: copy the plugin in, register it, and be idempotent."""
    marketplace = tmp_path / "marketplace"
    (marketplace / ".claude-plugin").mkdir(parents=True)
    config = marketplace / ".claude-plugin" / "marketplace.json"
    config.write_text(json.dumps({"name": "test", "owner": {"name": "t"}, "plugins": []}, indent=2))

    def sync(*extra: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(REPO / "tools" / "sync_plugin.py"), "--to", str(marketplace), *extra],
            capture_output=True,
            text=True,
        )

    assert sync().returncode == 0

    vendored = marketplace / "plugins" / "gelpy"
    assert (vendored / "skills" / "gelpy-analyze" / "SKILL.md").read_text() == SKILL.read_text()
    assert (vendored / ".claude-plugin" / "plugin.json").exists()
    # The paste-me snippet is a gelpy-side aid and must not be delivered.
    assert not (vendored / "marketplace-entry.json").exists()

    entries = json.loads(config.read_text())["plugins"]
    assert len(entries) == 1
    assert entries[0]["name"] == "gelpy"
    assert entries[0]["source"] == "./plugins/gelpy"

    # Idempotent: a second sync changes nothing, and --check now passes.
    assert sync().returncode == 0
    assert json.loads(config.read_text())["plugins"] == entries
    assert sync("--check").returncode == 0


def test_sync_plugin_does_not_duplicate_an_existing_entry(tmp_path):
    marketplace = tmp_path / "marketplace"
    (marketplace / ".claude-plugin").mkdir(parents=True)
    config = marketplace / ".claude-plugin" / "marketplace.json"
    config.write_text(
        json.dumps(
            {
                "name": "test",
                "owner": {"name": "t"},
                "plugins": [
                    {"name": "other", "source": "./plugins/other"},
                    {"name": "gelpy", "source": "./plugins/gelpy", "description": "stale"},
                ],
            },
            indent=2,
        )
    )
    subprocess.run(
        [sys.executable, str(REPO / "tools" / "sync_plugin.py"), "--to", str(marketplace)],
        capture_output=True,
        text=True,
        check=True,
    )
    entries = json.loads(config.read_text())["plugins"]
    names = [e["name"] for e in entries]
    assert names.count("gelpy") == 1, "the existing entry must be updated, not duplicated"
    assert "other" in names, "unrelated entries must be left alone"
    gelpy_entry = next(e for e in entries if e["name"] == "gelpy")
    assert gelpy_entry["description"] != "stale", "the stale description must be refreshed"


@pytest.mark.parametrize("script", sorted(EXAMPLES.glob("*.py")), ids=lambda p: p.name)
def test_skill_example_scripts_compile(script):
    """The skill points agents at these scripts; a syntax error would only surface in use."""
    compile(script.read_text(), str(script), "exec")


def test_component_fit_template_prints_its_help():
    """--help runs the imports and the argument parser without needing a run folder."""
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / "constrained_component_fit.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "--dip-window" in result.stdout
