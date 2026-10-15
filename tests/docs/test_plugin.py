"""The distributable plugin must stay consistent with the package and the marketplace snippet."""

from __future__ import annotations

import json
import re
import shlex
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


JUDGMENT = PLUGIN / "skills" / "gelpy-analyze" / "judgment.md"


def _cli():
    import typer.main

    from gelpy.cli.app import app

    return typer.main.get_command(app)


def _known_flags() -> set[str]:
    known: set[str] = {"--help"}

    def collect(command) -> None:
        for param in command.params:
            known.update(param.opts)
            known.update(param.secondary_opts)
        for child in getattr(command, "commands", {}).values():
            collect(child)

    collect(_cli())
    return known


def _command_lines(text: str) -> list[list[str]]:
    """Every `gelpy ...` command in inline code or a fenced block, placeholders left out."""
    spans = re.findall(r"`(gelpy [^`\n]+)`", text)
    fenced = [
        line.strip()
        for block in re.findall(r"```[a-z]*\n(.*?)```", text, re.S)
        for line in block.splitlines()
        if line.strip().startswith("gelpy ")
    ]
    argvs = [shlex.split(command.split("#")[0]) for command in spans + fenced]
    return [argv for argv in argvs if len(argv) > 1 and not argv[1].startswith(("<", "-"))]


def test_the_skill_points_to_its_judgment_file():
    assert JUDGMENT.exists()
    assert "judgment.md" in SKILL.read_text(encoding="utf-8")


def test_the_judgment_file_is_tool_agnostic():
    """judgment.md applies however gelpy is driven: no command lines, no flags, no tool names."""
    from gelpy.cli.tools import TOOLS

    text = JUDGMENT.read_text(encoding="utf-8")
    assert not re.findall(r"`gelpy [^`]*`", text), "a command line"
    assert not _command_lines(text), "a command line"
    assert not re.search(r"(?m)^\s*gelpy\s", text), "a command line in a block, with or without flags"
    assert not re.search(r"(?<![\w-])--[a-z]", text), "a CLI flag"
    named = sorted(name for name in TOOLS if "_" in name and re.search(rf"(?<![\w-]){name}(?![\w-])", text))
    assert not named, f"tool names (backticked or not): {named}"


@pytest.mark.parametrize(
    "text",
    ["```\ngelpy status RUN\n```", "run gelpy status\nthen `gelpy fit run`", "use lanes_detect to look", "call `fit_run`"],
    ids=["a block line without flags", "an inline command", "a bare tool name", "a backticked tool name"],
)
def test_the_tool_agnostic_rule_bites(text):
    """The checks of the test above, each on a text that breaks it."""
    from gelpy.cli.tools import TOOLS

    breaks = (
        bool(_command_lines(text))
        or bool(re.search(r"(?m)^\s*gelpy\s", text))
        or any("_" in name and re.search(rf"(?<![\w-]){name}(?![\w-])", text) for name in TOOLS)
    )
    assert breaks


def test_every_flag_the_skill_mentions_exists():
    flags = set(re.findall(r"(?<![\w-])(--[a-z0-9][a-z0-9-]*)", SKILL.read_text(encoding="utf-8")))
    assert flags <= _known_flags(), sorted(flags - _known_flags())


def test_every_command_the_skill_uses_exists_with_its_options():
    root, problems = _cli(), []
    for argv in _command_lines(SKILL.read_text(encoding="utf-8")):
        command, rest = root, argv[1:]
        while rest and getattr(command, "commands", None) and rest[0] in command.commands:
            command, rest = command.commands[rest[0]], rest[1:]
        if getattr(command, "commands", None):
            problems.append(" ".join(argv))
            continue
        known = {opt for param in command.params for opt in (*param.opts, *param.secondary_opts)}
        problems += [
            f"{' '.join(argv)}: {token}"
            for token in rest
            if token.startswith("--") and token.split("=")[0] not in known | {"--help"}
        ]
    assert not problems, problems


def test_every_contract_the_skill_names_exists():
    import gelpy.schemas

    text = SKILL.read_text(encoding="utf-8") + JUDGMENT.read_text(encoding="utf-8")
    pattern = r"\b[A-Z][A-Za-z]+(?:Request|Result|Config|Layout|Status|Overrides|Store)\b"
    named = set(re.findall(pattern, text))
    assert named <= set(gelpy.schemas.__all__), sorted(named - set(gelpy.schemas.__all__))


def _tools_in_the_skill_table() -> list[str]:
    lines = SKILL.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.replace(" ", "") == "|step|CLI|tool|")
    named = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        named += re.findall(r"`([a-z_]+)`", line.strip().strip("|").split("|")[2])
    return named


def test_every_tool_in_the_skill_table_exists():
    from gelpy.cli.tools import TOOLS

    named = _tools_in_the_skill_table()
    assert named and set(named) <= set(TOOLS), sorted(set(named) - set(TOOLS))


def test_the_skill_table_maps_every_tool():
    """A tool the table leaves out is a step an agent driving gelpy over MCP cannot find."""
    from gelpy.cli.tools import TOOLS

    assert set(TOOLS) - set(_tools_in_the_skill_table()) == set()


def test_every_judgment_section_the_skill_cites_exists():
    """SKILL.md sends the reader to judgment.md by section title; a renamed heading breaks that."""
    headings = re.findall(r"^#{1,6} (.+)$", JUDGMENT.read_text(encoding="utf-8"), re.M)
    skill = " ".join(SKILL.read_text(encoding="utf-8").split())  # a citation may wrap across lines
    cited = re.findall(r'judgment\.md, "([^"]+)"', skill)
    assert cited, "SKILL.md cites no section of judgment.md"
    missing = [title for title in cited if not any(h.startswith(title) for h in headings)]
    assert not missing, f"sections not in judgment.md: {missing}"


def test_the_skill_points_to_the_html_report_guide_page():
    assert "html-report.md" in SKILL.read_text(encoding="utf-8")
    assert (REPO / "docs" / "guide" / "html-report.md").exists()
