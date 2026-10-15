"""Every command block in the documentation is executed against the real example gel.

This is the mechanism that would have caught the largest failure in gelpy's previous
documentation: `examples/cli_walkthrough.sh` and the README both used five `fit` flags that
had been deleted, so the documented workflow aborted at exit code 2 -- and nothing noticed
for two months, because nothing ran it.

Conventions for a fenced block in docs/guide/ or docs/walkthrough/
------------------------------------------------------------------
- ```bash blocks are executed, in document order, sharing one run folder per page.
- Blocks reference `"$RUN"` (the run folder) and `"$IMAGE"` (the source gel). Both are set
  here and are also declared in the page, so a human copying the block sets the same two
  variables and gets the same result.
- A block that is deliberately expected to fail starts with `# doc-test: expect-failure`.
- A block that cannot run here (needs a GUI, edits a marketplace repo, is illustrative)
  starts with `# doc-test: skip` and must say in prose why.
- Use ```text or ```console for transcript output, which is never executed.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
GUIDE = REPO / "docs" / "guide"
WALKTHROUGH = REPO / "docs" / "walkthrough"
IMAGE = REPO / "examples" / "data" / "folding_screen.tif"

BLOCK = re.compile(r"^```bash[ \t]*\n(.*?)^```[ \t]*$", re.S | re.M)
SKIP = "# doc-test: skip"
EXPECT_FAILURE = "# doc-test: expect-failure"


def _pages() -> list[Path]:
    pages = []
    for directory in (GUIDE, WALKTHROUGH):
        if directory.exists():
            pages.extend(sorted(directory.rglob("*.md")))
    return [p for p in pages if BLOCK.search(p.read_text())]


PAGES = _pages()


def _blocks(text: str) -> list[str]:
    return [match.group(1) for match in BLOCK.finditer(text)]


@pytest.mark.slow
@pytest.mark.parametrize("page", PAGES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_every_command_block_executes(page, tmp_path_factory):
    """Run a page's blocks in order; every one must succeed (or fail, if it says so)."""
    if not IMAGE.exists():
        pytest.skip(f"{IMAGE.relative_to(REPO)} is absent")

    workspace = tmp_path_factory.mktemp(page.stem.replace(".", "_"))
    run_dir = workspace / "run"

    env = {
        **os.environ,
        "RUN": str(run_dir),
        "IMAGE": str(IMAGE),
        # Blocks invoke `gelpy`; make sure it is THIS checkout even if another is on PATH.
        "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}",
        "MPLBACKEND": "Agg",
        "COLUMNS": "100",
        "NO_COLOR": "1",
        "TERM": "dumb",
    }
    # CI (GitHub Actions) forces rich/typer colour on; the blocks' output must not depend on it.
    for name in ("FORCE_COLOR", "CLICOLOR_FORCE", "GITHUB_ACTIONS"):
        env.pop(name, None)
    if shutil.which("gelpy", path=env["PATH"]) is None:
        pytest.skip("the `gelpy` entry point is not installed; run `pip install -e '.[dev]'`")

    blocks = _blocks(page.read_text())
    assert blocks, f"{page.name} matched the block regex but yielded no blocks"

    def run(script: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", "-c", script], cwd=REPO, env=env, capture_output=True, text=True, timeout=3600
        )

    # A page's blocks are ONE script, concatenated in document order -- which is how a reader
    # follows it, and the only way a variable set in an early block is still set later. Running
    # them as separate shells would both diverge from the reader's experience and re-pay the
    # cost of the expensive steps once per block.
    normal: list[tuple[int, str]] = []
    failing: list[tuple[int, str]] = []
    for number, block in enumerate(blocks, start=1):
        if SKIP in block:
            continue
        (failing if EXPECT_FAILURE in block else normal).append((number, block))

    def context(label: str, script: str, result: subprocess.CompletedProcess) -> str:
        return (
            f"{page.relative_to(REPO)} {label}\n"
            f"--- script ---\n{script[-3000:]}\n"
            f"--- exit {result.returncode} ---\n"
            f"--- stdout ---\n{result.stdout[-3000:]}\n"
            f"--- stderr ---\n{result.stderr[-3000:]}"
        )

    if normal:
        numbers = ", ".join(str(n) for n, _ in normal)
        script = "set -eu\n" + "\n".join(block for _, block in normal)
        result = run(script)
        assert result.returncode == 0, (
            f"the documented sequence failed (blocks {numbers}).\n"
            f"{context(f'blocks {numbers}', script, result)}"
        )

    # An expect-failure block is checked on its own, on top of everything before it.
    for number, block in failing:
        prelude = "\n".join(b for n, b in normal if n < number)
        script = "set -u\n" + prelude + "\n" + block
        result = run(script)
        assert result.returncode != 0, (
            "block is marked expect-failure but succeeded; either the bug it documents was "
            "fixed (update the prose) or the marker is wrong.\n"
            f"{context(f'block {number}', block, result)}"
        )


def test_no_page_documents_a_flag_that_does_not_exist():
    """Cheap static sweep for the exact failure mode that rotted the old documentation.

    Executing the blocks is the real check, but it only covers blocks that run. This catches
    a removed flag anywhere in the prose -- inline code, a skipped block, a table.
    """
    import typer.main

    from gelpy.cli.app import app

    known: set[str] = {"--help", "--version"}

    def collect(command) -> None:
        for param in command.params:
            known.update(param.opts)
            known.update(param.secondary_opts)
        for child in getattr(command, "commands", {}).values():
            collect(child)

    collect(typer.main.get_command(app))

    # Flags belonging to other tools legitimately appear in the docs.
    foreign = {
        "--to", "--check", "--slug",           # tools/sync_plugin.py
        "--accept", "--upgrade",               # tools/docs.py, pip
        "-e", "--no-header", "-q", "-v", "-m", "-k",  # pip / pytest
        "--cov", "--strict", "--report",
        # Metasyntactic placeholders: the docs discuss "a `--flag`" in the abstract, notably
        # where they explain this very test.
        "--flag", "--option", "--your-flag",
    }

    # Deleted in 1.0. Naming them is the migration guide's whole job, so they may be
    # mentioned -- but ONLY in a passage that says they are gone. A page that uses one as if
    # it still worked is the exact failure this test exists to catch.
    removed_in_1_0 = {
        "--maxima-threshold",
        "--maxima-prominence",
        "--peak-width",
        "--sigma",
        "--model",  # still exists on `background remove`; removed from `fit run`
    }
    # Markers that make a mention legitimate: the prose is saying the flag does NOT exist.
    # Checked against the text immediately AROUND each occurrence, not the whole page -- a
    # single disclaimer somewhere must not license a bogus flag three sections later.
    negation = (
        "removed", "no longer", "are gone", "is gone", "deleted", "there is no",
        "does not exist", "do not exist", "never existed", "0.x needed", "gelpy 0.x",
    )
    window = 160

    offenders: dict[str, set[str]] = {}
    uncontextualised: dict[str, set[str]] = {}
    pattern = re.compile(r"`(--[a-z0-9][a-z0-9-]*)`|(?<![\w-])(--[a-z0-9][a-z0-9-]{2,})")
    for page in [*GUIDE.rglob("*.md"), *WALKTHROUGH.rglob("*.md"), REPO / "README.md"]:
        if not page.exists():
            continue
        text = page.read_text()
        lowered = text.lower()
        bad: set[str] = set()
        bare_removed: set[str] = set()
        for match in pattern.finditer(text):
            flag = match.group(1) or match.group(2)
            if flag in known or flag in foreign:
                continue
            if flag.startswith("--gelpy-"):
                continue  # CSS custom properties of the HTML report, not flags
            context = lowered[max(0, match.start() - window) : match.end() + window]
            disclaimed = any(marker in context for marker in negation)
            if flag in removed_in_1_0:
                if not disclaimed:
                    bare_removed.add(flag)
            elif not disclaimed:
                bad.add(flag)
        if bad:
            offenders[str(page.relative_to(REPO))] = bad
        if bare_removed:
            uncontextualised[str(page.relative_to(REPO))] = bare_removed

    assert not offenders, (
        "documentation mentions flags the CLI does not have:\n"
        + "\n".join(f"  {page}: {sorted(flags)}" for page, flags in offenders.items())
    )
    assert not uncontextualised, (
        "documentation mentions flags that were removed in 1.0 without saying so anywhere on "
        "the page, so a reader would take them as current:\n"
        + "\n".join(f"  {page}: {sorted(flags)}" for page, flags in uncontextualised.items())
    )
