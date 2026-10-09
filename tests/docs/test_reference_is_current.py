"""The generated reference must match what the code currently produces.

This is the mechanism that makes `docs/reference/` incapable of drifting: it is not
maintained, it is regenerated, and this test is what notices when the committed copy has
fallen behind (or when someone edited a generated file by hand).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools import docs as docs_tool  # noqa: E402

EXPECTED = docs_tool.build(write=False)


@pytest.mark.parametrize("path", sorted(EXPECTED), ids=lambda p: p.name)
def test_generated_file_is_current(path):
    """Regenerate into memory and compare byte for byte."""
    relative = path.relative_to(REPO)
    assert path.exists(), f"{relative} has never been generated. Run: python tools/docs.py build"

    actual, expected = path.read_text(), EXPECTED[path]
    if actual == expected:
        return

    import difflib

    diff = "\n".join(
        difflib.unified_diff(
            expected.splitlines(),
            actual.splitlines(),
            fromfile=f"{relative} (what the code says)",
            tofile=f"{relative} (what is committed)",
            lineterm="",
            n=2,
        )
    )
    pytest.fail(
        f"{relative} does not match the code.\n"
        f"Fix: python tools/docs.py build\n"
        f"(If you edited it by hand, that edit belongs in the source or in docs/guide/.)\n\n"
        f"{diff[:6000]}"
    )


def test_generated_files_carry_the_do_not_edit_marker():
    """A reader who opens one of these must learn immediately not to edit it."""
    for path in sorted(EXPECTED):
        if path.suffix != ".md":
            continue
        first_line = EXPECTED[path].splitlines()[0]
        assert "GENERATED" in first_line, f"{path.name} lacks the generated marker"
        assert "tools/docs.py" in first_line


def test_every_generator_produces_something_substantial():
    """Guards against a generator that silently returns an empty document."""
    for path, content in EXPECTED.items():
        assert len(content.splitlines()) > 15, f"{path.name} looks empty ({len(content)} chars)"


def test_docs_check_reports_no_drift():
    """`tools/docs.py check` is what CI runs; it must agree with this suite."""
    problems = docs_tool.check()
    assert not problems, "documentation drift:\n" + "\n".join(problems)
