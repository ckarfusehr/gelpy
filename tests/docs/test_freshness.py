"""Guide pages must be re-read when the code they describe changes shape.

Generated reference cannot drift, and documented commands that stop parsing get caught by
executing them. Neither helps with narrative: a page explaining *why* the profile hash
excludes the slice window goes quietly wrong the day someone changes what feeds that hash.

So each page declares its subjects and a digest of their public surface. When the digest
moves, this test names the page. The remedy is to re-read it and then re-stamp:

    python tools/docs.py accept docs/guide/<page>.md

Re-stamping without reading defeats the mechanism; the digest is deliberately insensitive to
edits that cannot falsify prose (comments, function bodies, definition order) so that a
failure here is worth taking seriously.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools import docs as docs_tool  # noqa: E402
from tools._api_digest import public_surface_digest, surface_text  # noqa: E402

PAGES = docs_tool.guide_pages()
REQUIRED_KEYS = ("title", "audience", "summary", "subject")


def test_there_are_guide_pages():
    """Guards against a glob that silently matches nothing, making every test below vacuous."""
    assert PAGES, "docs/guide/ contains no pages"


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_page_has_complete_front_matter(page):
    front, _ = docs_tool.split_front_matter(page.read_text())
    missing = [key for key in REQUIRED_KEYS if not front.get(key)]
    assert not missing, f"{page.relative_to(REPO)} front matter is missing: {missing}"
    assert front["audience"] in ("agent", "human", "both"), front["audience"]


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_page_subjects_exist(page):
    """A subject path that no longer exists means the page describes deleted code."""
    front, _ = docs_tool.split_front_matter(page.read_text())
    for subject in docs_tool.page_subjects(front):
        assert subject.exists(), (
            f"{page.relative_to(REPO)} claims to describe {subject.relative_to(REPO)}, "
            "which does not exist. Update `subject:` to the file that replaced it."
        )


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_page_is_current_with_the_code_it_describes(page):
    front, _ = docs_tool.split_front_matter(page.read_text())
    subjects = docs_tool.page_subjects(front)
    if not subjects:
        pytest.skip("page declares no subjects")

    expected = public_surface_digest(subjects, root=REPO)
    recorded = front.get("subject_digest")
    if recorded == expected:
        return

    listed = "\n      ".join(str(s.relative_to(REPO)) for s in subjects)
    pytest.fail(
        f"{page.relative_to(REPO)} may be out of date.\n"
        f"    The public surface of its subjects changed "
        f"(recorded {recorded or 'nothing'}, now {expected}).\n"
        f"    Subjects:\n      {listed}\n"
        f"    Re-read the page against them, correct anything now false, then:\n"
        f"      python tools/docs.py accept {page.relative_to(REPO)}\n"
        f"    To see exactly what the digest covers:\n"
        f"      python tools/docs.py surface {' '.join(str(s.relative_to(REPO)) for s in subjects)}"
    )


def test_every_source_module_is_described_by_some_page():
    """Every non-trivial source module should be within some page's remit.

    Not a coverage metric for prose quality -- it is a check that a whole subsystem cannot be
    added without anyone deciding where it gets explained. Add the module to an existing
    page's `subject:` list, or to the deliberate exemption list here with a reason.
    """
    exempt = {
        # Package markers and re-export shims: nothing to describe beyond their contents.
        "src/gelpy/__init__.py",
        "src/gelpy/__main__.py",
        "src/gelpy/core/__init__.py",
        "src/gelpy/core/background/__init__.py",
        "src/gelpy/core/peaks/__init__.py",
        "src/gelpy/core/peaks/strategies/__init__.py",
        "src/gelpy/render/__init__.py",
        "src/gelpy/runfolder/__init__.py",
        "src/gelpy/schemas/__init__.py",
        "src/gelpy/cli/__init__.py",
        "src/gelpy/utils/__init__.py",
        # Covered by the generated CLI reference rather than narrative prose.
        "src/gelpy/cli/_io.py",
        # Small leaf utilities whose docstrings are the whole story.
        "src/gelpy/utils/units.py",
        "src/gelpy/core/_registry.py",
    }

    described: set[str] = set()
    for page in PAGES:
        front, _ = docs_tool.split_front_matter(page.read_text())
        described.update(str(s.relative_to(REPO)) for s in docs_tool.page_subjects(front))

    modules = {
        str(path.relative_to(REPO))
        for path in (REPO / "src" / "gelpy").rglob("*.py")
        if "__pycache__" not in path.parts
    }
    undescribed = sorted(modules - described - exempt)
    assert not undescribed, (
        "no guide page lists these modules as a `subject:`, so nobody has decided where they "
        f"are explained:\n  " + "\n  ".join(undescribed)
    )


def test_surface_text_is_available_for_debugging():
    """The failure message points at `docs.py surface`; make sure that path works."""
    subject = REPO / "src/gelpy/schemas/baseline.py"
    text = surface_text([subject], root=REPO)
    assert "src/gelpy/schemas/baseline.py" in text
    assert "class BaselineConfig" in text
