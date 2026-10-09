"""Structural checks on the documentation set: links resolve, nothing is orphaned.

A dead internal link is a small lie with a large cost -- it is exactly what a reader follows
when they are already confused. And a page that exists but appears in no navigation and no
index is invisible, which is nearly the same as being wrong.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs"
sys.path.insert(0, str(REPO))

from tools import docs as docs_tool  # noqa: E402

# Markdown links, excluding images (handled separately) and anything absolute/anchored.
LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)#][^)]*)\)")
IMAGE_LINK = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")

MARKDOWN_FILES = sorted(
    path for path in DOCS.rglob("*.md") if "__pycache__" not in path.parts
) + [REPO / "README.md", REPO / "CHANGELOG.md"]


def _is_external(target: str) -> bool:
    return target.startswith(("http://", "https://", "mailto:"))


@pytest.mark.parametrize(
    "page", [p for p in MARKDOWN_FILES if p.exists()], ids=lambda p: str(p.relative_to(REPO))
)
def test_relative_links_resolve(page):
    broken = []
    for target in LINK.findall(page.read_text()):
        target = target.split()[0].strip()  # drop any "title" suffix
        if _is_external(target) or not target:
            continue
        resolved = (page.parent / target.split("#")[0]).resolve()
        if not resolved.exists():
            broken.append(target)
    assert not broken, f"{page.relative_to(REPO)} has dead links: {broken}"


@pytest.mark.parametrize(
    "page", [p for p in MARKDOWN_FILES if p.exists()], ids=lambda p: str(p.relative_to(REPO))
)
def test_images_resolve(page):
    broken = []
    for target in IMAGE_LINK.findall(page.read_text()):
        target = target.split()[0].strip()
        if _is_external(target) or not target:
            continue
        if not (page.parent / target).resolve().exists():
            broken.append(target)
    assert not broken, f"{page.relative_to(REPO)} references missing images: {broken}"


def test_every_page_is_in_the_mkdocs_nav_or_explicitly_excluded():
    """A page must be navigable or deliberately hidden -- never accidentally invisible.

    The exclude list lives in mkdocs.yml as a comment block so the decision is recorded next
    to the navigation it affects.
    """
    config_path = REPO / "mkdocs.yml"
    if not config_path.exists():
        pytest.skip("mkdocs.yml not present")

    import fnmatch

    config = config_path.read_text()
    # Deliberately not a YAML parse: this only needs to know whether each page is accounted
    # for, and the documentation tooling should not add a PyYAML dependency for that. Pages
    # in `nav` are matched by literal path; deliberate omissions are declared as
    # `# excluded: <glob>` comments, so the reason sits next to the navigation it affects.
    excluded_globs = re.findall(r"#\s*excluded:\s*(\S+)", config)
    assert excluded_globs, "mkdocs.yml declares no `# excluded:` globs -- did the format change?"

    missing = []
    for page in DOCS.rglob("*.md"):
        relative = str(page.relative_to(DOCS))
        if relative in config:
            continue
        if any(fnmatch.fnmatch(relative, pattern) for pattern in excluded_globs):
            continue
        missing.append(relative)
    assert not missing, (
        "these pages appear in neither mkdocs.yml nav nor its documented exclude list:\n  "
        + "\n  ".join(sorted(missing))
        + "\n\nAdd them to `nav`, or to an `# excluded:` glob with the reason."
    )


def test_docs_index_covers_every_guide_page():
    """The machine-readable index is what an agent reads to choose a page; it must be complete."""
    index = json.loads((DOCS / "docs-index.json").read_text())
    indexed = {entry["path"] for entry in index["pages"]}
    expected = {
        str(page.relative_to(REPO))
        for page in list(docs_tool.guide_pages()) + sorted((DOCS / "walkthrough").glob("*.md"))
        # Same rule the builder uses: front matter is what makes a file a documentation page.
        if docs_tool.split_front_matter(page.read_text())[0]
    }
    assert indexed == expected


def test_docs_index_entries_are_useful():
    """Every indexed page needs a summary an agent can route on."""
    index = json.loads((DOCS / "docs-index.json").read_text())
    for entry in index["pages"]:
        assert entry["summary"], f"{entry['path']} has no summary"
        assert entry["title"], f"{entry['path']} has no title"
        assert entry["audience"] in ("agent", "human", "both"), entry


def test_generated_pages_are_not_hand_edited():
    """Generated reference pages must keep their marker as the first line."""
    for page in sorted((DOCS / "reference").glob("*.md")):
        assert page.read_text().startswith("<!-- GENERATED"), (
            f"{page.relative_to(REPO)} lost its generated marker -- was it edited by hand?"
        )


def test_every_decision_record_labels_its_own_status():
    """A design spec must never read as current behaviour.

    Three of this repo's specs previously described features that had shipped differently, and
    one implementation plan had none of its 58 steps ticked despite being fully implemented and
    merged. A reader arriving at one needs to know, in the first few lines, whether it
    describes the system, something that shipped differently, or something never built. Each
    record therefore carries its own status banner -- a directory-level disclaimer is not
    enough, because people arrive at these by search and by link, not by browsing.
    """
    markers = ("historical", "superseded", "shipped", "never built", "completed", "open")
    unlabelled = []
    for page in (DOCS / "decisions").rglob("*.md"):
        if page.name == "README.md" and page.parent.name == "decisions":
            continue  # the index itself, which explains the rule
        head = " ".join(page.read_text().splitlines()[:14]).lower()
        if not any(marker in head for marker in markers):
            unlabelled.append(str(page.relative_to(REPO)))
    assert not unlabelled, (
        "these decision records do not state their status in the first 14 lines:\n  "
        + "\n  ".join(sorted(unlabelled))
    )
