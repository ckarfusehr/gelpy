"""Tests for the documentation machinery itself.

If the drift detector is wrong, every guarantee built on it is worthless -- so it gets
tested harder than the documents it guards.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools._api_digest import public_surface_digest, surface_text  # noqa: E402


def _write(tmp_path: Path, name: str, source: str) -> Path:
    """Write ``source`` to its own directory under a FIXED basename.

    The digest is deliberately path-sensitive (a page's subject list is a list of paths), so
    comparing two variants requires the same filename in different directories -- otherwise
    every comparison differs for the trivial reason that the names differ.
    """
    directory = tmp_path / name.removesuffix(".py")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "module.py"
    path.write_text(source)
    return path


# --- the digest must IGNORE changes that cannot falsify prose ----------------------------


def test_digest_ignores_function_bodies(tmp_path):
    """A refactor that changes no signature and no default must not trip the stamp.

    A stamp that fires on unrelated edits gets bumped without reading, which is worse
    than having no stamp at all.
    """
    a = _write(tmp_path, "a.py", "def f(a: int = 3) -> str:\n    return 'x'\n")
    b = _write(tmp_path, "b.py", "def f(a: int = 3) -> str:\n    y = 'x'  # comment\n    return y\n")
    assert public_surface_digest([a]) == public_surface_digest([b])


def test_digest_ignores_comments(tmp_path):
    a = _write(tmp_path, "a.py", "X = 2\ndef f() -> None: ...\n")
    b = _write(tmp_path, "b.py", "# a new comment\nX = 2\ndef f() -> None: ...\n")
    assert public_surface_digest([a]) == public_surface_digest([b])


def test_digest_ignores_definition_order(tmp_path):
    a = _write(tmp_path, "a.py", "def one() -> None: ...\ndef two() -> None: ...\n")
    b = _write(tmp_path, "b.py", "def two() -> None: ...\ndef one() -> None: ...\n")
    assert public_surface_digest([a]) == public_surface_digest([b])


def test_digest_ignores_private_dunders(tmp_path):
    a = _write(tmp_path, "a.py", "class C:\n    def __repr__(self) -> str: ...\n")
    b = _write(tmp_path, "b.py", "class C:\n    def __repr__(self) -> str:\n        return 'C'\n")
    assert public_surface_digest([a]) == public_surface_digest([b])


# --- the digest must CATCH changes that probably falsify prose ---------------------------


def test_digest_catches_a_changed_default(tmp_path):
    """Defaults are the single most-quoted fact in documentation."""
    a = _write(tmp_path, "a.py", "def f(a: int = 3) -> None: ...\n")
    b = _write(tmp_path, "b.py", "def f(a: int = 4) -> None: ...\n")
    assert public_surface_digest([a]) != public_surface_digest([b])


def test_digest_catches_a_changed_signature(tmp_path):
    a = _write(tmp_path, "a.py", "def f(a: int) -> None: ...\n")
    b = _write(tmp_path, "b.py", "def f(a: int, b: str) -> None: ...\n")
    assert public_surface_digest([a]) != public_surface_digest([b])


def test_digest_catches_a_changed_return_annotation(tmp_path):
    a = _write(tmp_path, "a.py", "def f() -> int: ...\n")
    b = _write(tmp_path, "b.py", "def f() -> str: ...\n")
    assert public_surface_digest([a]) != public_surface_digest([b])


def test_digest_catches_a_renamed_function(tmp_path):
    a = _write(tmp_path, "a.py", "def old_name() -> None: ...\n")
    b = _write(tmp_path, "b.py", "def new_name() -> None: ...\n")
    assert public_surface_digest([a]) != public_surface_digest([b])


def test_digest_catches_a_changed_module_constant(tmp_path):
    """The tuning constants prose quotes -- e.g. _SMOOTHING_WINDOW_FRACTION."""
    a = _write(tmp_path, "a.py", "_FRACTION = 0.7\n")
    b = _write(tmp_path, "b.py", "_FRACTION = 0.35\n")
    assert public_surface_digest([a]) != public_surface_digest([b])


def test_digest_catches_a_changed_docstring(tmp_path):
    """Docstrings are prose the runtime shows; a changed one may need the guide updated."""
    a = _write(tmp_path, "a.py", 'def f() -> None:\n    """One thing."""\n')
    b = _write(tmp_path, "b.py", 'def f() -> None:\n    """A different thing."""\n')
    assert public_surface_digest([a]) != public_surface_digest([b])


def test_digest_catches_a_changed_pydantic_field_default(tmp_path):
    source = 'class M:\n    order: int = Field(default={}, description="Polynomial order.")\n'
    a = _write(tmp_path, "a.py", source.format(2))
    b = _write(tmp_path, "b.py", source.format(3))
    assert public_surface_digest([a]) != public_surface_digest([b])


def test_digest_catches_a_changed_pydantic_field_description(tmp_path):
    """Field descriptions ARE the documentation `gelpy schema` prints."""
    source = 'class M:\n    order: int = Field(default=2, description="{}")\n'
    a = _write(tmp_path, "a.py", source.format("Polynomial order."))
    b = _write(tmp_path, "b.py", source.format("Polynomial order. Use 0 after a boundary."))
    assert public_surface_digest([a]) != public_surface_digest([b])


def test_digest_catches_a_changed_field_annotation(tmp_path):
    a = _write(tmp_path, "a.py", 'class M:\n    status: Literal["ok"] = "ok"\n')
    b = _write(tmp_path, "b.py", 'class M:\n    status: Literal["ok", "empty"] = "ok"\n')
    assert public_surface_digest([a]) != public_surface_digest([b])


def test_digest_catches_a_changed_method_inside_a_class(tmp_path):
    a = _write(tmp_path, "a.py", "class C:\n    def m(self, x: int = 1) -> None: ...\n")
    b = _write(tmp_path, "b.py", "class C:\n    def m(self, x: int = 2) -> None: ...\n")
    assert public_surface_digest([a]) != public_surface_digest([b])


# --- mechanics ---------------------------------------------------------------------------


def test_digest_is_deterministic_across_calls(tmp_path):
    a = _write(tmp_path, "a.py", "X = 1\ndef f(b: str = 'y') -> None: ...\n")
    assert public_surface_digest([a]) == public_surface_digest([a])


def test_digest_covers_every_listed_file(tmp_path):
    a = _write(tmp_path, "a.py", "X = 1\n")
    b = _write(tmp_path, "b.py", "Y = 2\n")
    assert public_surface_digest([a]) != public_surface_digest([a, b])


def test_digest_is_order_independent_across_files(tmp_path):
    a = _write(tmp_path, "a.py", "X = 1\n")
    b = _write(tmp_path, "b.py", "Y = 2\n")
    assert public_surface_digest([a, b]) == public_surface_digest([b, a])


def test_digest_works_on_a_module_that_cannot_be_imported(tmp_path):
    """AST-only by design: a page's subject must be digestible even if it fails to import."""
    a = _write(tmp_path, "a.py", "import a_module_that_does_not_exist\nX = 1\n")
    assert public_surface_digest([a])


def test_surface_text_is_human_readable(tmp_path):
    """The failure message shows this text, so it has to be diffable by eye."""
    a = _write(tmp_path, "a.py", '"""Mod."""\nX = 7\ndef f(q: int = 1) -> str:\n    """Doc."""\n')
    text = surface_text([a])
    assert "module-doc: Mod." in text
    assert "X = 7" in text
    assert "def f(q: int = 1) -> str" in text
    assert "doc: Doc." in text


def test_digest_on_the_real_package_is_stable():
    """Sanity check against actual gelpy source, not just toy files."""
    subjects = [REPO / "src/gelpy/schemas/baseline.py", REPO / "src/gelpy/core/lanes.py"]
    first = public_surface_digest(subjects, root=REPO)
    assert len(first) == 16
    assert first == public_surface_digest(subjects, root=REPO)


def test_digest_changes_if_a_real_default_changes(tmp_path):
    """End-to-end on real source: flip poly_order 2 -> 3 in a copy and the digest moves."""
    original = (REPO / "src/gelpy/schemas/baseline.py").read_text()
    assert "default=2," in original, "poly_order default moved; update this test"
    copy = _write(tmp_path, "baseline.py", original.replace("default=2,", "default=3,", 1))
    assert public_surface_digest([copy]) != public_surface_digest(
        [REPO / "src/gelpy/schemas/baseline.py"]
    )


def test_front_matter_stays_valid_yaml_for_every_page():
    """Front matter is a YAML block, so readers other than gelpy will parse it as one.

    gelpy's own parser partitions on the first colon and tolerates anything, which is exactly
    how an invalid document survives unnoticed: a title containing ": " (as
    "Concepts: the run folder and the hash chain" did) makes the block unparseable for any
    real YAML reader.
    """
    yaml = pytest.importorskip("yaml")

    from tools import docs as docs_tool

    for page in docs_tool.guide_pages() + sorted((REPO / "docs" / "walkthrough").glob("*.md")):
        text = page.read_text()
        if not text.startswith("---\n"):
            continue
        end = text.find("\n---\n", 4)
        block = text[4:end]
        try:
            parsed = yaml.safe_load(block)
        except yaml.YAMLError as exc:  # pragma: no cover - the failure we are preventing
            pytest.fail(f"{page.relative_to(REPO)} front matter is not valid YAML: {exc}")
        assert isinstance(parsed, dict), f"{page.relative_to(REPO)} front matter is not a mapping"
        # And gelpy's own parser must agree with the real one on the scalars it reads.
        mine, _ = docs_tool.split_front_matter(text)
        for key in ("title", "audience", "summary"):
            if key in parsed:
                assert mine[key] == parsed[key], f"{page.name}: {key} parses differently"


def test_render_front_matter_quotes_values_that_need_it():
    from tools.docs import render_front_matter, split_front_matter

    rendered = render_front_matter(
        {"title": "Concepts: the run folder", "audience": "both", "summary": "A: B", "subject": []}
    )
    yaml = pytest.importorskip("yaml")
    parsed = yaml.safe_load(rendered.strip().strip("-\n"))
    assert parsed["title"] == "Concepts: the run folder"
    assert parsed["summary"] == "A: B"
    # gelpy's own parser must strip the quotes it just added, so a round trip is lossless.
    mine, _ = split_front_matter(rendered + "\nbody\n")
    assert mine["title"] == "Concepts: the run folder"


def test_digest_spells_an_argless_lambda_the_same_on_every_python(tmp_path):
    """ast.unparse writes ``lambda :`` before 3.11 and ``lambda:`` after; the digest must not care."""
    path = _write(
        tmp_path,
        "m.py",
        "from pydantic import BaseModel, Field\n\n\nclass M(BaseModel):\n    x: list[int] = Field(default_factory=lambda: [1])\n",
    )
    assert "lambda:" in surface_text([path]) and "lambda :" not in surface_text([path])
