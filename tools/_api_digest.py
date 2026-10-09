"""A digest of a Python file's *public surface*, for detecting documentation drift.

The problem this solves
-----------------------
Generated reference documentation cannot drift (a test regenerates and diffs it), and a
documented command that stops parsing gets caught by executing it. Neither helps with
hand-written narrative: a page explaining *why* the profile hash excludes the slice window
goes quietly wrong the day someone changes what feeds that hash. Prose rot is silent.

So every guide page declares the source files it describes, plus a digest of their public
surface. When the digest moves, a test fails and names the page: re-read it, then re-stamp
with ``python tools/docs.py accept <page>``.

Why the digest is semantic, not a file hash
-------------------------------------------
Hashing bytes would fire on every unrelated edit -- a renamed local, a reflowed comment, a
rewritten loop body -- and a stamp that cries wolf gets bumped without reading, which makes
it worse than no stamp at all. So this hashes only what documentation actually depends on:

- the module docstring
- every class and function name, with its signature (argument names, annotations, defaults)
  and its docstring, at any nesting depth
- module- and class-level assignments whose value is a literal (the tuning constants and
  defaults prose quotes)
- pydantic ``Field(...)`` defaults and descriptions, which are prose the runtime itself shows

Rewriting a function body is invisible here. Changing a default, a signature, a constant, or
a documented Field description is not. Prose about behaviour *inside* a function is
deliberately out of scope -- that is what the claim tests in ``tests/docs/test_claims.py``
are for, and the two mechanisms are meant to be used together.

The digest is computed with ``ast`` and never imports the target module, so it works on a
file that does not currently import.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

DIGEST_LENGTH = 16
"""Characters of hex kept from the sha256. 16 hex chars = 64 bits: collision risk is nil
for a few dozen pages, and a short stamp stays readable in front matter."""


def _literal(node: ast.AST) -> str | None:
    """Render ``node`` if it is a literal (or a literal container), else None."""
    try:
        return repr(ast.literal_eval(node))
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return None


def _annotation(node: ast.AST | None) -> str:
    return ast.unparse(node) if node is not None else ""


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """``name(arg: ann = default, *args, kw: ann = default, **kwargs) -> ret``.

    Defaults are rendered via ``ast.unparse`` rather than evaluated, so a non-literal
    default (``Field(...)``, ``field(default_factory=...)``) still contributes its text --
    changing ``Field(description=...)`` therefore moves the digest, which is intended.
    """
    args = node.args
    parts: list[str] = []

    positional = args.posonlyargs + args.args
    pad = [None] * (len(positional) - len(args.defaults))
    for arg, default in zip(positional, pad + list(args.defaults)):
        rendered = f"{arg.arg}: {_annotation(arg.annotation)}"
        if default is not None:
            rendered += f" = {ast.unparse(default)}"
        parts.append(rendered)
        if args.posonlyargs and arg is args.posonlyargs[-1]:
            parts.append("/")

    if args.vararg:
        parts.append(f"*{args.vararg.arg}: {_annotation(args.vararg.annotation)}")
    elif args.kwonlyargs:
        parts.append("*")

    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        rendered = f"{arg.arg}: {_annotation(arg.annotation)}"
        if default is not None:
            rendered += f" = {ast.unparse(default)}"
        parts.append(rendered)

    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}: {_annotation(args.kwarg.annotation)}")

    return f"{node.name}({', '.join(parts)}) -> {_annotation(node.returns)}"


def _assignment_lines(node: ast.AST, prefix: str) -> list[str]:
    """Literal-valued assignments, and any assignment whose value mentions ``Field``.

    A bare ``x: int`` annotation with no value still contributes its annotation, because a
    pydantic field's *type* is part of the contract prose describes.
    """
    lines: list[str] = []
    if isinstance(node, ast.Assign):
        rendered = _literal(node.value)
        if rendered is None and "Field(" in ast.unparse(node.value):
            rendered = ast.unparse(node.value)
        if rendered is not None:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    lines.append(f"{prefix}{target.id} = {rendered}")
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        rendered = "" if node.value is None else (_literal(node.value) or ast.unparse(node.value))
        lines.append(f"{prefix}{node.target.id}: {_annotation(node.annotation)} = {rendered}")
    return lines


def _render_body(body: list[ast.stmt], prefix: str) -> list[str]:
    """Render one suite (module body, class body, ...) into canonical lines.

    Entries are sorted, so moving a definition within a file is not a change: reordering
    does not falsify prose, and treating it as drift would be noise.
    """
    entries: list[tuple[tuple[str, str], list[str]]] = []

    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Dunders other than __init__ are protocol plumbing, not documented surface.
            if node.name.startswith("__") and node.name != "__init__":
                continue
            rendered = [f"{prefix}def {_signature(node)}"]
            doc = ast.get_docstring(node)
            if doc:
                rendered.append(f"{prefix}  doc: {doc.strip()}")
            entries.append((("1def", node.name), rendered))

        elif isinstance(node, ast.ClassDef):
            bases = ", ".join(ast.unparse(b) for b in node.bases)
            rendered = [f"{prefix}class {node.name}({bases})"]
            doc = ast.get_docstring(node)
            if doc:
                rendered.append(f"{prefix}  doc: {doc.strip()}")
            rendered.extend(_render_body(node.body, prefix + "  "))
            entries.append((("2class", node.name), rendered))

        else:
            for line in _assignment_lines(node, prefix):
                entries.append((("0assign", line), [line]))

    entries.sort(key=lambda entry: entry[0])
    return [line for _, rendered in entries for line in rendered]


def _surface_lines(source: str, label: str) -> list[str]:
    """The canonical public-surface rendering of one module's source."""
    tree = ast.parse(source)
    lines = [f"# {label}"]
    doc = ast.get_docstring(tree)
    if doc:
        lines.append(f"module-doc: {doc.strip()}")
    lines.extend(_render_body(tree.body, ""))
    return lines


def surface_text(paths: list[Path], root: Path | None = None) -> str:
    """The canonical text whose hash is the digest. Exposed for readable diffs."""
    chunks: list[str] = []
    for path in sorted(paths, key=lambda p: str(p)):
        label = str(path.relative_to(root)) if root else path.name
        chunks.append("\n".join(_surface_lines(path.read_text(), label)))
    return "\n".join(chunks)


def public_surface_digest(paths: list[Path], root: Path | None = None) -> str:
    """Short hex digest of the public surface of every file in ``paths``."""
    text = surface_text(paths, root)
    return hashlib.sha256(text.encode()).hexdigest()[:DIGEST_LENGTH]
