"""Rules the source code follows everywhere, checked by reading the code."""

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
#: What the rules read: the package, and the tools that build and verify its documentation.
SCANNED = (REPO / "src" / "gelpy", REPO / "tools")

#: The encodings a text file may be opened with. ``utf-8-sig`` reads UTF-8 and drops a byte order mark.
ENCODINGS = {"utf-8", "utf-8-sig"}
#: ``X.open`` calls that never produce text: PIL decodes an image, ``os.open`` returns a file descriptor.
BINARY_ONLY = {("Image", "open"), ("os", "open")}
_MODE_CHARACTERS = set("rwxabt+")


def _is_binary_mode(node):
    """True for a constant mode such as ``"rb"`` (a file name that happens to contain a b is not one)."""
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and "b" in node.value
        and set(node.value) <= _MODE_CHARACTERS
    )


def _text_file_call(node):
    """A label for the text-file call ``node`` makes, or None when it makes none."""
    func = node.func
    if isinstance(func, ast.Attribute):
        owner = func.value.id if isinstance(func.value, ast.Name) else "..."
        if func.attr in {"read_text", "write_text"}:
            return f"{owner}.{func.attr}"
        if owner == "subprocess" and any(
            kw.arg in {"text", "universal_newlines"}
            and not (isinstance(kw.value, ast.Constant) and kw.value.value is False)
            for kw in node.keywords
        ):  # the child's output is decoded with the locale's encoding unless one is given
            return f"subprocess.{func.attr}"
        if func.attr != "open" or (owner, "open") in BINARY_ONLY:
            return None
        # pathlib's open(mode) and io.open(file, mode): look for a binary mode in either place
        modes = [*node.args[:2], *(kw.value for kw in node.keywords if kw.arg == "mode")]
        return None if any(_is_binary_mode(mode) for mode in modes) else f"{owner}.open"
    if isinstance(func, ast.Name) and func.id == "open":
        mode = node.args[1] if len(node.args) > 1 else next(
            (kw.value for kw in node.keywords if kw.arg == "mode"), None
        )
        return None if _is_binary_mode(mode) else "open"
    return None


def _text_io_without_utf8(tree):
    """(line, call) of every text-file read or write that does not say ``encoding="utf-8"``.

    Without an encoding Python uses the locale's, and one that is not a constant UTF-8 name cannot
    be told from the locale's by reading the code."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        call = _text_file_call(node)
        if call is None:
            continue
        encoding = next((kw.value for kw in node.keywords if kw.arg == "encoding"), None)
        if not (isinstance(encoding, ast.Constant) and encoding.value in ENCODINGS):
            yield node.lineno, call


def test_text_files_are_read_and_written_as_utf8():
    """Without an explicit encoding Python uses the locale's, which is not UTF-8 on every
    system: a non-ASCII label would be garbled, or fail, on another machine."""
    offenders = [
        f"{path.relative_to(REPO)}:{line}  {call}"
        for root in SCANNED
        for path in sorted(root.rglob("*.py"))
        for line, call in _text_io_without_utf8(ast.parse(path.read_text(encoding="utf-8")))
    ]
    assert not offenders, "text IO without a constant encoding of 'utf-8' or 'utf-8-sig':\n  " + "\n  ".join(
        offenders
    )


@pytest.mark.parametrize(
    "source",
    [
        "open(p)",
        "open(p, 'w')",
        "open(p, mode='a')",
        "open(p, mode_variable)",
        "open(p, 'w', encoding=None)",
        "p.open()",
        "p.open('w')",
        "io.open(p, 'w')",
        "builtins.open(p)",
        "p.read_text()",
        "p.write_text(text)",
        "p.read_text(encoding=None)",
        "p.read_text(encoding='latin-1')",
        "p.read_text(encoding=ENCODING)",
        "open('tab.txt')",
        "subprocess.run(cmd, text=True)",
        "subprocess.run(cmd, capture_output=True, text=True)",
        "subprocess.check_output(cmd, universal_newlines=True)",
        "subprocess.run(cmd, text=flag)",
        "subprocess.Popen(cmd, text=True, encoding='latin-1')",
    ],
)
def test_the_rule_flags_text_io_without_a_constant_utf8_encoding(source):
    assert list(_text_io_without_utf8(ast.parse(source)))


@pytest.mark.parametrize(
    "source",
    [
        "open(p, 'rb')",
        "open(p, mode='wb')",
        "p.open('rb')",
        "io.open(p, 'wb')",
        "Image.open(p)",
        "os.open(p, os.O_RDONLY)",
        "open(p, encoding='utf-8')",
        "open(p, 'w', encoding='utf-8')",
        "p.open('a', encoding='utf-8-sig')",
        "p.read_text(encoding='utf-8')",
        "p.write_text(text, encoding='utf-8-sig')",
        "subprocess.run(cmd, capture_output=True)",
        "subprocess.run(cmd, text=True, encoding='utf-8')",
        "subprocess.run(cmd, text=False)",
    ],
)
def test_the_rule_passes_binary_files_and_constant_utf8(source):
    assert not list(_text_io_without_utf8(ast.parse(source)))
