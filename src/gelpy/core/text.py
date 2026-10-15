"""Text from bytes, with the error a person can act on: the file, the line and the encoding."""

from __future__ import annotations

import re

from .errors import InvalidValueError

#: A line ending: CR LF, a lone LF, or a lone CR (what a file written on an old Mac uses).
_LINE_ENDING = re.compile(rb"\r\n|\r|\n")


def line_of(data: bytes, offset: int) -> int:
    """The 1-based line of ``data`` that ``offset`` lies on.

    CR LF, a lone LF and a lone CR each end one line, so the number is the one a text editor shows
    whichever system wrote the file.
    """
    return len(_LINE_ENDING.findall(data, 0, offset)) + 1


def decode_utf8(data: bytes, *, what: str, detail: str | None = None, bom: bool = False) -> str:
    """``data`` as UTF-8 text; a byte that is not UTF-8 is an ``InvalidValueError``.

    The error names ``what`` (the file, as the caller wants it quoted), the line of the first byte that
    is not UTF-8 and the encoding, so that a file saved in another encoding is not reported as a
    codec error, and not as a bug. ``detail`` is its next step (the default saves the file as UTF-8).
    ``bom`` drops the byte order mark that Notepad and Excel write at the start of a UTF-8 file.
    """
    try:
        return data.decode("utf-8-sig" if bom else "utf-8")
    except UnicodeDecodeError as exc:
        # ``exc.object`` is what the decoder was given: without the byte order mark that utf-8-sig drops
        raise InvalidValueError(
            f"{what} must be UTF-8 text, but line {line_of(exc.object, exc.start)} is not valid UTF-8.",
            detail=detail or "Save the file as UTF-8 text and repeat the command.",
        ) from exc
