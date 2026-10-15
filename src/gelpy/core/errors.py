"""Domain exceptions, and the one mapping from any exception to an ``ErrorResponse``.

Each ``GelpyError`` subclass carries a stable machine-readable ``code`` and a ``next_step``
sentence; an instance may add a more specific ``detail`` and the ``available`` valid choices.
``to_error_response`` turns *any* exception into an ``ErrorResponse``. Every front-end calls it,
so a failure has one shape and one code whichever front-end met it, and a caller never has to
read a traceback.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from ..schemas.common import ErrorResponse


class GelpyError(Exception):
    """Base class for gelpy domain errors."""

    code = "gelpy_error"
    next_step = "Read the error message; it names the value or file to change."

    def __init__(
        self, message: str = "", *, detail: str | None = None, available: list[str] | None = None
    ):
        super().__init__(message)
        self.detail = detail
        self.available = available


class LaneSelectionError(GelpyError):
    """A lane selector matched nothing (or referenced an out-of-range index)."""

    code = "lane_selection"
    next_step = (
        "Select lanes by a label from `available`, a 0-based index, or `re:<pattern>`; "
        "`gelpy status RUN` lists the lanes."
    )

    def __init__(
        self, message: str, available: list[str] | None = None, *, detail: str | None = None
    ):
        super().__init__(message, detail=detail, available=available or [])


class RegistryError(GelpyError):
    """An unknown name was requested from a registry."""

    code = "registry"
    next_step = "Use one of the names in `available`."

    def __init__(
        self, message: str, available: list[str] | None = None, *, detail: str | None = None
    ):
        super().__init__(message, detail=detail, available=available or [])


class SourceImageChangedError(GelpyError):
    """The source image on disk no longer matches the recipe's recorded identity.

    Raised instead of silently caching pixels under a stale hash: the recipe is the
    source of truth for the image identity, so a genuine byte change is surfaced for
    the caller to resolve (re-init, or point the recipe at the new file) rather than
    letting the cache go incoherent.
    """

    code = "source_image_changed"
    next_step = (
        "Restore the original file, or analyse the changed image in a new run folder "
        "(`gelpy init NEW_RUN IMAGE`)."
    )


class SourceImageMissingError(FileNotFoundError):
    """The source image a run records is not where the run folder looks for it.

    Not a ``GelpyError``: as far as the code goes it is a missing file (``not_found``), so a
    caller that branches on that code keeps working. Its ``detail`` says which image is missing
    and where to put it back, which the generic next step of ``not_found`` cannot.
    """

    def __init__(self, message: str, *, detail: str):
        super().__init__(message)
        self.detail = detail


class NonFiniteImageError(GelpyError):
    """The decoded image contains NaN/Inf pixels (would poison percentiles and fits)."""

    code = "nonfinite_image"
    next_step = (
        "Replace the NaN/Inf pixels (export the image again) and run `gelpy init` on the clean file."
    )


class UnsupportedImageError(GelpyError):
    """The image could not be interpreted (unexpected ndim/shape or unknown channel strategy)."""

    code = "unsupported_image"
    next_step = (
        "Export the image as a single-page grayscale or RGB TIFF or PNG, or choose another "
        "`--channel`, and run `gelpy init` again."
    )


class ImageDecodeError(GelpyError):
    """The file could not be decoded as an image: not an image, damaged, or a format the
    decoder does not read."""

    code = "decode_error"
    next_step = (
        "Check that the file is a TIFF, PNG or JPEG image; export it again if it is damaged."
    )


class ImageTooLargeError(GelpyError):
    """The file's header declares more pixels (summed over its pages) or more pages than the
    configured limits; it is refused before any pixel data is decoded."""

    code = "image_too_large"
    next_step = (
        "Crop or downsample the image, export only the page you need, or raise the limits with "
        "GELPY_MAX_PIXELS / GELPY_MAX_PAGES."
    )


class InvalidValueError(GelpyError, ValueError):
    """A request value is inconsistent with itself or with the run folder, for example two
    labels for twenty lanes. A bare ``ValueError`` raised anywhere maps to this code too."""

    code = "invalid_value"
    next_step = (
        "Check the value or file the message names; if it names none, this is a bug in gelpy: "
        "report it."
    )


class RunExistsError(GelpyError):
    """The folder already holds a run (a ``recipe.json``); ``init`` does not replace one silently."""

    code = "run_exists"
    next_step = "Pass --force (tool: force=true) to replace the run, or choose a new run folder."


class RunLockedError(GelpyError):
    """Another process, or another thread of this one, is writing the run folder."""

    code = "run_locked"
    next_step = "Wait for the other gelpy command on this run folder to finish, then repeat the call."


class BackgroundRequiredError(GelpyError):
    """Profiles or fits were requested on a run without background removal: each lane would
    carry its own background floor into its normalization, so lanes stop being comparable."""

    code = "background_required"
    next_step = (
        "Configure background removal first (`gelpy background remove RUN`, tool "
        "`background_remove`), or pass --allow-raw (tool: allow_raw=true) to measure the raw "
        "image on purpose."
    )


class MissingDependencyError(GelpyError):
    """An optional dependency the command needs is not installed."""

    code = "missing_dependency"
    next_step = 'Install the extra the message names, e.g. `pip install "gelpy[mcp]"`.'


class DisplayRequiredError(GelpyError):
    """A command that needs a window (``lanes pick``) ran where there is none: no GUI backend of
    matplotlib could be selected, or none could open a window."""

    code = "display_required"
    next_step = (
        "Set the lanes without a window: `lanes set RUN --endpoints LEFT RIGHT --n-lanes N`, or "
        "`--x-center X` once per lane (tool `lanes_set`); or run `lanes pick` on a machine with a display."
    )


class PickCancelledError(GelpyError):
    """The window of ``lanes pick`` ended before all the clicks were made (it was closed, or Enter
    was pressed early). Nothing was changed."""

    code = "pick_cancelled"
    next_step = "Run `lanes pick` again, click every lane center it asks for, then press Enter."


#: The longest value of the caller's that an error repeats, in characters: the ``input`` and the name
#: of a field in a ``validation_error``, the run folder, tool or lane selector a message quotes. A
#: caller can send a megabyte where a name belongs, or a damaged file whole; an error must stay small
#: enough to read.
MAX_ECHOED_INPUT = 200


def _safe_str(value: Any) -> str:
    """``str(value)``, or the name of its type when that raises."""
    try:
        return str(value)
    except Exception:
        return f"<{type(value).__name__}>"


def _cut(text: str) -> str:
    """``text`` whole while it is short, else its first ``MAX_ECHOED_INPUT`` characters and its length."""
    if len(text) <= MAX_ECHOED_INPUT:
        return text
    return f"{text[:MAX_ECHOED_INPUT]}... [cut: {len(text)} characters in all]"


def shown(value: Any) -> str:
    """``value`` as an error message quotes it: its ``repr``, cut like every other echo.

    A message that names a run folder, a tool or a lane selector passes it through here, so a
    megabyte sent in its place does not come back. A ``repr`` that raises is survived.
    """
    try:
        text = repr(value)
    except Exception:  # a repr that raises, an integer of thousands of digits
        text = f"<{type(value).__name__}>"
    return _cut(text)


def echoed(value: Any) -> Any:
    """``value`` as a ``validation_error`` echoes it: whole while its JSON text is short, so a
    structured input stays structured, else the first ``MAX_ECHOED_INPUT`` characters of that text
    and its full length. What JSON cannot write at all (a cycle, 4300 digits) is shown instead."""
    try:
        text = value if isinstance(value, str) else json.dumps(value, default=_safe_str)
    except Exception:
        return shown(value)
    return value if len(text) <= MAX_ECHOED_INPUT else _cut(text)


#: Codes for failures that are not ``GelpyError``s: code -> (raised when, next step).
NON_DOMAIN_CODES: dict[str, tuple[str, str]] = {
    "not_found": (
        "a file or run folder does not exist (`FileNotFoundError`); for the source image a run "
        "records, `detail` says where to put it back",
        "Check the path; a run folder is created with `gelpy init RUN IMAGE`.",
    ),
    "validation_error": (
        "a request failed validation (pydantic `ValidationError`); `detail` carries the field "
        f"errors as a JSON string, each echoed `input` and field name cut at {MAX_ECHOED_INPUT} "
        "characters, and `available` lists the allowed values when the first failing field takes "
        "one of a fixed set",
        "Fix the fields listed in `detail`; `gelpy schema <Model>` prints the contract.",
    ),
    "usage_error": (
        "the command line is malformed: an unknown option or command, a missing argument "
        "(exit code 2)",
        "Run the command with `--help`; `available` lists its options or subcommands.",
    ),
    "os_error": (
        "the operating system refused a file operation (permissions, a full disk, ...)",
        "Check permissions and free space for the path in the message.",
    ),
    "out_of_memory": (
        "the process ran out of memory",
        "Crop or downsample the image, or fit the background on a subsample "
        "(`--param subsample=4`).",
    ),
    "internal_error": (
        "any other exception, reported instead of a traceback",
        "This is a bug in gelpy; report it together with the command that triggered it.",
    ),
}


def _non_domain(code: str, error: str, detail: str | None = None) -> ErrorResponse:
    return ErrorResponse(error=error, code=code, detail=detail or NON_DOMAIN_CODES[code][1])


#: One allowed value in the ``expected`` text of a pydantic ``literal_error``, e.g.
#: ``'raw', 'linear' or 'corrected'``: the ``repr`` of each value, joined by ", " and a last " or ".
#: A quoted string is matched whole, so a value that contains a comma or " or " stays one value.
_LITERAL_VALUE = re.compile(r"""'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*"|[^\s,]+""")


def _allowed_values(error: Mapping[str, Any], errors: list[Mapping[str, Any]]) -> list[str] | None:
    """The values a pydantic ``literal_error`` allows, as ``available`` lists them.

    None when ``error`` is another kind of error, when the list would mislead, or when it cannot
    be read: a missing list is better than a wrong one. It misleads for one branch of a union
    (``lanes`` takes the literal ``'all'``, but also an index, a label or a list of them): such a
    branch's location ends in the branch's own name, ``literal['all']``. It misleads as well when
    another error names other values for a field of the same name, as the branches of a union of
    models do: each branch reports the values of its own.
    """
    if error.get("type") != "literal_error":
        return None
    location = error.get("loc") or ()
    if location and str(location[-1]).startswith("literal["):
        return None
    expected = (error.get("ctx") or {}).get("expected")
    if not isinstance(expected, str):
        return None
    for other in errors:
        elsewhere = (other.get("ctx") or {}).get("expected")
        if elsewhere not in (None, expected) and (other.get("loc") or ())[-1:] == location[-1:]:
            return None
    values: list[str] = []
    for token in _LITERAL_VALUE.findall(expected):
        if token == "or":  # the last separator; no value is the bare word
            continue
        try:
            value = ast.literal_eval(token)
        except (ValueError, SyntaxError):
            return None  # an enum member, bytes: not a value a JSON caller could send
        if isinstance(value, str):
            values.append(value)
        elif value is None or isinstance(value, (bool, int, float)):
            values.append(json.dumps(value))  # as a JSON caller writes it: true, null, 1
        else:
            return None
    return values or None


def _os_text(exc: OSError) -> str:
    """``str(exc)`` of an error the operating system raised, its file name cut like every other echo."""
    text = _safe_str(exc)
    name = getattr(exc, "filename", None)
    if isinstance(name, str) and len(name) > MAX_ECHOED_INPUT:
        text = text.replace(repr(name), shown(name))
    return text


def to_error_response(exc: BaseException) -> ErrorResponse:
    """Translate any exception into an ``ErrorResponse`` whose ``detail`` names the next step.

    The single raise -> response mapping, shared by every front-end. ``validation_error`` is
    the one code whose ``detail`` is data rather than a sentence: the pydantic field errors as
    a JSON string, each echoed ``input`` and each part of a location cut at ``MAX_ECHOED_INPUT``
    characters. Its ``error`` names the first failing field (when the error has a location;
    invalid JSON has none) and the schema command instead, and its ``available`` lists that
    field's allowed values when it takes one of a fixed set (a ``Literal``).
    """
    if isinstance(exc, GelpyError):
        return ErrorResponse(
            error=_safe_str(exc) or exc.code,
            code=exc.code,
            detail=exc.detail or exc.next_step,
            available=exc.available,
        )
    if isinstance(exc, ValidationError):  # before ValueError: pydantic's is a subclass of it
        errors = exc.errors(include_url=False)
        # A union reports every branch, in order: a list over its limit is reported first, as the one
        # error that says what to change (``lanes.list[...]: List should have at most 1000 items``).
        errors.sort(key=lambda error: error["type"] != "too_long")
        for error in errors:
            error["loc"] = tuple(echoed(part) for part in error["loc"])
            if "input" in error:
                error["input"] = echoed(error["input"])
        first = errors[0] if errors else {"loc": (), "msg": "invalid"}
        where = echoed(".".join(_safe_str(part) for part in first["loc"]))
        prefix = f"{exc.title}: {where}: " if where else f"{exc.title}: "
        more = f" (and {len(errors) - 1} more)" if len(errors) > 1 else ""
        return ErrorResponse(
            error=f"{prefix}{first['msg']}{more}. `gelpy schema {exc.title}` prints the contract.",
            code="validation_error",
            detail=json.dumps(errors, default=_safe_str),
            available=_allowed_values(first, errors),
        )
    if isinstance(exc, FileNotFoundError):  # one that knows what to do about it says so
        return _non_domain("not_found", _os_text(exc), getattr(exc, "detail", None))
    if isinstance(exc, ValueError):
        return ErrorResponse(
            error=_safe_str(exc) or "invalid value",
            code=InvalidValueError.code,
            detail=InvalidValueError.next_step,
        )
    if isinstance(exc, MemoryError):
        return _non_domain("out_of_memory", _safe_str(exc) or "MemoryError")
    if isinstance(exc, OSError):
        return _non_domain("os_error", _os_text(exc))
    return _non_domain("internal_error", f"{type(exc).__name__}: {_safe_str(exc)}")
