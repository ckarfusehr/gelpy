"""Domain exceptions raised by core algorithms.

Each carries a stable machine-readable ``code`` and enough structured context (e.g.
the available choices) that the layers above can turn it into an ``ErrorResponse``
without re-deriving anything. ``to_error_response`` is that translation — kept here,
next to the exceptions, so the CLI and a future MCP server share one mapping instead
of each reimplementing an except-ladder.
"""

from __future__ import annotations

import json

from pydantic import ValidationError

from ..schemas.common import ErrorResponse


class GelpyError(Exception):
    """Base class for gelpy domain errors."""

    code = "gelpy_error"


class LaneSelectionError(GelpyError):
    """A lane selector matched nothing (or referenced an out-of-range index)."""

    code = "lane_selection"

    def __init__(self, message: str, available: list[str] | None = None):
        super().__init__(message)
        self.available = available or []


class RegistryError(GelpyError):
    """An unknown name was requested from a registry."""

    code = "registry"

    def __init__(self, message: str, available: list[str] | None = None):
        super().__init__(message)
        self.available = available or []


class SourceImageChangedError(GelpyError):
    """The source image on disk no longer matches the recipe's recorded identity.

    Raised instead of silently caching pixels under a stale hash: the recipe is the
    source of truth for the image identity, so a genuine byte change is surfaced for
    the caller to resolve (re-init, or point the recipe at the new file) rather than
    letting the cache go incoherent.
    """

    code = "source_image_changed"


class NonFiniteImageError(GelpyError):
    """The decoded image contains NaN/Inf pixels (would poison percentiles and fits)."""

    code = "nonfinite_image"


class UnsupportedImageError(GelpyError):
    """The image could not be interpreted (unexpected ndim/shape or unknown channel strategy)."""

    code = "unsupported_image"


def to_error_response(exc: Exception) -> ErrorResponse:
    """Translate an exception into a structured ``ErrorResponse``.

    The single raise→response mapping, shared by every presentation layer so the
    machine-readable ``code`` and the ``available`` recovery hint are produced in
    exactly one place. Unknown exceptions get a generic ``internal_error`` rather
    than leaking a traceback.
    """
    if isinstance(exc, GelpyError):
        return ErrorResponse(
            error=str(exc), code=exc.code, available=getattr(exc, "available", None)
        )
    if isinstance(exc, FileNotFoundError):
        return ErrorResponse(error=str(exc), code="not_found")
    if isinstance(exc, ValidationError):
        return ErrorResponse(
            error="validation error",
            code="validation_error",
            detail=json.dumps(exc.errors(include_url=False), default=str),
        )
    return ErrorResponse(error=str(exc), code="internal_error")
