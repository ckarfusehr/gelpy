"""Base model and shared response types."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class CanonicalModel(BaseModel):
    """Base for every gelpy contract.

    ``extra="forbid"`` turns an unrecognized field into a hard ``ValidationError``
    instead of a silently ignored typo. ``validate_assignment`` keeps mutated
    instances honest. ``populate_by_name`` lets fields with aliases (e.g. the
    ``lambda_``/``lambda`` EMG parameter) be set either way.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
    )


class ErrorResponse(CanonicalModel):
    """Structured error returned by commands instead of a bare print/raise.

    ``code`` is a stable machine-readable discriminator (e.g. ``lane_selection``,
    ``registry``, ``source_image_changed``) so an agent can branch on the failure
    kind without string-matching the message. ``available`` carries the valid
    choices when the error is a bad selector or an unknown registry name, so an
    agent can recover without reading source.
    """

    error: str
    code: str | None = None
    detail: str | None = None
    available: list[str] | None = None
