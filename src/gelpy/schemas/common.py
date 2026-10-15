"""Base model and shared response types."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


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
    choices when the error is a bad selector, an unknown registry name or a value
    outside a fixed set, so an agent can recover without reading source.
    """

    error: str = Field(description="What failed, for people and logs. Do not parse it.")
    code: str | None = Field(
        default=None,
        description="Stable machine-readable kind of failure; branch on this. Catalogue: "
        "docs/reference/errors.md.",
    )
    detail: str | None = Field(
        default=None,
        description="The next step: what to run or change. For code 'validation_error' it is "
        "the JSON list of field errors instead.",
    )
    available: list[str] | None = Field(
        default=None,
        description="The valid choices, when the failure is an unknown name or option, or a "
        "validation_error on a field that takes one of a fixed set of values.",
    )
