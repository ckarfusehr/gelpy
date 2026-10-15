"""Lane layout and lane-selection grammar."""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import Field

from .common import CanonicalModel

#: The most lanes a request, a recipe or a fit configuration may hold: the count of lanes, and any
#: list or table with one entry per lane. A count is untyped JSON from a person or an agent, and a
#: recipe can be edited by hand: ten million lanes would run for minutes and fill the disk before
#: any other check spoke. Over it, a field constraint fails with ``validation_error``, which names
#: the field and the limit.
MAX_LANES = 1000

# A single selector term is either an integer index or a string.
#
# String terms carry a tiny prefix grammar interpreted by
# ``core.lanes.resolve_lane_selector``:
#   - ``re:<pattern>``    -> regex search against lane labels
#   - ``label:<text>``    -> forced-literal label match (escape hatch for
#                            numeric-looking labels that would otherwise be
#                            ambiguous with an index)
#   - anything else       -> literal label match
#
# The CLI's ``parse_lane_term`` produces these terms from argv text; ``re:``
# replaces the old ``#``-prefix (``#`` is a shell comment character).
LaneSelectorTerm = Union[int, str]

# A full selector is the literal ``"all"``, a single term, or a list of terms (at most MAX_LANES).
LaneSelector = Union[
    Literal["all"], LaneSelectorTerm, Annotated[list[LaneSelectorTerm], Field(max_length=MAX_LANES)]
]

# Schema-visible description of the selector grammar, attached to request fields so
# it travels into every JSON schema an agent (or MCP client) reads — not just a
# code comment.
LANE_TERM_DESCRIPTION = (
    "A lane selector term: an integer index (0-based), or a string label. "
    "String prefixes: 're:<pattern>' matches labels by regex; 'label:<text>' forces "
    "a literal label match (escape hatch for numeric-looking labels); anything else "
    "is a literal label."
)
LANE_SELECTOR_DESCRIPTION = (
    "Lane selection: the literal 'all', a single selector term, or a list of terms. "
    + LANE_TERM_DESCRIPTION
)


class LaneDefinition(CanonicalModel):
    """One lane: its position on the gel and its human label."""

    index: int = Field(ge=0, description="Zero-based lane index (left to right).")
    label: str = Field(description="Human-readable lane label.")
    x_center_px: float = Field(ge=0, description="Lane center x-position in pixels.")


class LaneLayout(CanonicalModel):
    """The full set of lanes plus the shared profile-extraction width."""

    lanes: list[LaneDefinition] = Field(
        default_factory=list,
        max_length=MAX_LANES,
        description="Lanes in left-to-right order (at most 1000).",
    )
    profile_width_px: int = Field(
        gt=0, description="Width (px) of the averaged column strip per lane."
    )
    label_rotation: float = Field(
        default=90.0, description="Rotation angle (deg) for lane labels in plots."
    )

    def labels(self) -> list[str]:
        return [lane.label for lane in self.lanes]


class LaneDetectionResult(CanonicalModel):
    """Outcome of ``lanes detect``: the fitted N-lane grid plus a confidence signal.

    The large projection array is intentionally omitted: ``commands.run_lane_detection`` hands it
    to the overlay renderer next to this result rather than routing it through the result.
    """

    centers: list[float] = Field(description="Detected lane-center x-positions (px), left to right.")
    pitch: float = Field(description="Fitted center-to-center lane spacing (px).")
    profile_width: int = Field(gt=0, description="Suggested profile strip width (px), from the pitch.")
    n_detected: int = Field(ge=0, description="Number of lane peaks actually found in the projection.")
    n_expected: int = Field(gt=0, description="Requested lane count (ground truth).")
    confidence: Literal["ok", "low"] = Field(
        description="'low' means inspect the overlay and consider manual endpoints (e.g. detected "
        "peaks far off N, or the detected span doesn't cover N-1 lane intervals)."
    )
    confidence_reason: str = Field(
        default="", description="Short human-readable reason when confidence == 'low' (empty otherwise)."
    )
    projection_source: str = Field(
        default="full",
        description="Which row-aggregation strategy produced the projection the grid was fit to "
        "('full' = full-height mean; an adaptive strategy name if one beat the baseline).",
    )
    y_band: list[list[int]] | None = Field(
        default=None,
        description="The informative y-window(s) an adaptive strategy chose, each a [y_start, y_end] "
        "pair (None for the full-height baseline). Multiple pairs for multi-band designs.",
    )
    applied: bool = Field(default=False, description="Whether the centers were persisted to the recipe.")
    overlay: str | None = Field(
        default=None, description="Run-relative path of the verification overlay PNG, if rendered."
    )
