"""Assembled analysis result across lanes."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from .common import CanonicalModel
from .peaks import LaneFitResult

# Fixed leading column order; per-shape ``param_*`` columns are appended in first-seen order.
RESULT_COLUMNS = [
    "lane_index",
    "label",
    "band_number",
    "component_class",
    "shape",
    "area",
    "area_fraction",
    "area_uncertainty",
    "maximum_position",
    "residual_fraction",
    "confidence",
    "status",
]


class AnalysisResult(CanonicalModel):
    """All per-lane fit results for a run."""

    lanes: list[LaneFitResult] = Field(default_factory=list)

    def to_rows(self) -> list[dict[str, Any]]:
        """Flatten to result rows (one per band; failed/empty lanes emit one lane-level row)."""
        rows: list[dict[str, Any]] = []
        for lane in self.lanes:
            rows.extend(lane.to_rows())
        return rows

    def to_dataframe(self):
        """Convenience bridge into the optional notebook workflow (pandas lazy-imported)."""
        import pandas as pd  # noqa: PLC0415 — intentional lazy import

        return pd.DataFrame(self.to_rows())
