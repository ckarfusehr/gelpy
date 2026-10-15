"""Render a browsable Markdown report from a recipe + results (stdlib templating)."""

from __future__ import annotations

from pathlib import Path
from string import Template

from ..schemas.recipe import Recipe
from ..schemas.results import AnalysisResult
from ..utils.atomic import atomic_write

_TEMPLATE = Template(
    """# gelpy analysis report

**Source image:** `$image_name` (sha256 `$sha256`)
**Dimensions:** ${width} x ${height} px
**Lanes:** $n_lanes (profile width ${profile_width} px)
**Background:** $background

## Results

$results_table

## Lanes

$lane_list
"""
)


def _results_table(result: AnalysisResult) -> str:
    rows = result.to_rows()
    if not rows:
        return "_No fits yet._"
    header = [
        "lane_index", "label", "band", "class", "area_fraction", "x", "confidence", "status"
    ]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in rows:
        # A failed / band-less lane emits a status-only row (no band columns); render
        # those cells blank rather than assuming every row carries band data.
        band = row.get("band_number")
        frac = row.get("area_fraction")
        maxpos = row.get("maximum_position")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["lane_index"]),
                    str(row["label"]),
                    "" if band is None else str(band),
                    str(row.get("component_class", "")),
                    "" if frac is None else f"{frac:.3f}",
                    "" if maxpos is None else f"{maxpos:.1f}",
                    str(row.get("confidence", "")),
                    str(row.get("status", "")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _lane_list(result: AnalysisResult) -> str:
    if not result.lanes:
        return "_No lanes fit yet._"
    lines = []
    for lane in result.lanes:
        if lane.status == "ok":
            cov = f", coverage {lane.coverage:.0%}" if lane.coverage is not None else ""
            flag = "" if lane.confidence == "ok" else f" — **{lane.confidence}**: {'; '.join(lane.confidence_reasons[:2])}"
            lines.append(
                f"- **{lane.label}** (lane {lane.lane_index}): {len(lane.bands)} band(s){cov}{flag}"
            )
        else:
            lines.append(f"- **{lane.label}** (lane {lane.lane_index}): fit {lane.status} — {lane.error}")
    return "\n".join(lines)


def render_report(recipe: Recipe, result: AnalysisResult, out_path: str | Path) -> Path:
    background = recipe.background.model if recipe.background else "none"
    text = _TEMPLATE.substitute(
        image_name=Path(recipe.source_image.path).name,
        sha256=recipe.source_image.sha256,
        width=recipe.source_image.width,
        height=recipe.source_image.height,
        n_lanes=len(recipe.lanes.lanes),
        profile_width=recipe.lanes.profile_width_px,
        background=background,
        results_table=_results_table(result),
        lane_list=_lane_list(result),
    )
    atomic_write(out_path, text.encode("utf-8"))
    return Path(out_path)
