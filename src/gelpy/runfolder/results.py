"""Assemble results.csv / results.json from cached fits — never recomputes."""

from __future__ import annotations

import csv
import io
from pathlib import Path

from ..schemas.fit_config import FitConfigStore
from ..schemas.peaks import LaneFitResult
from ..schemas.recipe import Recipe
from ..schemas.results import RESULT_COLUMNS, AnalysisResult
from ..schemas.run import RunManifest
from . import stages
from .cache import write_json
from .layout import RunPaths


def _fit_is_current(record, recipe: Recipe, store: FitConfigStore) -> bool:
    """True if a cached fit still matches the current recipe.

    A recipe edit (lane position, width, background, or a dropped lane) changes the
    expected ``fit_hash``; a stale record whose hash no longer matches must be
    excluded so ``results.csv`` never reports a fit that no longer reflects intent.
    """
    lane_index = _lane_index_of(record.key)
    lane = next((ln for ln in recipe.lanes.lanes if ln.index == lane_index), None)
    if lane is None:
        return False  # lane was dropped from the recipe
    cfg = store.lanes.get(lane_index)
    if cfg is None:
        return True  # no recorded config to validate against; keep (best effort)
    expected = stages.fit_hash(recipe, lane, cfg.profile, cfg.fit)
    return record.hash == expected


def _lane_index_of(key: str) -> int:
    # keys look like "fit:lane=12"
    return int(key.rsplit("=", 1)[1])


def collect_results(
    manifest: RunManifest, recipe: Recipe, store: FitConfigStore
) -> AnalysisResult:
    """Load cached fits that still match the current recipe, sorted by lane index."""
    lanes: list[LaneFitResult] = []
    for record in manifest.stages.values():
        if record.stage != "fit":
            continue
        path = Path(record.cache_path)
        if not path.exists():
            continue
        if not _fit_is_current(record, recipe, store):
            continue
        lanes.append(LaneFitResult.model_validate_json(path.read_text()))
    lanes.sort(key=lambda r: r.lane_index)
    return AnalysisResult(lanes=lanes)


def results_csv_text(result: AnalysisResult) -> str:
    """Render an assembled result as CSV text.

    Shared by ``write_results`` (which materializes the file during a fit pass) and by
    ``results show --format csv`` (which renders the *live* assembly). Reading the file from
    disk there instead would let the two formats of one command disagree: the file is a
    fit-time artifact, so after a recipe edit staled a lane it still lists rows that
    ``--format json`` correctly drops.
    """
    rows = result.to_rows()

    # Column order: legacy fixed columns first, then any extra (param) columns in
    # first-seen order across all rows.
    columns = list(RESULT_COLUMNS)
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()


def write_results(
    paths: RunPaths, manifest: RunManifest, recipe: Recipe, store: FitConfigStore
) -> AnalysisResult:
    """Materialize results.csv and results.json from the cache ledger."""
    result = collect_results(manifest, recipe, store)
    paths.results_csv.write_text(results_csv_text(result))
    write_json(paths.results_json, result)
    return result


def lane_summary(result: LaneFitResult) -> dict:
    """Terse per-lane summary for a command's ``RunUpdateSummary``."""
    if result.status != "ok":
        return {"status": result.status, "confidence": result.confidence, "error": result.error}
    return {
        "status": "ok",
        "confidence": result.confidence,
        "method": result.method,
        "bands": len(result.bands),
        "coverage": round(result.coverage, 3) if result.coverage is not None else None,
        "area_fractions": [round(b.area_fraction, 4) for b in result.bands],
        "classes": [b.component_class for b in result.bands],
        "reasons": result.confidence_reasons or None,
    }
