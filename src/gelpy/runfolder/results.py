"""Assemble results.csv / results.json from cached fits — never recomputes."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..core.errors import InvalidValueError, shown
from ..schemas.fit_config import FitConfigStore
from ..schemas.peaks import LaneFitResult
from ..schemas.recipe import Recipe
from ..schemas.results import RESULT_COLUMNS, AnalysisResult
from ..schemas.run import RunManifest
from ..utils.atomic import atomic_write
from . import stages
from .cache import write_json
from .layout import RunPaths
from .manifest import record_stage
from .stages import StageName, stage_key


def _fit_is_current(record, recipe: Recipe, store: FitConfigStore) -> bool:
    """True if a cached fit still matches the current recipe and the running gelpy version.

    A recipe edit (lane position, width, background, or a dropped lane) or a new gelpy version
    changes the expected ``fit_hash``; a stale record whose hash no longer matches must be
    excluded so ``results.csv`` never reports a fit that no longer reflects intent.

    A fit with no entry in the fit-config store has no config to recompute the expected hash
    from, so only its version can be checked: it is kept while the running gelpy computed it
    (best effort) and stale otherwise. A record an older release wrote carries no version, so
    that counts as another version.
    """
    lane_index = _lane_index_of(record.key)
    lane = next((ln for ln in recipe.lanes.lanes if ln.index == lane_index), None)
    if lane is None:
        return False  # lane was dropped from the recipe
    cfg = store.lanes.get(lane_index)
    if cfg is None:
        return record.gelpy_version == stages.code_version()
    expected = stages.fit_hash(recipe, lane, cfg.profile, cfg.fit)
    return record.hash == expected


def _lane_index_of(key: str) -> int:
    """The lane of a stage key such as ``fit:lane=12``.

    A key that names none (a hand-edited manifest.json) is an ``InvalidValueError`` that says which
    record and what to do: the manifest is a cache ledger, and is built again."""
    try:
        return int(key.rsplit("=", 1)[1])
    except (IndexError, ValueError):
        raise InvalidValueError(
            f"manifest.json holds a record with the key {shown(key)}, which names no lane.",
            detail="manifest.json is the cache ledger. Run `gelpy fit run` to fit the lanes again (the "
            "cache is rebuilt), or remove only that record from manifest.json; deleting the whole "
            "file leaves `results show` with no lanes until a fit command runs.",
        ) from None


def read_fit_result(path: Path) -> LaneFitResult:
    """The fit in the cache file ``path``.

    A file that does not validate is an ``InvalidValueError`` that names the file and the first
    problem (which field, and what kind of problem) and never what the file holds: a manifest is a
    plain file, and its record may point at any file the process can read. The field is named only
    when it is a field of the result (a name from the file is the file's content), and the kind of
    problem is pydantic's own code (``int_parsing``, ``missing``), never its message or its input.
    """
    where = shown(str(path))
    try:
        return LaneFitResult.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        errors = exc.errors(include_url=False, include_context=False, include_input=False)
        first = errors[0] if errors else {"loc": (), "type": "invalid"}
        top = first["loc"][0] if first["loc"] else None
        field = top if top in LaneFitResult.model_fields else "an unknown field"
        more = f" (and {len(errors) - 1} more)" if len(errors) > 1 else ""
        problem = f"{field}: {first['type']}{more}"
        raise InvalidValueError(
            f"The fit file {where} is not a valid fit result ({problem}).",
            detail=FIT_AGAIN,
        ) from None
    except UnicodeDecodeError:
        raise InvalidValueError(f"The fit file {where} is not UTF-8 text.", detail=FIT_AGAIN) from None


#: What to do about a fit file that does not validate: a fit command computes it again.
FIT_AGAIN = (
    "Run `fit run` (tool `fit_run`) for the lane again: a cached fit that does not validate is "
    "computed again."
)


def follow_the_cache(paths: RunPaths, recipe: Recipe, manifest: RunManifest, store: FitConfigStore) -> None:
    """Make the ledger's fit records name the fits the recipe and the running gelpy call for.

    The cache keeps every fit it computed, keyed by what it was computed from, so an edit that is
    undone, or a gelpy version that is returned to, makes an older fit current again; the ledger
    still names the newer one, because only a command that computes or reuses a stage records it. A
    fit is current when its expected artifact exists (``status.stale_stages`` has always judged so),
    so a reader calls this on the manifest it loaded before it judges a fit: each lane the ledger
    records, whose stored configuration says which fit is expected, gets a record that names it
    when that fit is cached. Only the manifest in memory changes; a command that writes saves the
    ledger it holds. A lane the ledger does not record has no fit to follow (a wiped ledger is a
    ledger to rebuild), and a lane with no stored configuration keeps its record: there is no
    configuration to compute the expected hash from.
    """
    for lane in recipe.lanes.lanes:
        key = stage_key(StageName.FIT, lane.index)
        config = store.lanes.get(lane.index)
        if key not in manifest.stages or config is None:
            continue
        expected = stages.fit_hash(recipe, lane, config.profile, config.fit)
        artifact = paths.fit_cache(lane.index, expected)
        if artifact.exists():
            record_stage(manifest, key, StageName.FIT.value, expected, paths.store_path(artifact))


def collect_results(
    paths: RunPaths,
    manifest: RunManifest,
    recipe: Recipe,
    store: FitConfigStore,
    *,
    damaged: list[int] | None = None,
) -> AnalysisResult:
    """Load the current fits, sorted by lane index.

    A fit is current when the recipe, the stored configuration and the running gelpy version call
    for it and it is in the cache (``follow_the_cache``, which this calls on ``manifest``): an edit
    that is undone makes its fits current again. Each lane carries its current label: the fit hash
    leaves the label out (renaming a lane must not refit it), so a cached fit can hold the label
    the lane had when it was fitted.

    A current fit whose file does not validate is left out, as ``status`` counts it missing; with
    ``damaged`` (a list) its lane index is appended to it, so that a caller that must say which
    lanes it left out (the HTML report) can. A ledger record of a fit that names no lane is an
    ``InvalidValueError`` (``_lane_index_of``).
    """
    follow_the_cache(paths, recipe, manifest, store)
    labels = {lane.index: lane.label for lane in recipe.lanes.lanes}
    lanes: list[LaneFitResult] = []
    for record in manifest.stages.values():
        if record.stage != "fit":
            continue
        lane_index = _lane_index_of(record.key)
        path = paths.resolve(record.cache_path)
        if not path.exists():
            continue
        if not _fit_is_current(record, recipe, store):
            continue
        try:
            result = read_fit_result(path)
        except InvalidValueError:  # the file does not validate: it counts as missing, as in ``status``
            if damaged is not None:
                damaged.append(lane_index)
            continue
        lanes.append(result.model_copy(update={"label": labels.get(result.lane_index, result.label)}))
    lanes.sort(key=lambda r: r.lane_index)
    if damaged is not None:
        damaged.sort()
    return AnalysisResult(lanes=lanes, gelpy_version=stages.code_version())


#: A spreadsheet evaluates a cell that starts with one of these as a formula (tab and carriage
#: return can start one after the program strips them).
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def spreadsheet_safe(value: Any) -> Any:
    """``value`` as a CSV cell that a spreadsheet will not evaluate.

    A text cell that starts like a formula gets a leading apostrophe, which spreadsheets read as
    "this is text". Numbers, including negative ones, are returned unchanged.
    """
    if isinstance(value, str) and value.startswith(_FORMULA_PREFIXES):
        return "'" + value
    return value


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

    # Python 3.10's csv writer leaves a field with a bare CR unquoted (3.11 and later quote it).
    # A reader that takes a bare CR for a record break would then split the record, and the text
    # after the CR would start a record of its own, possibly a live formula. So when any cell holds
    # a CR, every cell that is not a number is quoted; numbers stay as they are.
    safe_rows = [{key: spreadsheet_safe(value) for key, value in row.items()} for row in rows]
    has_cr = any(isinstance(v, str) and "\r" in v for row in safe_rows for v in row.values())
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=columns,
        lineterminator="\n",
        quoting=csv.QUOTE_NONNUMERIC if has_cr else csv.QUOTE_MINIMAL,
    )
    writer.writeheader()
    writer.writerows(safe_rows)
    return buffer.getvalue()


def write_results(
    paths: RunPaths, manifest: RunManifest, recipe: Recipe, store: FitConfigStore
) -> AnalysisResult:
    """Materialize results.csv and results.json from the cache ledger, each written atomically (a
    link at either name is replaced, never written through)."""
    result = collect_results(paths, manifest, recipe, store)
    atomic_write(paths.results_csv, results_csv_text(result).encode("utf-8"))
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
