"""Load/save the manifest (cache ledger)."""

from __future__ import annotations

from ..schemas.run import RunManifest, StageRecord
from .cache import write_json
from .layout import RunPaths


def load_manifest(paths: RunPaths) -> RunManifest:
    if paths.manifest.exists():
        try:
            return RunManifest.model_validate_json(paths.manifest.read_text())
        except Exception:
            pass  # disposable ledger — rebuild rather than fail
    return RunManifest(recipe_hash="")


def save_manifest(paths: RunPaths, manifest: RunManifest) -> None:
    write_json(paths.manifest, manifest)


def record_stage(
    manifest: RunManifest,
    key: str,
    stage: str,
    stage_hash: str,
    cache_path: str,
    inputs: dict | None = None,
) -> None:
    manifest.stages[key] = StageRecord(
        stage=stage,
        key=key,
        hash=stage_hash,
        cache_path=cache_path,
        inputs=inputs or {},
    )
