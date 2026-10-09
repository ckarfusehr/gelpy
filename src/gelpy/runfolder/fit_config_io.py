"""Load/save the per-lane fit-config store (``fit_config.json``)."""

from __future__ import annotations

from ..schemas.fit_config import FitConfigStore
from .cache import write_json
from .layout import RunPaths


def load_fit_config(paths: RunPaths) -> FitConfigStore:
    """Load the durable per-lane fit config.

    Unlike the disposable manifest, this is precious input state, so a parse error is
    surfaced (fail loud) rather than silently rebuilt — corrupt intent must never be
    masked. A missing file is a fresh, empty store.
    """
    if paths.fit_config.exists():
        return FitConfigStore.model_validate_json(paths.fit_config.read_text())
    return FitConfigStore()


def save_fit_config(paths: RunPaths, store: FitConfigStore) -> None:
    write_json(paths.fit_config, store)
