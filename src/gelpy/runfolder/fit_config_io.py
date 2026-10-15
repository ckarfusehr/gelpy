"""Load/save the per-lane fit-config store (``fit_config.json``)."""

from __future__ import annotations

from ..core.errors import shown
from ..core.text import decode_utf8
from ..schemas.fit_config import FitConfigStore
from .cache import write_json
from .layout import RunPaths


def load_fit_config(paths: RunPaths) -> FitConfigStore:
    """Load the durable per-lane fit config.

    Unlike the disposable manifest, this is precious input state, so a parse error is
    surfaced (fail loud) rather than silently rebuilt — corrupt intent must never be
    masked. A missing file is a fresh, empty store. A file that is not UTF-8 (saved by an editor
    in another encoding) is an ``InvalidValueError`` that names the file, the line and the encoding.
    """
    if paths.fit_config.exists():
        text = decode_utf8(
            paths.fit_config.read_bytes(),
            what=f"The fit configuration {shown(str(paths.fit_config))}",
            detail="Save fit_config.json as UTF-8 text and repeat the command.",
        )
        return FitConfigStore.model_validate_json(text)
    return FitConfigStore()


def save_fit_config(paths: RunPaths, store: FitConfigStore) -> None:
    write_json(paths.fit_config, store)
