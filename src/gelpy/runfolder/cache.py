"""Content-addressed cache with atomic writes and corrupt-file recovery.

Not a generic argument-hashing memoizer (those would hash the multi-megabyte image
just to build a key). Callers compute a small content hash themselves and hand it
here as the cache filename; ``get_or_compute`` only invokes the (possibly expensive)
``compute`` thunk on a miss or when an existing file is unreadable/corrupt.

Every file goes through ``utils.atomic.atomic_write``, the one atomic writer of gelpy: a temp
file, then a rename. ``np.save`` and ``np.savez`` are handed its open file, so they cannot
append an extension to the temp name.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from pydantic import BaseModel

from ..utils.atomic import atomic_write


def get_or_compute_npy(path: Path, compute: Callable[[], np.ndarray]) -> np.ndarray:
    if path.exists():
        try:
            return np.load(path)
        except (ValueError, OSError, EOFError):
            path.unlink(missing_ok=True)  # corrupt -> recompute
    value = compute()
    atomic_write(path, lambda fh: np.save(fh, value))  # a file object: np.save adds no ".npy"
    return value


def get_or_compute_npz(
    path: Path, compute: Callable[[], dict[str, np.ndarray]]
) -> dict[str, np.ndarray]:
    if path.exists():
        try:
            with np.load(path) as data:
                return {k: data[k] for k in data.files}
        except (ValueError, OSError, EOFError):
            path.unlink(missing_ok=True)
    value = compute()
    atomic_write(path, lambda fh: np.savez(fh, **value))  # a file object: np.savez adds no ".npz"
    return value


def get_or_compute_json(path: Path, compute: Callable[[], Any]) -> Any:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            path.unlink(missing_ok=True)
    value = compute()
    write_json(path, value)
    return value


def write_json(path: Path, value: Any) -> None:
    if isinstance(value, BaseModel):
        text = value.model_dump_json(indent=2)
    else:
        text = json.dumps(value, indent=2, sort_keys=True, default=str)
    atomic_write(path, text.encode("utf-8"))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
