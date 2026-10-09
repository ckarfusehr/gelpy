"""Content-addressed cache with atomic writes and corrupt-file recovery.

Not a generic argument-hashing memoizer (those would hash the multi-megabyte image
just to build a key). Callers compute a small content hash themselves and hand it
here as the cache filename; ``get_or_compute`` only invokes the (possibly expensive)
``compute`` thunk on a miss or when an existing file is unreadable/corrupt.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
from pydantic import BaseModel


def _atomic_write_bytes(path: Path, write: Callable[[Path], None]) -> None:
    """Write via a temp file in the same dir, then atomically rename into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def get_or_compute_npy(path: Path, compute: Callable[[], np.ndarray]) -> np.ndarray:
    if path.exists():
        try:
            return np.load(path)
        except (ValueError, OSError, EOFError):
            path.unlink(missing_ok=True)  # corrupt -> recompute
    value = compute()

    def _write(tmp: Path) -> None:
        with open(tmp, "wb") as fh:  # file object -> np.save won't append ".npy"
            np.save(fh, value)

    _atomic_write_bytes(path, _write)
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

    # np.savez appends ".npz" to a plain path, mangling the atomic temp name; pass
    # an open file object so the bytes land exactly where we tell them to.
    def _write(tmp: Path) -> None:
        with open(tmp, "wb") as fh:
            np.savez(fh, **value)

    _atomic_write_bytes(path, _write)
    return value


def get_or_compute_json(path: Path, compute: Callable[[], Any]) -> Any:
    if path.exists():
        try:
            return json.loads(path.read_text())
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
    _atomic_write_bytes(path, lambda tmp: tmp.write_text(text))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())
