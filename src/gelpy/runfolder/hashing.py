"""Deterministic content hashing over validated pydantic models.

The cache keys hash the *validated* model's canonical JSON, not raw CLI args — so
type coercion (argv ``"0"`` vs JSON ``0``) and defaults are resolved before hashing.
Otherwise equivalent calls could silently miss the cache, or collide.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def canonical_json(value: BaseModel | dict[str, Any]) -> str:
    """Canonical JSON string: sorted keys, compact separators, JSON-mode dump."""
    if isinstance(value, BaseModel):
        payload = value.model_dump(mode="json")
    else:
        payload = value
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def content_hash(value: BaseModel | dict[str, Any]) -> str:
    """SHA-256 (hex) of the canonical JSON of ``value``."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def hash_fields(**fields: Any) -> str:
    """Content hash of an ad-hoc set of named fields (a plain dict)."""
    return content_hash(fields)


def file_hash(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 (hex) of a file's raw bytes, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()
