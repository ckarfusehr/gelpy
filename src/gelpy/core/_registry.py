"""A tiny generic registry, shared by the background and peak model families.

Replaces the old ``if/elif`` string-dispatch (and its ``print("No valid model
selected")``-and-continue bug) with a single mechanism that raises a structured
error listing the available names and can emit JSON schemas for discovery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from ..schemas.common import CanonicalModel
from .errors import RegistryError, shown

T = TypeVar("T")


@dataclass
class _Entry(Generic[T]):
    factory: T
    config_model: type[CanonicalModel] | None
    description: str


class Registry(Generic[T]):
    """Maps a string name to a factory plus an optional pydantic config model."""

    def __init__(self, kind: str):
        self.kind = kind
        self._entries: dict[str, _Entry[T]] = {}

    def register(
        self,
        name: str,
        *,
        config_model: type[CanonicalModel] | None = None,
        description: str = "",
    ):
        """Decorator: register a factory (class or callable) under ``name``."""

        def _decorator(factory: T) -> T:
            if name in self._entries:
                raise ValueError(f"{self.kind} model {name!r} already registered")
            self._entries[name] = _Entry(factory, config_model, description)
            return factory

        return _decorator

    def get(self, name: str) -> T:
        try:
            return self._entries[name].factory
        except KeyError:
            raise RegistryError(
                f"Unknown {self.kind} model {shown(name)}.", available=self.names()
            ) from None

    def config_model(self, name: str) -> type[CanonicalModel] | None:
        self.get(name)  # raises if unknown
        return self._entries[name].config_model

    def names(self) -> list[str]:
        return sorted(self._entries)

    def json_schemas(self) -> dict[str, dict]:
        """Name -> JSON schema of its config model (empty dict if none)."""
        out: dict[str, dict] = {}
        for name, entry in self._entries.items():
            out[name] = (
                entry.config_model.model_json_schema() if entry.config_model else {}
            )
        return out

    def descriptions(self) -> dict[str, str]:
        return {name: entry.description for name, entry in self._entries.items()}
