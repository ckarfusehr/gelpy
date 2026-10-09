"""The fit-strategy registry (its own module so strategies can import it without a cycle)."""

from __future__ import annotations

from ..._registry import Registry

fit_strategy_registry: Registry = Registry("fit-strategy")
