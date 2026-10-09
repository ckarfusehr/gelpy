"""Fit strategies (the tier ladder), selected by name via ``fit_strategy_registry``.

Mirrors the background / projection-strategy registries: a name maps to a callable
``fit(profile, config, *, lane_index, label) -> LaneFitResult``.
"""

from __future__ import annotations

from .registry import fit_strategy_registry

# Register the built-in strategies by importing their modules.
from . import grow_select as _grow_select  # noqa: F401,E402
from . import sparse_seed as _sparse_seed  # noqa: F401,E402
from . import bayesian as _bayesian  # noqa: F401,E402

__all__ = ["fit_strategy_registry"]
