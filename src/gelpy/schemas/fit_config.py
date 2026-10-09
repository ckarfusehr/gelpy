"""Per-lane fit configuration — the durable, editable record of fit intent.

Unlike the ``Recipe`` (which excludes fit config so one recipe can back several fit
passes) and the ``RunManifest`` (a disposable cache ledger), this store persists the
exact ``(profile, fit)`` last applied to each lane in its own precious
``fit_config.json``: it survives a manifest wipe, is hand/agent editable, and is the
surface an auto-tuner / escalation loop writes into.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from .baseline import BaselineConfig
from .common import CanonicalModel
from .profiles import LineProfileConfig

#: Default per-component shape menu the fitter is allowed to select from.
DEFAULT_SHAPES: tuple[str, ...] = ("gaussian", "emg", "bi_gaussian", "pseudo_voigt")


class PriorConfig(CanonicalModel):
    """Soft priors + stopping controls shared by every fit strategy."""

    width_prior_strength: float = Field(
        default=0.5,
        ge=0,
        description="Strength of the log-normal width prior around each seed's detected scale "
        "(0 = off). Keeps a component's width anchored to what was actually detected there "
        "instead of ballooning to absorb drift.",
    )
    tail_shrinkage: float = Field(
        default=0.3,
        ge=0,
        description="Shrinkage pulling EMG tail (tau) toward 0 (0 = off): a peak stays symmetric "
        "unless the data earns a tail.",
    )
    max_components: int = Field(default=12, ge=1, description="Hard cap on components per lane.")
    min_snr: float = Field(
        default=3.0,
        ge=0,
        description="A residual bump must exceed min_snr * noise_sigma to seed a new component.",
    )
    max_width_frac: float = Field(
        default=0.9,
        gt=0,
        le=1.0,
        description="A single component's sigma may not exceed this fraction of the window "
        "(prevents a 'peak' spanning the whole trace).",
    )


class FitConfig(CanonicalModel):
    """Peak-fit strategy selection + its shared knobs.

    Defaults are tuned to work with no manual tuning: ``fit run RUN`` uses grow_select with
    a conservative order-2 polynomial baseline and the full shape menu.
    """

    method: str = Field(default="grow_select", description="Fit strategy name (see `models list`).")
    baseline: BaselineConfig = Field(default_factory=BaselineConfig)
    shapes: list[str] = Field(
        default_factory=lambda: list(DEFAULT_SHAPES),
        description="Allowed peak shapes for per-component selection.",
    )
    priors: PriorConfig = Field(default_factory=PriorConfig)
    robust_loss: str = Field(
        default="soft_l1",
        description="Robust loss for least_squares: 'linear' | 'soft_l1' | 'huber' | 'cauchy'.",
    )
    seed: int = Field(
        default=0,
        ge=0,
        description="RNG seed; makes stochastic strategies (bayesian) reproducible and cacheable.",
    )
    params: dict[str, Any] = Field(
        default_factory=dict, description="Extra strategy-specific parameters."
    )


class PerLaneFitConfig(CanonicalModel):
    """The resolved fit intent for one lane: the exact profile + fit config applied."""

    profile: LineProfileConfig = Field(default_factory=LineProfileConfig)
    fit: FitConfig = Field(default_factory=FitConfig)


class FitConfigStore(CanonicalModel):
    """Durable per-lane fit configuration, persisted as ``fit_config.json`` (sparse)."""

    lanes: dict[int, PerLaneFitConfig] = Field(
        default_factory=dict, description="Per-lane fit intent, keyed by lane index."
    )
