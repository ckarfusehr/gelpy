"""Background-model protocol, result container, and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ...schemas.common import CanonicalModel
from .._registry import Registry


@dataclass
class BackgroundFitResult:
    """Outputs of a background fit. Holds ndarrays (not persisted through pydantic)."""

    corrected_image: np.ndarray
    background_plane: np.ndarray
    #: Model-specific scalar parameters (e.g. plane coefficients), JSON-serializable.
    params: dict[str, float] = field(default_factory=dict)
    #: Fraction of fit samples RANSAC (or equivalent) kept as inliers, if applicable.
    inlier_fraction: float | None = None
    #: Per-pixel background trust in ``[0, 1]`` (1 = treated as background). ``None`` for
    #: models without a soft weighting. Not cached — used by the diagnostic renderer to
    #: show which pixels the fit trusted (i.e. did it correctly ignore bands/smears?).
    weights: np.ndarray | None = None


class BackgroundModel(ABC):
    """Interface every background model implements."""

    #: Registered name.
    name: str

    @abstractmethod
    def remove(self, image: np.ndarray, params: CanonicalModel) -> BackgroundFitResult:
        """Fit the background on ``image`` and return the corrected image + diagnostics."""


background_registry: Registry[type[BackgroundModel]] = Registry("background")
