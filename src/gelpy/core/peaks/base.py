"""Peak-model protocol and registry.

A ``PeakModel`` fully describes one peak shape: how to evaluate it, its closed-form
area, where its maximum sits, how to seed a fit from a detected ``(center, height,
scale)``, the *role* of each parameter (so the shared driver builds bounds and priors
generically, without branching on the model), and how to package fitted values into a
pydantic ``PeakParams``. The driver never branches on a model's name.

Parameterization is chosen so shape-selection is continuous: EMG with ``tau -> 0`` and
pseudo-Voigt with ``eta -> 0`` both reduce to the Gaussian of the same
``amplitude``/``center``/``sigma``. ``amplitude`` is the peak height for every shape
except that EMG's height is spread by the tail (its ``amplitude`` is the Gaussian
component's height); ``area`` gives the true integral in every case.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ...schemas.peaks import PeakParams
from .._registry import Registry

peak_registry: Registry[type["PeakModel"]] = Registry("peak-shape")

SQRT_2PI = float(np.sqrt(2.0 * np.pi))

#: Parameter roles the driver understands. "amplitude" -> non-negative height;
#: "center" -> position (bounded near its seed); "width" -> spatial scale (positivity +
#: width prior + max-width cap); "tail" -> non-negative tail length (shrinkage prior);
#: "unit" -> bounded to [0, 1].
ParamRole = str


class PeakModel(ABC):
    """Interface every peak shape implements."""

    kind: str

    @abstractmethod
    def params_per_peak(self) -> int:
        """Number of fit parameters per peak."""

    @abstractmethod
    def param_names(self) -> list[str]:
        """Parameter attribute names, in fit order."""

    @abstractmethod
    def param_roles(self) -> list[ParamRole]:
        """Role of each parameter (same order/length as ``param_names``)."""

    @abstractmethod
    def single_peak(self, x: np.ndarray, *params: float) -> np.ndarray:
        """Evaluate one peak at ``x`` (non-negative everywhere for non-negative amplitude)."""

    @abstractmethod
    def area(self, *params: float) -> float:
        """Closed-form integral of one peak over the whole real line."""

    @abstractmethod
    def initial_guess(self, center: float, height: float, scale: float) -> list[float]:
        """Seed parameters for a component detected at ``center`` with ``height`` and ``scale``."""

    @abstractmethod
    def build_params(self, params: tuple[float, ...]) -> PeakParams:
        """Package one peak's fitted values into a ``PeakParams`` model."""

    def peak_maximum_x(self, *params: float) -> float:
        """x-position of this peak's maximum. Defaults to the ``center`` parameter."""
        return float(params[self._role_index("center")])

    def effective_width(self, *params: float) -> float:
        """A single sigma-equivalent width for classification (dust vs band vs smear).

        Defaults to the first ``width``-role parameter; shapes with a tail or split widths
        override to fold those in.
        """
        idx = self.role_indices("width")
        return float(params[idx[0]]) if idx else 1.0

    # --- role helpers (used by the driver) ------------------------------------
    def _role_index(self, role: ParamRole) -> int:
        return self.param_roles().index(role)

    def role_indices(self, role: ParamRole) -> list[int]:
        return [i for i, r in enumerate(self.param_roles()) if r == role]

    def multi_peak(self, x: np.ndarray, *params: float) -> np.ndarray:
        """Sum of ``len(params)/params_per_peak`` peaks of THIS model (same shape)."""
        result = np.zeros_like(x, dtype=np.float64)
        ppp = self.params_per_peak()
        for i in range(0, len(params), ppp):
            result = result + self.single_peak(x, *params[i : i + ppp])
        return result

    def _grid_argmax(self, params: tuple[float, ...], center: float, span: float) -> float:
        """Maximum x by dense evaluation over ``center +/- span`` (for asymmetric shapes)."""
        grid = np.linspace(center - span, center + span, 2001)
        y = self.single_peak(grid, *params)
        return float(grid[int(np.argmax(y))])
