"""Gaussian peak model — symmetric band."""

from __future__ import annotations

import numpy as np

from ...schemas.peaks import GaussianPeakParams, PeakParams
from .base import SQRT_2PI, PeakModel, peak_registry


@peak_registry.register("gaussian", description="Symmetric Gaussian peak.")
class GaussianFitModel(PeakModel):
    kind = "gaussian"

    def params_per_peak(self) -> int:
        return 3

    def param_names(self) -> list[str]:
        return ["amplitude", "center", "sigma"]

    def param_roles(self) -> list[str]:
        return ["amplitude", "center", "width"]

    def single_peak(self, x: np.ndarray, amplitude: float, center: float, sigma: float) -> np.ndarray:
        return amplitude * np.exp(-((x - center) ** 2) / (2.0 * sigma**2))

    def area(self, amplitude: float, center: float, sigma: float) -> float:
        return float(amplitude * sigma * SQRT_2PI)

    def initial_guess(self, center: float, height: float, scale: float) -> list[float]:
        return [float(height), float(center), float(scale)]

    def build_params(self, params: tuple[float, ...]) -> PeakParams:
        amplitude, center, sigma = params
        return GaussianPeakParams(amplitude=amplitude, center=center, sigma=sigma)
