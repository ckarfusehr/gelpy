"""Split (bi-)Gaussian peak — independent left/right widths for cheap asymmetry.

Useful for skewed bands and broad smears where a single sigma can't match both flanks.
Continuous at the center (both halves reach ``amplitude`` there). area is exact:
``amplitude * sqrt(pi/2) * (sigma_left + sigma_right)``.
"""

from __future__ import annotations

import numpy as np

from ...schemas.peaks import BiGaussianPeakParams, PeakParams
from .base import SQRT_2PI, PeakModel, peak_registry


@peak_registry.register("bi_gaussian", description="Split Gaussian with independent left/right widths.")
class BiGaussianFitModel(PeakModel):
    kind = "bi_gaussian"

    def params_per_peak(self) -> int:
        return 4

    def param_names(self) -> list[str]:
        return ["amplitude", "center", "sigma_left", "sigma_right"]

    def param_roles(self) -> list[str]:
        return ["amplitude", "center", "width", "width"]

    def single_peak(
        self,
        x: np.ndarray,
        amplitude: float,
        center: float,
        sigma_left: float,
        sigma_right: float,
    ) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        sigma = np.where(x < center, sigma_left, sigma_right)
        return amplitude * np.exp(-((x - center) ** 2) / (2.0 * sigma**2))

    def area(
        self, amplitude: float, center: float, sigma_left: float, sigma_right: float
    ) -> float:
        # Each half contributes half a Gaussian's area: amplitude*sigma*sqrt(2pi)/2.
        return float(amplitude * SQRT_2PI / 2.0 * (sigma_left + sigma_right))

    def effective_width(
        self, amplitude: float, center: float, sigma_left: float, sigma_right: float
    ) -> float:
        return 0.5 * (sigma_left + sigma_right)

    def initial_guess(self, center: float, height: float, scale: float) -> list[float]:
        return [float(height), float(center), float(scale), float(scale)]

    def build_params(self, params: tuple[float, ...]) -> PeakParams:
        amplitude, center, sigma_left, sigma_right = params
        return BiGaussianPeakParams(
            amplitude=amplitude, center=center, sigma_left=sigma_left, sigma_right=sigma_right
        )
