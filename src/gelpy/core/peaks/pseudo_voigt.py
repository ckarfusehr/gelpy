"""Pseudo-Voigt peak — linear Gaussian/Lorentzian blend for sharp, heavy-winged peaks.

The Lorentzian HWHM is FWHM-matched to the Gaussian (``gamma = sigma*sqrt(2*ln2)``) so the
blend keeps a single width parameter plus ``eta`` (Lorentzian fraction). ``eta -> 0`` is a
Gaussian; larger ``eta`` sharpens the core and fattens the wings — a good match for the
well/pocket aggregate. Height at the center is ``amplitude`` for any ``eta``.
"""

from __future__ import annotations

import numpy as np

from ...schemas.peaks import PeakParams, PseudoVoigtPeakParams
from .base import SQRT_2PI, PeakModel, peak_registry

_FWHM_K = float(np.sqrt(2.0 * np.log(2.0)))  # gamma = sigma * sqrt(2 ln 2)


@peak_registry.register("pseudo_voigt", description="Gaussian/Lorentzian blend (sharp, heavy wings).")
class PseudoVoigtFitModel(PeakModel):
    kind = "pseudo_voigt"

    def params_per_peak(self) -> int:
        return 4

    def param_names(self) -> list[str]:
        return ["amplitude", "center", "sigma", "eta"]

    def param_roles(self) -> list[str]:
        return ["amplitude", "center", "width", "unit"]

    def single_peak(
        self, x: np.ndarray, amplitude: float, center: float, sigma: float, eta: float
    ) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        gamma = sigma * _FWHM_K
        gauss = np.exp(-((x - center) ** 2) / (2.0 * sigma**2))
        lorentz = gamma**2 / ((x - center) ** 2 + gamma**2)
        return amplitude * ((1.0 - eta) * gauss + eta * lorentz)

    def area(self, amplitude: float, center: float, sigma: float, eta: float) -> float:
        gamma = sigma * _FWHM_K
        return float(amplitude * ((1.0 - eta) * sigma * SQRT_2PI + eta * np.pi * gamma))

    def initial_guess(self, center: float, height: float, scale: float) -> list[float]:
        return [float(height), float(center), float(scale), 0.3]

    def build_params(self, params: tuple[float, ...]) -> PeakParams:
        amplitude, center, sigma, eta = params
        return PseudoVoigtPeakParams(amplitude=amplitude, center=center, sigma=sigma, eta=eta)
