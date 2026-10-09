"""Exponentially-modified-Gaussian (EMG) peak — a Gaussian with a one-sided tail.

Parameterized as a Gaussian of height ``amplitude`` convolved with a unit-area one-sided
exponential of scale ``tau``, so ``area == amplitude * sigma * sqrt(2*pi)`` for every
``tau`` and the ``tau -> 0`` limit is exactly ``gaussian(amplitude, center, sigma)``. The
evaluation uses ``erfcx`` (scaled complementary error function) so the exp*erfc product
never overflows — the old implementation's overflow guards are unnecessary here.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erfc, erfcx

from ...schemas.peaks import EmgPeakParams, PeakParams
from .base import SQRT_2PI, PeakModel, peak_registry

# Below this tau (px) the EMG is numerically indistinguishable from a Gaussian; evaluate
# the Gaussian directly to avoid a 1/tau blow-up.
_TAU_FLOOR = 1e-6


@peak_registry.register("emg", description="Exponentially-modified Gaussian (right-tailed) peak.")
class EmgFitModel(PeakModel):
    kind = "emg"

    def params_per_peak(self) -> int:
        return 4

    def param_names(self) -> list[str]:
        return ["amplitude", "center", "sigma", "tau"]

    def param_roles(self) -> list[str]:
        return ["amplitude", "center", "width", "tail"]

    def single_peak(
        self, x: np.ndarray, amplitude: float, center: float, sigma: float, tau: float
    ) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if abs(tau) <= _TAU_FLOOR:
            return amplitude * np.exp(-((x - center) ** 2) / (2.0 * sigma**2))
        if tau < 0.0:
            # Left tail = the right-tail EMG mirrored about the center.
            return self.single_peak(2.0 * center - x, amplitude, center, sigma, -tau)
        # EMG = amplitude*sigma*sqrt(2pi) * f(x). Evaluate each branch only where it is
        # numerically stable: erfcx form for z>=0 (core/left flank), plain exp*erfc for
        # z<0 (right tail). Splitting avoids the far-tail 0*inf that either form alone hits.
        scalar = x.ndim == 0
        xf = np.atleast_1d(x)
        pref = amplitude * sigma * SQRT_2PI / (2.0 * tau)
        z = ((center - xf) / sigma + sigma / tau) / np.sqrt(2.0)
        out = np.empty_like(xf)
        left = z >= 0.0
        xl = xf[left]
        out[left] = pref * np.exp(-((xl - center) ** 2) / (2.0 * sigma**2)) * erfcx(z[left])
        xr = xf[~left]
        a_right = sigma**2 / (2.0 * tau**2) + (center - xr) / tau
        out[~left] = pref * np.exp(a_right) * erfc(z[~left])
        return float(out[0]) if scalar else out

    def area(self, amplitude: float, center: float, sigma: float, tau: float) -> float:
        return float(amplitude * sigma * SQRT_2PI)

    def peak_maximum_x(self, amplitude: float, center: float, sigma: float, tau: float) -> float:
        # Mode sits between `center` and the mean on the tail side; the span brackets it for
        # either tail direction.
        span = 6.0 * sigma + 4.0 * abs(tau)
        return self._grid_argmax((amplitude, center, sigma, tau), center, span)

    def effective_width(self, amplitude: float, center: float, sigma: float, tau: float) -> float:
        return float(np.hypot(sigma, tau))

    def initial_guess(self, center: float, height: float, scale: float) -> list[float]:
        return [float(height), float(center), float(scale), 0.5 * float(scale)]

    def build_params(self, params: tuple[float, ...]) -> PeakParams:
        amplitude, center, sigma, tau = params
        return EmgPeakParams(amplitude=amplitude, center=center, sigma=sigma, tau=tau)
