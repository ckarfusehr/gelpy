"""Synthetic 1-D line-profile generator with ground-truth component areas.

Unlike ``generator.py`` (2-D images for lane *detection*), this makes the 1-D profiles the
peak *fitter* consumes: a structured (polynomial) baseline + heterogeneous components
(Gaussian / EMG / bi-Gaussian / pseudo-Voigt) + optional dust spikes + read noise, each with
an exactly known area (computed from the same shape models the fitter uses). Everything is
seeded, so a spec always yields the identical profile.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gelpy.core.peaks import peak_registry


@dataclass(frozen=True)
class TrueComponent:
    """One ground-truth component. ``cls`` is the expected label (sample/well/smear/dust)."""

    kind: str
    center: float
    amplitude: float
    sigma: float
    tau: float = 0.0
    sigma_right: float = 0.0
    eta: float = 0.0
    cls: str = "sample"

    def params(self) -> tuple[float, ...]:
        if self.kind == "gaussian":
            return (self.amplitude, self.center, self.sigma)
        if self.kind == "emg":
            return (self.amplitude, self.center, self.sigma, self.tau)
        if self.kind == "bi_gaussian":
            return (self.amplitude, self.center, self.sigma, self.sigma_right or self.sigma)
        if self.kind == "pseudo_voigt":
            return (self.amplitude, self.center, self.sigma, self.eta)
        raise ValueError(f"unknown kind {self.kind!r}")

    def area(self) -> float:
        return peak_registry.get(self.kind)().area(*self.params())


@dataclass(frozen=True)
class ProfileSpec:
    """Full parameterization of one synthetic profile. Randomness derives from ``seed``."""

    name: str
    length: int = 400
    seed: int = 0
    components: tuple[TrueComponent, ...] = ()
    baseline: tuple[float, ...] = (0.3,)  # polynomial coeffs (c0, c1, c2, ...) in px
    noise: float = 0.01
    n_dust: int = 0
    dust_amp: float = 0.4
    dust_sigma: float = 2.0
    tags: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class GeneratedProfile:
    profile: np.ndarray
    truth: tuple[tuple[float, float, str, str], ...]  # (center, area, cls, kind)
    a_sig: float  # total true signal area (sum of component areas)
    baseline: np.ndarray
    spec: ProfileSpec

    def true_fractions(self) -> list[tuple[float, float, str]]:
        """(center, area_fraction, cls) per true component."""
        return [(c, a / self.a_sig, cls) for (c, a, cls, _kind) in self.truth]


def generate(spec: ProfileSpec) -> GeneratedProfile:
    rng = np.random.default_rng(spec.seed)
    t = np.arange(spec.length, dtype=np.float64)
    baseline = np.zeros_like(t)
    for i, coeff in enumerate(spec.baseline):
        baseline = baseline + coeff * t**i

    signal = np.zeros_like(t)
    truth: list[tuple[float, float, str, str]] = []
    for tc in spec.components:
        model = peak_registry.get(tc.kind)()
        signal = signal + model.single_peak(t, *tc.params())
        maxx = model.peak_maximum_x(*tc.params())
        truth.append((float(maxx), tc.area(), tc.cls, tc.kind))

    gauss = peak_registry.get("gaussian")()
    for _ in range(spec.n_dust):
        center = float(rng.uniform(0.05, 0.95) * spec.length)
        amp = spec.dust_amp * (1.0 + rng.uniform(-0.3, 0.3))
        signal = signal + gauss.single_peak(t, amp, center, spec.dust_sigma)
        truth.append((center, gauss.area(amp, center, spec.dust_sigma), "dust", "gaussian"))

    profile = baseline + signal + rng.normal(0.0, spec.noise, spec.length)
    a_sig = float(sum(a for _c, a, _cls, _k in truth))
    return GeneratedProfile(
        profile=profile, truth=tuple(truth), a_sig=a_sig, baseline=baseline, spec=spec
    )
