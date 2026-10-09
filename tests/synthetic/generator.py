"""Parametric synthetic gel-image generator with ground-truth lane centers.

Purpose: exercise (and score) adaptive y-band lane detection. Unlike the inline
``_synthetic_gel`` helpers in ``tests/core/test_lanes.py`` — whose bands span the FULL
image height (constant in y) so a full-height column projection is already optimal — this
generator models realistic **y-localized** bands: a band is wide in x (a fraction of the
lane pitch) and thin in y (a horizontal streak at a migration distance). That leaves empty
rows above/below the migration region, which is exactly what makes an adaptive y-band choice
matter and what full-height projection dilutes.

Intensity convention matches the detection contract and the existing tests: **bands are
bright positive bumps on a low background** (the array is already in "loaded lanes are bright
columns" orientation), floats roughly in [0, ~1+]. ``detect_lane_centers`` normalizes, so the
absolute scale is not important.

Everything is seeded, so a given ``GelSpec`` always yields the identical image — branches are
compared on byte-identical inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np


@dataclass(frozen=True)
class GelSpec:
    """Full parameterization of one synthetic gel. All randomness is derived from ``seed``."""

    name: str
    seed: int = 0
    n_lanes: int = 12
    width: int = 1400
    height: int = 500

    # --- lane geometry -------------------------------------------------------------------
    # Fraction of the image width the N lanes occupy, centered. 1.0 == crop-tight (pitch ==
    # width / n_lanes, the case the detector's width/N prior assumes). < 1.0 models a gel that
    # fills only part of the crop (stresses the prior; expected to read as low confidence).
    fill_frac: float = 1.0

    # --- band geometry -------------------------------------------------------------------
    band_x_sigma_frac: float = 0.26  # x-sigma of a band, as a fraction of the pitch (wide-in-x)
    band_y_sigma: float = 6.0        # y-sigma of a band, px (thin-in-y streak)
    band_amp: float = 1.0
    # Relative y-positions (0..1 of height) at which every populated lane gets a band. One
    # entry == single band per lane; several == a ladder / multi-band lane.
    band_rows: tuple[float, ...] = (0.5,)
    band_y_jitter: float = 0.0       # px std of independent per-(lane,row) band-y wobble

    # Per-lane single-band migration override (0..1 of height), length n_lanes. When set it
    # replaces ``band_rows`` with one band per lane at these y's — models lanes whose one band
    # migrated to different distances (no single narrow y-window captures them all).
    per_lane_band_y: tuple[float, ...] | None = None

    smile_px: float = 0.0            # extra downward band-y shift at the left/right edges (parabola)

    # --- lane population -----------------------------------------------------------------
    empty_lanes: tuple[int, ...] = ()          # indices with no bands at all
    faint_lanes: tuple[int, ...] = ()          # indices whose bands are scaled by faint_factor
    faint_factor: float = 0.22

    # --- dust ----------------------------------------------------------------------------
    n_dust: int = 0
    dust_sigma: float = 2.5          # px, circular (narrow in BOTH x and y — the band-vs-dust signal)
    dust_amp: float = 0.9
    dust_amp_jitter: float = 0.3     # multiplicative amp spread (uniform +/-)
    dust_in_band_only: bool = False  # if True, dust is confined to the band y-range (worst case)

    # --- nuisances -----------------------------------------------------------------------
    bg: float = 0.05
    gradient: tuple[float, float] = (0.0, 0.0)  # (x-slope, y-slope) per px, added as a plane
    noise_sigma: float = 0.0                    # additive gaussian read noise

    tags: tuple[str, ...] = field(default_factory=tuple)  # free-form labels for the harness


@dataclass(frozen=True)
class SynthGel:
    """A generated gel plus its ground truth."""

    image: np.ndarray             # (H, W) float64, bands bright
    centers: np.ndarray           # (N,) true lane x-centers (all N, incl. empty lanes)
    pitch: float                  # true center-to-center spacing (px)
    band_y_lo: float              # top of the informative band region (px), for review/overlap
    band_y_hi: float              # bottom of the informative band region (px)
    empty_lanes: tuple[int, ...]
    spec: GelSpec


def _lane_centers(spec: GelSpec) -> tuple[np.ndarray, float]:
    """True lane x-centers and pitch. Crop-tight (fill_frac==1) gives pitch == width/N."""
    n, w = spec.n_lanes, spec.width
    if n < 1:
        raise ValueError("n_lanes must be >= 1")
    if n == 1:
        return np.array([w / 2.0]), float(w)
    pitch = spec.fill_frac * w / n
    block = n * pitch
    start = (w - block) / 2.0 + pitch / 2.0
    return start + np.arange(n) * pitch, float(pitch)


def generate(spec: GelSpec) -> SynthGel:
    """Render ``spec`` to a ``SynthGel`` (image + ground truth). Deterministic in ``spec.seed``."""
    rng = np.random.default_rng(spec.seed)
    h, w = spec.height, spec.width
    centers, pitch = _lane_centers(spec)
    sx = spec.band_x_sigma_frac * pitch
    sy = spec.band_y_sigma

    x = np.arange(w, dtype=np.float64)
    y = np.arange(h, dtype=np.float64)
    img = np.full((h, w), float(spec.bg), dtype=np.float64)

    # Background plane (illumination gradient).
    gx, gy = spec.gradient
    if gx or gy:
        img += gx * x[None, :] + gy * y[:, None]

    # Which (relative-y) bands each lane gets.
    if spec.per_lane_band_y is not None:
        if len(spec.per_lane_band_y) != spec.n_lanes:
            raise ValueError("per_lane_band_y must have length n_lanes")
        lane_rows = [(ry,) for ry in spec.per_lane_band_y]
    else:
        lane_rows = [spec.band_rows] * spec.n_lanes

    band_ys: list[float] = []
    empty = set(spec.empty_lanes)
    faint = set(spec.faint_lanes)
    for i, cx in enumerate(centers):
        if i in empty:
            continue
        amp_i = spec.band_amp * (spec.faint_factor if i in faint else 1.0)
        smile = spec.smile_px * ((cx - w / 2.0) / (w / 2.0)) ** 2
        gauss_x = np.exp(-((x - cx) ** 2) / (2.0 * sx**2))
        for ry in lane_rows[i]:
            cy = ry * h + smile + rng.normal(0.0, spec.band_y_jitter)
            band_ys.append(cy)
            gauss_y = np.exp(-((y - cy) ** 2) / (2.0 * sy**2))
            img += amp_i * np.outer(gauss_y, gauss_x)

    # Informative band region (for review overlay + later y-band overlap scoring). Padded by
    # ~2 sigma so it brackets the visible streaks; falls back to full height if no bands.
    if band_ys:
        band_y_lo = max(0.0, min(band_ys) - 2 * sy - spec.smile_px)
        band_y_hi = min(float(h), max(band_ys) + 2 * sy + spec.smile_px)
    else:
        band_y_lo, band_y_hi = 0.0, float(h)

    # Dust: narrow circular blobs. Confined to the band y-range in the worst case, else anywhere.
    for _ in range(spec.n_dust):
        dx = rng.uniform(0, w)
        if spec.dust_in_band_only:
            dy = rng.uniform(band_y_lo, band_y_hi)
        else:
            dy = rng.uniform(0, h)
        amp = spec.dust_amp * (1.0 + rng.uniform(-spec.dust_amp_jitter, spec.dust_amp_jitter))
        gx_d = np.exp(-((x - dx) ** 2) / (2.0 * spec.dust_sigma**2))
        gy_d = np.exp(-((y - dy) ** 2) / (2.0 * spec.dust_sigma**2))
        img += amp * np.outer(gy_d, gx_d)

    if spec.noise_sigma:
        img += rng.normal(0.0, spec.noise_sigma, size=img.shape)

    return SynthGel(
        image=img,
        centers=centers,
        pitch=pitch,
        band_y_lo=band_y_lo,
        band_y_hi=band_y_hi,
        empty_lanes=tuple(sorted(empty)),
        spec=spec,
    )


def with_seed(spec: GelSpec, seed: int) -> GelSpec:
    """Return a copy of ``spec`` with a new seed (for repeat-draw variance in the harness)."""
    return replace(spec, seed=seed)
