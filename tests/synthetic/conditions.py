"""The named synthetic-gel suite — one ``GelSpec`` per condition the detector must survive.

Grouped by what they stress. Two calibration facts drive the amplitudes/noise here (measured
against the current full-height detector on this generator):

  1. **Noise alone barely hurts** when signal is strong — the matched-filter grid + width/N
     prior recover accurate centers even at high noise. The difficulty axis that matters is
     **signal amplitude relative to noise** (SNR). The cliff sits near band amp ~0.15-0.30 at
     read-noise sigma ~0.12.
  2. In that marginal regime, full-height and a band-restricted projection reach similar center
     *accuracy*, but differ sharply in **detectability**: full-height flips to ``confidence
     "low"`` (peaks buried) while the band region stays ``"ok"``. So the suite is tuned to sit
     ON the cliff — where the methods actually diverge — not far above it (too easy) or far
     below (impossible, no discrimination).

Every condition therefore carries real read noise (``_NOISE``), and signal is pulled down into
the marginal band. ``snr_*`` is an explicit increasing-signal ladder at fixed noise. The lone
trivial floor is ``control_clean_thin``.

``SUITE`` is the canonical ordered list the harness and the review contact-sheet iterate.
"""

from __future__ import annotations

from .generator import GelSpec

# Base read-noise sigma applied to nearly every condition (calibrated to bite in the marginal
# regime without swamping strong signal). A couple of cases override it up (harder) or down.
_NOISE = 0.12

# Per-lane migration for the "spread" case: each lane's single band lands at a different depth,
# so no single narrow y-window captures them all (stresses window-search / multi-band union).
_SPREAD_12 = tuple(0.25 + 0.5 * (i / 11) for i in range(12))


def _snr_ladder() -> tuple[GelSpec, ...]:
    """Increasing-signal ladder: fixed noise, one thin mid band, band amp climbing across the
    cliff. Lets the harness plot each design's accuracy/confidence vs SNR on identical geometry.
    """
    amps = (0.15, 0.20, 0.28, 0.40, 0.60, 1.00)
    out = []
    for k, amp in enumerate(amps):
        out.append(
            GelSpec(
                name=f"snr_{int(amp * 100):03d}",
                seed=30 + k,
                band_rows=(0.5,),
                band_y_sigma=6.0,
                band_amp=amp,
                noise_sigma=_NOISE,
                tags=("snr_ladder", "core"),
            )
        )
    return tuple(out)


SUITE: tuple[GelSpec, ...] = (
    *_snr_ladder(),
    # --- core SNR family: thin bands near the cliff (adaptive band should recover confidence) --
    GelSpec(
        name="narrowband_mid",
        seed=1,
        band_rows=(0.5,),
        band_y_sigma=6.0,
        band_amp=0.28,
        noise_sigma=_NOISE,
        tags=("narrowband", "core"),
    ),
    GelSpec(
        name="narrowband_top",  # migration barely left the wells — band in the top third
        seed=2,
        band_rows=(0.22,),
        band_y_sigma=6.0,
        band_amp=0.28,
        noise_sigma=_NOISE,
        tags=("narrowband", "offcenter", "core"),
    ),
    GelSpec(
        name="narrowband_hard",  # thin band + heavier noise + low signal: full-height buried
        seed=3,
        band_rows=(0.55,),
        band_y_sigma=5.0,
        band_amp=0.25,
        noise_sigma=0.18,
        tags=("narrowband", "noise", "core", "hard"),
    ),
    # --- controls ---------------------------------------------------------------------------
    GelSpec(
        name="control_fullheight",  # thick bands spanning most of the height (old-style)
        seed=4,
        band_rows=(0.5,),
        band_y_sigma=120.0,
        band_amp=0.6,
        noise_sigma=0.08,
        tags=("control", "fullheight"),
    ),
    GelSpec(
        name="control_clean_thin",  # thin band, strong signal, ~no noise — the trivial floor
        seed=5,
        band_rows=(0.5,),
        band_y_sigma=6.0,
        band_amp=1.0,
        noise_sigma=0.01,
        tags=("control",),
    ),
    # --- multi-band / ladder ----------------------------------------------------------------
    GelSpec(
        name="ladder_4band",
        seed=6,
        band_rows=(0.30, 0.45, 0.62, 0.80),
        band_y_sigma=5.0,
        band_amp=0.3,
        noise_sigma=_NOISE,
        tags=("multiband", "ladder"),
    ),
    GelSpec(
        name="spread_migration",  # one band per lane, each at a different depth (diagonal-ish)
        seed=7,
        per_lane_band_y=_SPREAD_12,
        band_y_sigma=6.0,
        band_amp=0.30,
        noise_sigma=0.10,
        tags=("multiband", "spread"),
    ),
    # --- faint lanes ------------------------------------------------------------------------
    GelSpec(
        name="faint_subset",
        seed=8,
        band_rows=(0.5,),
        band_amp=0.45,
        faint_lanes=(1, 4, 8, 10),
        faint_factor=0.3,  # faint lanes ~0.18 amp: on the cliff
        band_y_sigma=6.0,
        noise_sigma=_NOISE,
        tags=("faint",),
    ),
    GelSpec(
        name="faint_right_cluster",  # mirrors a real-gel regression: right third faint
        seed=9,
        band_rows=(0.5,),
        band_amp=0.45,
        faint_lanes=(8, 9, 10, 11),
        faint_factor=0.28,
        band_y_sigma=6.0,
        noise_sigma=_NOISE,
        tags=("faint",),
    ),
    # --- empty lanes ------------------------------------------------------------------------
    GelSpec(
        name="empty_edge",  # unloaded lane 0 (edge) — the empty-edge-well placement case
        seed=10,
        band_rows=(0.5,),
        band_amp=0.30,
        empty_lanes=(0,),
        band_y_sigma=6.0,
        noise_sigma=_NOISE,
        tags=("empty",),
    ),
    GelSpec(
        name="empty_edge_and_interior",
        seed=11,
        band_rows=(0.5,),
        band_amp=0.30,
        empty_lanes=(0, 5, 6),
        band_y_sigma=6.0,
        noise_sigma=_NOISE,
        tags=("empty",),
    ),
    # --- dust (narrow circular blobs; brightness comparable to a band, rejected by width/shape)--
    GelSpec(
        name="dusty",
        seed=12,
        band_rows=(0.5,),
        band_amp=0.32,
        n_dust=40,
        dust_sigma=2.5,
        dust_amp=0.32,
        band_y_sigma=6.0,
        noise_sigma=0.10,
        tags=("dust",),
    ),
    GelSpec(
        name="dusty_in_band",  # dust confined to the band y-range — hardest to reject by y alone
        seed=13,
        band_rows=(0.5,),
        band_amp=0.32,
        n_dust=30,
        dust_sigma=2.5,
        dust_amp=0.36,
        dust_in_band_only=True,
        band_y_sigma=6.0,
        noise_sigma=0.10,
        tags=("dust", "hard"),
    ),
    GelSpec(
        name="dusty_faint",  # faint lanes AND dust: band-vs-dust discrimination under low SNR
        seed=14,
        band_rows=(0.5,),
        band_amp=0.42,
        faint_lanes=(2, 5, 9),
        faint_factor=0.3,
        n_dust=35,
        dust_sigma=2.5,
        dust_amp=0.36,
        band_y_sigma=6.0,
        noise_sigma=_NOISE,
        tags=("dust", "faint", "hard"),
    ),
    # --- geometry / illumination nuisances --------------------------------------------------
    GelSpec(
        name="smile",  # bands curve down at the edges; a narrow y-window misses edge lanes
        seed=15,
        band_rows=(0.5,),
        band_amp=0.30,
        smile_px=55.0,
        band_y_sigma=6.0,
        noise_sigma=0.10,
        tags=("smile", "geometry"),
    ),
    GelSpec(
        name="gradient_bg",  # strong diagonal illumination gradient under a thin band
        seed=16,
        band_rows=(0.5,),
        band_amp=0.45,
        gradient=(0.0004, 0.0006),
        band_y_sigma=6.0,
        noise_sigma=0.10,
        tags=("gradient",),
    ),
    GelSpec(
        name="partial_fill",  # gel fills only ~55% of the crop — should read low-confidence
        seed=17,
        fill_frac=0.55,
        band_rows=(0.5,),
        band_amp=0.5,
        band_y_sigma=6.0,
        noise_sigma=0.10,
        tags=("partial", "lowconf"),
    ),
    # --- lane-count variants ----------------------------------------------------------------
    GelSpec(
        name="n8_wide",
        seed=18,
        n_lanes=8,
        width=1100,
        band_rows=(0.5,),
        band_amp=0.30,
        band_y_sigma=6.0,
        noise_sigma=_NOISE,
        tags=("count",),
    ),
    GelSpec(
        name="n20_realistic",  # folding_screen-like: 20 lanes, thin band cluster, mild dust+noise
        seed=19,
        n_lanes=20,
        width=2400,
        height=600,
        band_rows=(0.48,),
        band_amp=0.30,
        band_y_sigma=7.0,
        n_dust=25,
        dust_sigma=3.0,
        dust_amp=0.4,
        noise_sigma=_NOISE,
        tags=("count", "realistic"),
    ),
)


def by_name(name: str) -> GelSpec:
    for spec in SUITE:
        if spec.name == name:
            return spec
    raise KeyError(f"no synthetic condition named {name!r}")


def by_tag(tag: str) -> tuple[GelSpec, ...]:
    return tuple(s for s in SUITE if tag in s.tags)
