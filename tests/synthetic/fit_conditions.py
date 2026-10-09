"""Named synthetic line-profile conditions the peak fitter must survive.

Grouped by what they stress: clean/faint single peaks, resolvable-doublet ladder, shoulders,
tails (both directions), the smear+band yield case, well+bands, dust, heavy baseline drift,
and a full mixed lane. ``SUITE`` is the canonical ordered list the harness iterates.
"""

from __future__ import annotations

from .profiles import ProfileSpec, TrueComponent as TC

_L = 400


def _doublet(name: str, sep: float, sigma: float = 9.0, tags=("doublet",)) -> ProfileSpec:
    c = _L / 2
    return ProfileSpec(
        name=name,
        components=(
            TC("gaussian", c - sep / 2, 1.0, sigma),
            TC("gaussian", c + sep / 2, 0.8, sigma),
        ),
        tags=tags,
    )


SUITE: tuple[ProfileSpec, ...] = (
    # --- single peaks -----------------------------------------------------------------------
    ProfileSpec("single_clean", components=(TC("gaussian", 200, 1.0, 8),), tags=("single", "core")),
    ProfileSpec(
        "single_faint", components=(TC("gaussian", 200, 0.15, 8),), noise=0.03, tags=("single", "snr")
    ),
    ProfileSpec(
        "single_wide_sigma",
        components=(TC("gaussian", 200, 0.5, 18),),
        tags=("single", "width"),
    ),
    # --- resolvable-doublet ladder (decreasing separation) ----------------------------------
    _doublet("doublet_sep60", 60, tags=("doublet", "core")),
    _doublet("doublet_sep36", 36, tags=("doublet",)),
    _doublet("doublet_sep24", 24, tags=("doublet", "shoulder")),  # ~2.7 sigma: a real shoulder
    _doublet("doublet_sep16", 16, tags=("doublet", "shoulder", "hard")),  # ~1.8 sigma: borderline
    # --- tails (both directions) ------------------------------------------------------------
    ProfileSpec("tail_right", components=(TC("emg", 200, 1.0, 7, tau=22),), tags=("tail", "core")),
    ProfileSpec("tail_left", components=(TC("emg", 220, 1.0, 7, tau=-22),), tags=("tail",)),
    # --- smear (the yield case) -------------------------------------------------------------
    ProfileSpec(
        "smear_and_band",
        components=(
            TC("gaussian", 140, 1.0, 7, cls="sample"),
            TC("gaussian", 275, 0.2, 30, cls="smear"),
        ),
        tags=("smear", "core", "yield"),
    ),
    ProfileSpec(
        "smear_only", components=(TC("gaussian", 200, 0.2, 40, cls="smear"),), tags=("smear",)
    ),
    # --- well + bands -----------------------------------------------------------------------
    ProfileSpec(
        "well_and_bands",
        components=(
            TC("gaussian", 15, 5.0, 3, cls="well"),
            TC("gaussian", 180, 1.0, 8),
            TC("gaussian", 260, 0.6, 8),
        ),
        tags=("well", "core"),
    ),
    # --- dust -------------------------------------------------------------------------------
    ProfileSpec(
        "dust_on_band",
        components=(TC("gaussian", 200, 0.8, 8),),
        n_dust=6,
        dust_amp=0.3,
        seed=3,
        tags=("dust",),
    ),
    # --- baseline drift ---------------------------------------------------------------------
    ProfileSpec(
        "heavy_drift",
        components=(TC("gaussian", 200, 0.8, 8),),
        baseline=(0.5, 0.003, 3e-6),
        tags=("drift", "core"),
    ),
    # --- many peaks -------------------------------------------------------------------------
    ProfileSpec(
        "ladder_6",
        components=tuple(
            TC("gaussian", 60 + i * 55, amp, 7)
            for i, amp in enumerate((0.6, 1.0, 0.8, 0.5, 0.9, 0.7))
        ),
        tags=("multi", "ladder"),
    ),
    # --- full mixed lane --------------------------------------------------------------------
    ProfileSpec(
        "mixed_five",
        components=(
            TC("gaussian", 12, 4.0, 3, cls="well"),
            TC("gaussian", 150, 1.0, 8),
            TC("emg", 230, 0.8, 7, tau=20),
            TC("gaussian", 320, 0.25, 30, cls="smear"),
        ),
        n_dust=1,
        dust_amp=0.6,
        seed=7,
        baseline=(0.3, 0.0015, 2e-6),
        tags=("mixed", "core", "yield"),
    ),
)


def by_name(name: str) -> ProfileSpec:
    for spec in SUITE:
        if spec.name == name:
            return spec
    raise KeyError(f"no fit condition named {name!r}")


def by_tag(tag: str) -> tuple[ProfileSpec, ...]:
    return tuple(s for s in SUITE if tag in s.tags)
