"""Render a per-lane fit overlay: profile + baseline + fitted components + residual."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..core.peaks import peak_registry
from ..schemas.peaks import BandFit, LaneFitResult
from ..utils.units import cm_to_inch
from ._figure import plain_text, save_figure

# Distinct-ish colors per component class (well/smear/dust get their own so the plot reads
# the way the labels do); sample bands cycle through the default color list.
_CLASS_COLORS = {"well": "tab:purple", "smear": "tab:orange", "dust": "tab:gray"}


def _component_curve(band: BandFit, x: np.ndarray) -> np.ndarray | None:
    if band.params is None:
        return None
    model = peak_registry.get(band.params.kind)()
    dumped = band.params.model_dump()
    params = tuple(float(dumped[name]) for name in model.param_names())
    return model.single_peak(x, *params)


def render_lane_fit(
    prepared_profile: np.ndarray,
    baseline: np.ndarray,
    lane_result: LaneFitResult,
    out_path: str | Path,
) -> Path:
    """Two panels: (left) raw profile + baseline + total model; (right) baseline-subtracted
    signal with each component colored by class, plus a residual strip."""
    x = np.arange(len(prepared_profile), dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    signal = prepared_profile - baseline

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(cm_to_inch(19), cm_to_inch(7.5)), sharey=False
    )
    ax_left.plot(x, prepared_profile, color="black", alpha=0.9, label="line profile")
    ax_left.plot(x, baseline, color="tab:blue", ls="--", lw=1, alpha=0.8, label="baseline")
    ax_left.set_title("Profile + baseline")
    ax_left.set_xlabel("pixel")

    ax_right.axhline(0.0, color="0.8", lw=0.8)
    ax_right.plot(x, signal, color="black", alpha=0.9, zorder=0.5, label="signal - baseline")
    conf = lane_result.confidence
    ax_right.set_title(f"Components  [{conf}]")
    ax_right.set_xlabel("pixel")

    if lane_result.status == "ok" and lane_result.bands:
        total = np.zeros_like(x)
        for band in lane_result.bands:
            curve = _component_curve(band, x)
            color = _CLASS_COLORS.get(band.component_class)
            label = f"{band.component_class} {band.area_fraction:.0%}"
            if curve is not None:
                total = total + curve
                ax_right.plot(x, curve, color=color, lw=1.2, label=label)
            elif band.region is not None:  # region-integrated smear
                lo, hi = band.region
                ax_right.fill_between(
                    x[lo:hi], 0, signal[lo:hi], color=color or "tab:orange", alpha=0.3, label=label
                )
                total[lo:hi] = total[lo:hi] + np.clip(signal[lo:hi], 0, None)
        ax_left.plot(x, baseline + total, color="red", ls="dotted", lw=1.2, label="model")
        ax_right.legend(fontsize=6, ncol=2)
        cov = lane_result.coverage or 0.0
        ax_right.text(
            0.98, 0.95, f"coverage {cov:.0%}", transform=ax_right.transAxes, ha="right", va="top",
            fontsize=7, color="0.3",
        )
    else:
        ax_right.text(
            0.5, 0.5, f"fit {lane_result.status}", transform=ax_right.transAxes, ha="center"
        )

    ax_left.legend(fontsize=6)

    # One shared, wrapped title for the whole figure (not per-subplot) — a per-axes title is
    # only as wide as its own panel, so a long lane label reliably overlapped its neighbor's.
    label = lane_result.label
    title_fontsize = 12 if len(label) <= 30 else 9
    fig.suptitle(plain_text(label), fontsize=title_fontsize, wrap=True, y=0.99)
    top = 0.90
    if lane_result.confidence_reasons:
        fig.text(
            0.5, 0.90, "  |  ".join(lane_result.confidence_reasons[:2]),
            ha="center", va="top", fontsize=7, color="tab:red", wrap=True,
        )
        top = 0.83
    fig.tight_layout(rect=(0, 0, 1, top))
    return save_figure(fig, out_path, dpi=300)
