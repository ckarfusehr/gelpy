"""Render a lane-detection verification overlay (image + y-projection with fitted grid)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from ..utils.units import cm_to_inch


def render_lane_detection(
    image: np.ndarray,
    projection: np.ndarray,
    peak_indices: np.ndarray,
    centers: list[float],
    out_path: str | Path,
    *,
    y_start: int | None = None,
    y_end: int | None = None,
    y_band: list | tuple | None = None,
) -> Path:
    """Verification overlay for ``lanes detect``.

    Top: the gel at high contrast (a fixed 1-99.5 percentile stretch, independent of any
    configured ``ContrastConfig``, so every well is visible) with a vertical line at each
    fitted lane center — the single glance that confirms the grid sits on the real lanes.
    Bottom: the column projection with the detected peaks (x) and the fitted grid centers
    (dashed) marked, so a mismatch between "what was found" and "where lanes were placed"
    is obvious.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lo, hi = np.percentile(image, [1.0, 99.5])
    disp = np.clip((image - lo) / (hi - lo), 0, 1) if hi > lo else image

    fig = plt.figure(figsize=(cm_to_inch(22), cm_to_inch(12)))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[3, 1])

    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(disp, cmap="gray", aspect="auto")
    for c in centers:
        ax_img.axvline(c, color="red", lw=0.8, alpha=0.7)
    # Shade the projection band(s) detection actually used. An adaptive strategy's chosen
    # ``y_band`` (one or more [lo, hi] ranges) takes precedence; otherwise fall back to the
    # manual ``y_start:y_end`` window if one was given.
    if y_band:
        ranges = y_band if isinstance(y_band[0], (list, tuple)) else [y_band]
        for y0, y1 in ranges:
            ax_img.axhspan(y0, y1, color="deepskyblue", alpha=0.12)
    elif y_start is not None or y_end is not None:
        y0 = y_start or 0
        y1 = y_end if y_end is not None else image.shape[0]
        ax_img.axhspan(y0, y1, color="deepskyblue", alpha=0.12)
    ax_img.set_title(f"Detected lane grid — {len(centers)} centers (red)")
    ax_img.set_ylabel("[px]")
    ax_img.set_xticks([])

    ax_proj = fig.add_subplot(gs[1], sharex=ax_img)
    x = np.arange(len(projection))
    ax_proj.plot(x, projection, color="black", lw=0.8, label="y-projection")
    if len(peak_indices):
        ax_proj.plot(
            peak_indices, projection[np.asarray(peak_indices, dtype=int)], "x",
            color="tab:orange", label="detected peaks",
        )
    for c in centers:
        ax_proj.axvline(c, color="red", lw=0.6, ls="--", alpha=0.6)
    ax_proj.set(xlabel="x [px]", ylabel="mean intensity")
    ax_proj.legend(fontsize=6, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path
