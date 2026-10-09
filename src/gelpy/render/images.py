"""Render the gel image (optionally contrast-adjusted) with a lane-strip overlay."""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from ..core.contrast import apply_gamma, rescale_linear
from ..schemas.contrast import ContrastConfig
from ..schemas.lanes import LaneLayout
from ..utils.units import cm_to_inch


def _for_display(image: np.ndarray) -> np.ndarray:
    """Clamp negative pixels away for display only.

    The background-corrected image deliberately keeps *signed* residuals: rectifying it
    would bias every band area upward (see ``core/background``'s ``clip_negative``), so the
    cache stores the signed array and only the display path clamps. Clamping at zero — rather
    than shifting by the minimum — keeps the intensity scale intact, so a band at 0.5 still
    renders at 0.5 instead of being offset by whatever the single most negative noise pixel
    happens to be. Negative excursions are noise below the fitted background; inspect them in
    ``background show``, which plots the signed residual slices on purpose.
    """
    return np.clip(image, 0.0, None) if image.min() < 0.0 else image


def _adjust(image: np.ndarray, contrast: ContrastConfig | None, mode: str) -> np.ndarray:
    # "raw" means raw: the canonical array as gelpy sees it, no display contrast at all.
    # It is usually dark for a fluorescence gel -- use "non_linear" to actually look at the
    # sample, and "raw" to check what canonicalization/polarity handed the pipeline.
    if contrast is None or mode == "raw":
        return image
    if mode == "non_linear":
        display = _for_display(image)
        return apply_gamma(display, contrast.gamma.gamma, contrast.gamma.gain)
    if mode == "linear" and contrast.intensity_range is not None:
        ir = contrast.intensity_range
        return rescale_linear(_for_display(image), ir.low, ir.high, ir.mode)
    return image


def render_gel_image(
    image: np.ndarray,
    out_path: str | Path,
    *,
    contrast: ContrastConfig | None = None,
    layout: LaneLayout | None = None,
    mode: str = "non_linear",
    title: str = "",
    show_lanes: bool = True,
) -> Path:
    """Save a grayscale gel image with optional contrast and lane-strip overlay.

    ``show_lanes`` gates only the semi-transparent red lane rectangles/boundaries — the
    per-lane tick labels above the image are drawn whenever a ``layout`` is given, since
    they don't obscure the sample and are useful even on a plain contrast view.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    disp = _adjust(image, contrast, mode)

    fig, ax = plt.subplots(figsize=(cm_to_inch(18), cm_to_inch(12)))
    ax.imshow(disp, cmap="gray")
    ax.set(title=title, ylabel="[px]")

    if layout is not None and layout.lanes:
        if show_lanes:
            width = layout.profile_width_px
            for lane in layout.lanes:
                rect = patches.Rectangle(
                    (lane.x_center_px - width / 2, 0),
                    width,
                    image.shape[0],
                    linewidth=0.5,
                    edgecolor="darkred",
                    facecolor=(1, 0, 0, 0.12),
                )
                ax.add_patch(rect)
        ax.set_xticks([lane.x_center_px for lane in layout.lanes])
        ax.set_xticklabels(layout.labels(), rotation=layout.label_rotation)
        ax.xaxis.set_ticks_position("top")

    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out_path
