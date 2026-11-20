"""Render the gel image (optionally contrast-adjusted) with a lane-strip overlay."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from ..core.contrast import apply_gamma, rescale_linear
from ..schemas.contrast import ContrastConfig
from ..schemas.lanes import LaneLayout
from ..utils.units import cm_to_inch
from ._figure import plain_text, save_figure


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
        ax.set_xticklabels([plain_text(label) for label in layout.labels()], rotation=layout.label_rotation)
        ax.xaxis.set_ticks_position("top")

    return save_figure(fig, out_path, bbox_inches="tight", dpi=300)


def grid_step(extent: int) -> int:
    """A round grid step (1, 2 or 5 times a power of ten) giving about ten lines over ``extent``."""
    magnitude = 10 ** max(0, math.floor(math.log10(max(extent, 1) / 10)))
    return min((m * magnitude for m in (1, 2, 5, 10)), key=lambda s: abs(extent / s - 10))


def render_grid(
    image: np.ndarray,
    out_path: str | Path,
    *,
    contrast: ContrastConfig | None = None,
    crop: tuple[int, int, int, int] | None = None,
    outside: np.ndarray | None = None,
    step: int | None = None,
) -> Path:
    """Save the oriented image with a labelled pixel grid: the frame a crop box is given in.

    Grid lines every ``step`` pixels (default ``grid_step``), labelled on all four sides; the
    crop box outlined in orange with the area outside it dimmed; the pixels ``outside`` marks
    (the corners a fine rotation fills from outside the source) tinted blue. The display contrast
    is the recipe's non-linear one, as in the default gel view.
    """
    height, width = image.shape
    step = step or grid_step(max(height, width))
    disp = _adjust(image, contrast, "non_linear")
    fig_width = cm_to_inch(18)
    fig_height = min(max(fig_width * height / width, cm_to_inch(6)), cm_to_inch(30))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.imshow(disp, cmap="gray")
    if outside is not None and outside.any():
        tint = np.zeros((height, width, 4), dtype=np.uint8)
        tint[outside] = (40, 110, 255, 120)
        ax.imshow(tint)
    if crop is not None:
        x0, y0, x1, y1 = crop
        shade = {"facecolor": (0, 0, 0, 0.5), "edgecolor": "none"}
        for x, y, w, h in (
            (-0.5, -0.5, width, y0),  # above
            (-0.5, y1 - 0.5, width, height - y1),  # below
            (-0.5, y0 - 0.5, x0, y1 - y0),  # left
            (x1 - 0.5, y0 - 0.5, width - x1, y1 - y0),  # right
        ):
            if w > 0 and h > 0:
                ax.add_patch(patches.Rectangle((x, y), w, h, **shade))
        ax.add_patch(
            patches.Rectangle(
                (x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, fill=False, edgecolor="orange", linewidth=1.0
            )
        )
    ax.set_xticks(np.arange(0, width, step))
    ax.set_yticks(np.arange(0, height, step))
    ax.grid(color="cyan", alpha=0.45, linewidth=0.4)
    ax.tick_params(
        labelsize=5, top=True, right=True, labeltop=True, labelright=True, length=2, width=0.4
    )
    ax.tick_params(axis="x", labelrotation=90)
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    box = f"crop [{', '.join(str(v) for v in crop)}]" if crop is not None else "no crop"
    ax.set_title(f"oriented image {width} x {height} px, {box}, grid every {step} px", fontsize=6)
    return save_figure(fig, out_path, bbox_inches="tight", dpi=300)
