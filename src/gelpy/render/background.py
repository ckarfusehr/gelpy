"""Render a background-fit diagnostic figure (ports ``PlaneFit2d.visualize_fit_data``)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from ..utils.units import cm_to_inch


def _stripe_overlay(shape: tuple[int, int], stripes: list[tuple[int, int]]) -> np.ndarray:
    h, w = shape
    overlay = np.zeros((h, w, 4))
    overlay[:, :, 0] = 1  # red
    for position, height in stripes:
        overlay[position : position + height, :, 3] = 0.9
    return overlay


def render_background_diagnostic(
    image: np.ndarray,
    background_plane: np.ndarray,
    stripes: list[tuple[int, int]],
    out_path: str | Path,
) -> Path:
    """Top: image with the fit stripes highlighted. Bottom: original/background/corrected
    column-mean profiles across a few vertical slices, to eyeball the correction."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    corrected = image - background_plane
    corrected[corrected < 0] = 0
    h, w = image.shape
    slice_indices = np.linspace(0, w, 4).astype(int)

    fig = plt.figure(figsize=(cm_to_inch(18), cm_to_inch(16)))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[2, 1])

    vmin, vmax = np.percentile(image, [5, 95])
    ax_img = fig.add_subplot(gs[0, :])
    ax_img.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
    ax_img.imshow(_stripe_overlay(image.shape, stripes))
    ax_img.set_title("Background-fit stripes (red) and validation slices")
    ax_img.set_ylabel("[px]")
    ax_img.set_xticks([])

    for i in range(3):
        start, end = slice_indices[i], slice_indices[i + 1]
        ax = fig.add_subplot(gs[1, i])
        ax.plot(image[:, start:end].mean(axis=1), "--", color="gray", label="original")
        ax.plot(background_plane[:, start:end].mean(axis=1), color="gray", label="background")
        ax.plot(corrected[:, start:end].mean(axis=1), color="green", label="corrected")
        ax.set_yticks([])
        if i == 0:
            ax.legend(fontsize=6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def render_background_surface(
    image: np.ndarray,
    surface: np.ndarray,
    weights: np.ndarray | None,
    corrected: np.ndarray,
    out_path: str | Path,
) -> Path:
    """Diagnostic for the fitted-surface background models (``robust_plane``).

    Top: image, the fitted background surface, and the per-pixel weight map (bright =
    trusted as background). The weight map is the key diagnostic — dark bands/smears there
    mean the fit correctly ignored them. Bottom: original/background/corrected column-mean
    profiles across a few vertical slices, to eyeball the correction.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    h, w = image.shape
    slice_indices = np.linspace(0, w, 4).astype(int)
    fig = plt.figure(figsize=(cm_to_inch(18), cm_to_inch(16)))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[2, 1])

    vmin, vmax = np.percentile(image, [5, 95])
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
    ax_img.set_title("image")
    ax_img.set_xticks([])
    ax_img.set_ylabel("[px]")

    ax_surf = fig.add_subplot(gs[0, 1])
    ax_surf.imshow(surface, cmap="gray")
    ax_surf.set_title("fitted background surface")
    ax_surf.set_xticks([])
    ax_surf.set_yticks([])

    ax_w = fig.add_subplot(gs[0, 2])
    if weights is not None:
        im = ax_w.imshow(weights, cmap="magma", vmin=0.0, vmax=1.0)
        fig.colorbar(im, ax=ax_w, fraction=0.046, pad=0.04)
        ax_w.set_title("background trust (weights)")
    else:
        ax_w.axis("off")
    ax_w.set_xticks([])
    ax_w.set_yticks([])

    for i in range(3):
        start, end = slice_indices[i], slice_indices[i + 1]
        ax = fig.add_subplot(gs[1, i])
        ax.plot(image[:, start:end].mean(axis=1), "--", color="gray", label="original")
        ax.plot(surface[:, start:end].mean(axis=1), color="gray", label="background")
        ax.plot(corrected[:, start:end].mean(axis=1), color="green", label="corrected")
        ax.axhline(0, color="k", lw=0.4)
        ax.set_yticks([])
        if i == 0:
            ax.legend(fontsize=6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path
