"""Render the synthetic suite to a contact-sheet PNG for human review.

For each condition it draws the gel (percentile-stretched, like the real overlay) with the
ground-truth lane centers (red) and the informative band region (cyan span), and below it two
column projections on a shared x-axis:

  - grey  = full-height projection (what the current detector sees)
  - blue  = band-region-only projection (the SNR the adaptive band could recover)

The gap between the two curves on the narrow-band / noisy / dusty conditions is the whole
motivation for the feature; on the ``control_*`` conditions they should nearly coincide.

Run:  python -m tests.synthetic.render_samples [out.png]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gelpy.core.lanes import column_projection

from .conditions import SUITE
from .generator import generate


def _stretch(img: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(img, 1)), float(np.percentile(img, 99.5))


def _norm(v: np.ndarray) -> np.ndarray:
    span = float(v.max() - v.min())
    return (v - v.min()) / span if span > 0 else v - v.min()


def render_contact_sheet(out_path: Path, ncols: int = 3) -> Path:
    gels = [generate(spec) for spec in SUITE]
    n = len(gels)
    nrows = (n + ncols - 1) // ncols

    fig = plt.figure(figsize=(ncols * 5.2, nrows * 3.4))
    outer = fig.add_gridspec(nrows, ncols, hspace=0.45, wspace=0.18)

    for idx, gel in enumerate(gels):
        r, c = divmod(idx, ncols)
        inner = outer[r, c].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
        ax_img = fig.add_subplot(inner[0])
        ax_proj = fig.add_subplot(inner[1], sharex=ax_img)

        img = gel.image
        vlo, vhi = _stretch(img)
        ax_img.imshow(img, cmap="gray", aspect="auto", vmin=vlo, vmax=vhi)
        for cx in gel.centers:
            ax_img.axvline(cx, color="red", lw=0.6, alpha=0.7)
        ax_img.axhspan(gel.band_y_lo, gel.band_y_hi, color="cyan", alpha=0.12)
        empties = f"  empty={list(gel.empty_lanes)}" if gel.empty_lanes else ""
        ax_img.set_title(
            f"{gel.spec.name}  (N={gel.spec.n_lanes}, amp={gel.spec.band_amp:.2f}, "
            f"noise={gel.spec.noise_sigma:.2f}){empties}",
            fontsize=8.5,
        )
        ax_img.set_xticks([])
        ax_img.set_yticks([])

        full = _norm(column_projection(img))
        band = _norm(
            column_projection(img, int(round(gel.band_y_lo)), int(round(gel.band_y_hi)))
        )
        xs = np.arange(img.shape[1])
        ax_proj.plot(xs, full, color="0.5", lw=0.7, label="full-height")
        ax_proj.plot(xs, band, color="tab:blue", lw=0.8, label="band-region")
        for cx in gel.centers:
            ax_proj.axvline(cx, color="red", lw=0.4, alpha=0.4)
        ax_proj.set_ylim(-0.05, 1.1)
        ax_proj.set_yticks([])
        ax_proj.tick_params(labelsize=6)
        if idx == 0:
            ax_proj.legend(fontsize=5.5, loc="upper right", framealpha=0.6)

    fig.suptitle(
        "Synthetic gel suite — GT centers (red), band region (cyan); "
        "grey=full-height projection, blue=band-region projection",
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("synthetic_suite_review.png")
    print(render_contact_sheet(dest))
