"""Render an overview of selected line profiles."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..utils.units import cm_to_inch


def render_profiles_overview(
    profiles: list[tuple[str, np.ndarray]],
    out_path: str | Path,
    *,
    slice_start: int = 0,
    normalization: str = "area",
) -> Path:
    """Overlay one line per (label, values) profile.

    ``profiles`` is a list of ``(label, values)``; all values are plotted against a
    shared pixel x-axis offset by ``slice_start``.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(cm_to_inch(18), cm_to_inch(10)))
    for label, values in profiles:
        x = np.arange(slice_start, slice_start + len(values))
        ax.plot(x, values, label=label)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.set(ylabel=f"normalized to {normalization}", xlabel="[px]")

    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out_path
