"""What every figure renderer shares: how a label is drawn, and how a figure is saved."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from ..utils.atomic import atomic_write


def plain_text(label: str) -> str:
    r"""``label`` as matplotlib draws it literally.

    Matplotlib reads text that holds two or more unescaped ``$`` as mathtext, so a lane called
    ``$\frac$`` fails the whole figure. Escaping every ``$`` (matplotlib draws ``\$`` as ``$``)
    keeps a label what the person typed.
    """
    return label.replace("$", r"\$")


def save_figure(fig: plt.Figure, out_path: str | Path, **options) -> Path:
    """Write ``fig`` as a PNG to ``out_path`` atomically, and close it however the save ends.

    The file is made by ``utils.atomic.atomic_write``: a link at ``out_path`` is replaced and never
    written through, and a save that fails keeps the plot that was there.
    """
    try:
        atomic_write(out_path, lambda handle: fig.savefig(handle, format="png", **options))
    finally:
        plt.close(fig)
    return Path(out_path)
