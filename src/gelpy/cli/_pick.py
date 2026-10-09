"""Interactive lane-center picker — opt-in human fallback for ``lanes detect``.

Not routed through ``render/`` (which forces the headless ``Agg`` backend at call time):
this module selects an interactive matplotlib backend itself and is only ever invoked
from the ``lanes pick`` CLI command, which needs a real display.
"""

from __future__ import annotations

import numpy as np

from ..core.contrast import apply_gamma
from ..schemas.contrast import ContrastConfig


def _use_interactive_backend() -> None:
    import matplotlib

    for backend in ("MacOSX", "QtAgg", "TkAgg"):
        try:
            matplotlib.use(backend, force=True)
            return
        except Exception:
            continue
    raise RuntimeError(
        "No interactive matplotlib backend available (tried MacOSX/QtAgg/TkAgg) — "
        "'lanes pick' needs a GUI/display. Use 'lanes detect' or 'lanes set --endpoints' instead."
    )


def pick_lane_centers(
    image: np.ndarray, contrast: ContrastConfig | None, n_clicks: int, title: str
) -> list[float]:
    """Show the image at non-linear contrast and collect ``n_clicks`` x-positions.

    Blocks until the user clicks ``n_clicks`` points (in any order — the result is
    sorted left to right) or closes the window early, which raises. Intentionally
    simple: no snapping, no drag-to-adjust — read positions straight off the same
    high-contrast view the detection overlay already uses.
    """
    _use_interactive_backend()
    import matplotlib.pyplot as plt

    disp = apply_gamma(image, contrast.gamma.gamma, contrast.gamma.gain) if contrast else image

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.imshow(disp, cmap="gray", aspect="auto")
    ax.set_title(f"{title}\nClick {n_clicks} lane center(s), then press Enter.")
    points = fig.ginput(n_clicks, timeout=0)
    plt.close(fig)
    if len(points) != n_clicks:
        raise RuntimeError(
            f"expected {n_clicks} click(s), got {len(points)} (window closed early?)"
        )
    return sorted(float(x) for x, _y in points)
