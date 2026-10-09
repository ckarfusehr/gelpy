"""Rendering layer — the ONLY place matplotlib is imported.

Every renderer is a pure function: data in, a PNG (or markdown) file out. Nothing
here ever calls ``plt.show()``. The Agg backend is selected once, here, so the whole
package is headless-safe.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from .images import render_gel_image  # noqa: E402
from .lanes import render_lane_detection  # noqa: E402
from .profiles import render_profiles_overview  # noqa: E402
from .fits import render_lane_fit  # noqa: E402
from .background import render_background_diagnostic, render_background_surface  # noqa: E402
from .report import render_report  # noqa: E402

__all__ = [
    "render_gel_image",
    "render_lane_detection",
    "render_profiles_overview",
    "render_lane_fit",
    "render_background_diagnostic",
    "render_background_surface",
    "render_report",
]
