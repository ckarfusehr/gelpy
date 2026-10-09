"""Run the fitter on real lane profiles extracted from example gel images.

Synthetic conditions verify accuracy against known areas; this is the qualitative check on
real data (no hard ground truth — inspect the summaries / a contact sheet by eye).
Skipped automatically when the example TIFF is absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gelpy.core.image_io import canonicalize_image
from gelpy.core.lanes import detect_lane_centers
from gelpy.core.line_profiles import extract_line_profile, normalize_profile
from gelpy.schemas.fit_config import FitConfig

DATA_DIR = Path(__file__).resolve().parents[2] / "examples" / "data"
DEFAULT_TIF = DATA_DIR / "folding_screen.tif"


@dataclass
class RealLane:
    label: str
    profile: np.ndarray


def load_real_profiles(
    path: Path = DEFAULT_TIF, n_lanes: int = 20, width: int | None = None, normalization: str = "area"
) -> list[RealLane]:
    """Extract per-lane normalized profiles from a real gel image (detected lane centers)."""
    image, _meta = canonicalize_image(str(path))
    det = detect_lane_centers(image, n_lanes)
    strip = int(width or det.profile_width)
    lanes: list[RealLane] = []
    for i, center in enumerate(det.centers):
        raw = extract_line_profile(image, float(center), strip)
        lanes.append(RealLane(label=str(i), profile=normalize_profile(raw, normalization)))
    return lanes


def fit_real(
    path: Path = DEFAULT_TIF, n_lanes: int = 20, method: str = "grow_select"
) -> list:
    """Fit every extracted real lane; returns the list of ``LaneFitResult``."""
    from gelpy.core.peaks.strategies import fit_strategy_registry

    fit = fit_strategy_registry.get(method)
    return [
        fit(lane.profile, FitConfig(), lane_index=i, label=lane.label)
        for i, lane in enumerate(load_real_profiles(path, n_lanes))
    ]


def summarize(results: list) -> str:
    lines = [f"{'lane':>4s} {'status':>6s} {'nbands':>6s} {'cov':>5s} {'conf':>5s}  reasons"]
    for r in results:
        cov = f"{r.coverage:.2f}" if r.coverage is not None else "  NA"
        reason = "; ".join(r.confidence_reasons[:1])
        lines.append(
            f"{r.lane_index:4d} {r.status:>6s} {len(r.bands):6d} {cov:>5s} {r.confidence:>5s}  {reason}"
        )
    return "\n".join(lines)
