"""Real-image ground truth + evaluation, complementing the synthetic suite.

Loads hand-labeled lane centers from ``real_labels.json`` and scores a detection function on
the actual example TIFFs (``examples/data/*``). Detection is otherwise NOT exercised on the real
images anywhere in the repo (the e2e replay hard-codes ``--linspace``), so this is the reality
check that a design branch tuned on synthetic gels still behaves on real ones.

Images are large and may be absent locally; ``load_labeled_images`` silently skips missing
files (same policy as ``tests/e2e``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gelpy.core.image_io import load_image

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LABELS = Path(__file__).with_name("real_labels.json")


@dataclass
class RealGel:
    name: str
    image: np.ndarray
    centers: np.ndarray  # true lane x-centers (px)
    pitch: float
    n_lanes: int
    band_y: list[list[float]]
    notes: str


def _centers_from_entry(entry: dict) -> np.ndarray:
    n = entry["n_lanes"]
    if "centers" in entry:
        c = np.asarray(entry["centers"], dtype=np.float64)
        if len(c) != n:
            raise ValueError(f"{entry['name']}: {len(c)} centers for n_lanes={n}")
        return c
    lo, hi = entry["centers_linspace"]
    return np.linspace(lo, hi, n)


def load_labeled_images() -> list[RealGel]:
    """Load every labeled image that exists on disk (missing ones are skipped)."""
    spec = json.loads(_LABELS.read_text())
    gels: list[RealGel] = []
    for entry in spec["images"]:
        path = _REPO_ROOT / entry["path"]
        if not path.exists():
            continue
        centers = _centers_from_entry(entry)
        pitch = float(np.diff(centers).mean()) if len(centers) > 1 else float("nan")
        gels.append(
            RealGel(
                name=entry["name"],
                image=load_image(str(path)),
                centers=centers,
                pitch=pitch,
                n_lanes=entry["n_lanes"],
                band_y=entry.get("band_y", []),
                notes=entry.get("notes", ""),
            )
        )
    return gels


def score_real(gel: RealGel, detect_fn) -> dict:
    """Center error (px and xpitch) + confidence for one detection function on a real gel."""
    res = detect_fn(gel.image, gel.n_lanes)
    centers = np.sort(np.asarray(res.centers, dtype=np.float64))
    err = np.abs(centers - np.sort(gel.centers))
    return {
        "name": gel.name,
        "max_err_px": float(err.max()),
        "max_err_xpitch": float(err.max() / gel.pitch),
        "median_err_px": float(np.median(err)),
        "confidence": res.confidence,
    }


if __name__ == "__main__":
    from .harness import baseline_detect

    gels = load_labeled_images()
    if not gels:
        print("no labeled real images present on disk")
    for g in gels:
        print(f"{g.name}: shape={g.image.shape} N={g.n_lanes} pitch={g.pitch:.1f}")
        print("  baseline:", score_real(g, baseline_detect))
