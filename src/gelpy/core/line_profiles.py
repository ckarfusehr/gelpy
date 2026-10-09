"""Line-profile extraction, slicing, and normalization."""

from __future__ import annotations

import numpy as np


def extract_line_profile(image: np.ndarray, x_center: float, width: int) -> np.ndarray:
    """Average a vertical strip of columns centered on ``x_center`` into a 1D profile.

    Returns the *full, unsliced* column-mean intensity (one value per image row).
    Slicing and normalization are applied later so the cached profile stays reusable
    across different slice/normalization choices.
    """
    if width <= 0:
        raise ValueError("width must be a positive integer")
    half = width / 2
    start_x = max(0, int(x_center - half))
    end_x = min(image.shape[1], int(x_center + half))
    if end_x <= start_x:
        raise ValueError(f"empty column strip for x_center={x_center}, width={width}")
    return np.mean(image[:, start_x:end_x], axis=1)


def strips_overlap(x_centers: list[float], width: int) -> list[tuple[int, int]]:
    """Return index pairs of adjacent lanes whose ``width``-strips overlap.

    Used to surface a non-fatal warning instead of silently averaging overlapping
    columns into two different lanes.
    """
    order = sorted(range(len(x_centers)), key=lambda i: x_centers[i])
    overlaps: list[tuple[int, int]] = []
    for a, b in zip(order, order[1:]):
        if (x_centers[b] - x_centers[a]) < width:
            overlaps.append((a, b))
    return overlaps


def apply_slice(profile: np.ndarray, start: int, end: int | None) -> np.ndarray:
    """Slice a profile with the same semantics as the config (``end=None`` -> to end)."""
    return profile[start:end]


def normalize_profile(profile: np.ndarray, normalization: str) -> np.ndarray:
    """Normalize a profile. ``"area"`` (trapezoid), ``"min_max"``, or ``"safe"`` (no-op)."""
    if normalization == "safe":
        return profile
    if normalization == "min_max":
        min_val = np.min(profile)
        max_val = np.max(profile)
        span = max_val - min_val
        if span == 0:
            return np.zeros_like(profile)
        return (profile - min_val) / span
    if normalization == "area":
        area = np.trapezoid(profile)
        if area == 0:
            return profile
        return profile / area
    raise ValueError(
        f"Unknown normalization {normalization!r}; expected 'area', 'min_max', or 'safe'."
    )
