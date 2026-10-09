"""Unit helpers used by the plotting code to size figures in centimeters."""

from __future__ import annotations


def cm_to_inch(cm: float) -> float:
    """Convert centimeters to inches (matplotlib figure sizing)."""
    return cm * 0.393701
