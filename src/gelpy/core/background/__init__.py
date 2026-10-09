"""Background-removal models."""

from __future__ import annotations

from .base import BackgroundFitResult, BackgroundModel, background_registry

# Register built-in models.
from . import plane_ransac as _plane_ransac  # noqa: F401,E402
from . import robust_plane as _robust_plane  # noqa: F401,E402

__all__ = ["BackgroundFitResult", "BackgroundModel", "background_registry"]
