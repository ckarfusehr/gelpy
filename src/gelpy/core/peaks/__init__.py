"""Peak-shape models and the shared fitting engine."""

from __future__ import annotations

from .base import SQRT_2PI, PeakModel, peak_registry

# Register the built-in shapes by importing their modules.
from . import gaussian as _gaussian  # noqa: F401,E402
from . import emg as _emg  # noqa: F401,E402
from . import bi_gaussian as _bi_gaussian  # noqa: F401,E402
from . import pseudo_voigt as _pseudo_voigt  # noqa: F401,E402

__all__ = ["PeakModel", "peak_registry", "SQRT_2PI"]
