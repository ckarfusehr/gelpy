"""Synthetic gel-image suite with ground-truth lane centers, for adaptive y-band detection.

- ``generator.generate(spec)`` -> ``SynthGel`` (image + ground truth), seeded/deterministic.
- ``conditions.SUITE`` -> the canonical ordered list of ``GelSpec`` conditions.
- ``render_samples`` -> a contact-sheet PNG for human review.
"""

from .conditions import SUITE, by_name, by_tag
from .generator import GelSpec, SynthGel, generate, with_seed

__all__ = [
    "SUITE",
    "by_name",
    "by_tag",
    "GelSpec",
    "SynthGel",
    "generate",
    "with_seed",
]
