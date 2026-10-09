"""The persisted analysis intent for a run."""

from __future__ import annotations

from pydantic import Field

from .background import BackgroundStageConfig
from .common import CanonicalModel
from .contrast import ContrastConfig
from .image import ImageMeta, ImageRef
from .lanes import LaneLayout


class Recipe(CanonicalModel):
    """Human/agent-editable intent for a run (``recipe.json``).

    Deliberately excludes fit config: the real workflow runs several distinct fit
    passes (broad / refit / tight) against one recipe, so each fit is its own
    request rather than a recipe field.
    """

    source_image: ImageRef
    image: ImageMeta = Field(
        default_factory=ImageMeta,
        description="Interpretation of the source image (channel/polarity/page). Feeds image_hash.",
    )
    lanes: LaneLayout
    contrast: ContrastConfig = Field(default_factory=ContrastConfig)
    background: BackgroundStageConfig | None = Field(
        default=None, description="Background-removal stage; None = no removal (passthrough)."
    )
