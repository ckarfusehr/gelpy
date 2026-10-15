"""Stage names and the hash chain that decides what invalidates what.

The chain is deliberately hand-rolled over exactly four fixed stages::

    image -> background (optional; absent = passthrough) -> profile[lane] -> fit[lane]

Every hash is computed from the recipe + small config objects alone — never from
the multi-megabyte image — so deciding what to recompute costs a handful of
sha256-over-small-JSON operations, independent of image size.
"""

from __future__ import annotations

from enum import Enum

from ..schemas.background import BackgroundStageConfig
from ..schemas.fit_config import FitConfig
from ..schemas.lanes import LaneDefinition
from ..schemas.profiles import LineProfileConfig
from ..schemas.recipe import Recipe
from .hashing import hash_fields


class StageName(str, Enum):
    IMAGE = "image"
    BACKGROUND = "background"
    PROFILE = "profile"
    FIT = "fit"


def code_version() -> str:
    """The running gelpy version.

    Part of ``image_hash`` and therefore of every stage hash chained to it, so a new version
    never reuses a cache an older one wrote. Read at call time, not at import.
    """
    import gelpy

    return gelpy.__version__


def image_hash(recipe: Recipe) -> str:
    """Identity of the CANONICAL image = source bytes + how they are interpreted + the gelpy version.

    Folds the ``ImageMeta`` interpretation fields (channel + invert + page) into the hash so
    that changing ``--invert``/``--channel``/``--page`` invalidates the cached array (and, by
    chaining, every downstream stage). Mirrors ``background_hash`` hashing validated params.
    """
    m = recipe.image
    return hash_fields(
        source=recipe.source_image.sha256,
        channel_strategy=m.channel_strategy,
        channel_index=m.channel_index,
        inverted=m.inverted,
        page=m.page,
        code=code_version(),
    )


def background_hash(recipe: Recipe) -> str:
    """Hash of the background-corrected image.

    If background removal is off, this equals the image hash (passthrough), so no
    background cache entry is ever created or consulted.
    """
    bg: BackgroundStageConfig | None = recipe.background
    if bg is None:
        return image_hash(recipe)
    # Hash the *validated* params (defaults resolved, types coerced), not the raw
    # dict from the recipe — otherwise a hand-edited recipe.json could miss/collide
    # in the cache. Lazy import keeps this module free of a core-package import at
    # load time.
    from ..core.background import background_registry

    config_model = background_registry.config_model(bg.model)
    params = (
        config_model.model_validate(bg.params).model_dump(mode="json")
        if config_model is not None
        else bg.params
    )
    return hash_fields(image=image_hash(recipe), model=bg.model, params=params)


def profile_hash(recipe: Recipe, lane: LaneDefinition) -> str:
    """Hash of a lane's full, unsliced profile.

    Deliberately excludes slice bounds and normalization (applied after the cached
    profile is loaded — cheap array ops, not re-averaging) and the lane label
    (cosmetic; renaming a lane must not force a recompute).
    """
    return hash_fields(
        background=background_hash(recipe),
        x_center_px=lane.x_center_px,
        profile_width_px=recipe.lanes.profile_width_px,
    )


def fit_hash(
    recipe: Recipe,
    lane: LaneDefinition,
    profile_config: LineProfileConfig,
    fit_config: FitConfig,
) -> str:
    """Hash of a lane's fit = profile hash + slice/normalization + the full fit config.

    ``fit_config`` folds in the strategy name, baseline, allowed shapes, priors, and the RNG
    ``seed`` — so even the stochastic bayesian tier is content-addressed and cacheable.
    """
    return hash_fields(
        profile=profile_hash(recipe, lane),
        profile_config=profile_config.model_dump(mode="json"),
        fit_config=fit_config.model_dump(mode="json"),
    )


def stage_key(stage: StageName, lane_index: int | None = None) -> str:
    """Fully-qualified stage key, e.g. ``"image"`` or ``"fit:lane=12"``."""
    if lane_index is None:
        return stage.value
    return f"{stage.value}:lane={lane_index}"
