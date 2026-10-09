"""Load/save the recipe and build an ``ImageRef`` from a source file."""

from __future__ import annotations

import os
from pathlib import Path

from ..core.image_io import canonicalize_image, load_image
from ..schemas.image import ImageMeta, ImageRef
from ..schemas.recipe import Recipe
from .cache import write_json
from .hashing import file_hash
from .layout import RunPaths


def build_image_ref(path: str | Path) -> ImageRef:
    """Decode ``path`` just enough to record its identity and dimensions."""
    path = Path(path)
    stat = path.stat()
    image = load_image(str(path))
    return ImageRef(
        path=str(path),
        sha256=file_hash(path),
        size=stat.st_size,
        mtime=stat.st_mtime,
        height=int(image.shape[0]),
        width=int(image.shape[1]),
        dtype=str(image.dtype),
    )


def build_image_ref_and_meta(
    path: str | Path, *, channel: str = "auto", invert: bool | str = "auto", page: int = 0
) -> tuple[ImageRef, ImageMeta]:
    """Build the file-identity ref AND the interpretation meta in one decode."""
    path = Path(path)
    stat = path.stat()
    image, meta = canonicalize_image(str(path), channel=channel, invert=invert, page=page)
    ref = ImageRef(
        path=str(path), sha256=file_hash(path), size=stat.st_size, mtime=stat.st_mtime,
        height=int(image.shape[0]), width=int(image.shape[1]), dtype=str(image.dtype),
    )
    return ref, meta


def load_recipe(paths: RunPaths) -> Recipe:
    return Recipe.model_validate_json(paths.recipe.read_text())


def save_recipe(paths: RunPaths, recipe: Recipe) -> None:
    write_json(paths.recipe, recipe)


def source_unchanged(ref: ImageRef) -> bool:
    """Fast path: an unchanged ``(path, size, mtime)`` means no need to re-hash bytes."""
    try:
        stat = os.stat(ref.path)
    except OSError:
        return False
    return stat.st_size == ref.size and stat.st_mtime == ref.mtime
