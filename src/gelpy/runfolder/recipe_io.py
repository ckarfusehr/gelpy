"""Load/save the recipe and build an ``ImageRef`` from a source file."""

from __future__ import annotations

import os
from pathlib import Path

from ..core.errors import SourceImageChangedError, SourceImageMissingError, shown
from ..core.image_io import canonicalize_image, load_image
from ..core.text import decode_utf8
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
    """The recipe of a run. A recipe.json that is not UTF-8 (saved by an editor in another
    encoding) is an ``InvalidValueError`` that names the file, the line and the encoding."""
    text = decode_utf8(
        paths.recipe.read_bytes(),
        what=f"The recipe {shown(str(paths.recipe))}",
        detail="Save recipe.json as UTF-8 text and repeat the command.",
    )
    return Recipe.model_validate_json(text)


def save_recipe(paths: RunPaths, recipe: Recipe) -> None:
    write_json(paths.recipe, recipe)


def source_unchanged(ref: ImageRef, path: str | Path | None = None) -> bool:
    """Fast path: an unchanged size and mtime of the source file means its bytes need no
    re-hashing. ``path`` is where the file is now (``RunPaths.resolve(ref.path)``)."""
    try:
        stat = os.stat(path if path is not None else ref.path)
    except OSError:
        return False
    return stat.st_size == ref.size and stat.st_mtime == ref.mtime


def _where(ref: ImageRef, source: Path) -> str:
    """Where the image is looked for and how the recipe records it, for an error message: the
    recorded path is relative to the run folder, which a reader of the message cannot place."""
    resolved = os.path.realpath(source)
    return repr(resolved) if ref.path == resolved else f"{resolved!r} (recorded as {ref.path!r})"


def require_source(ref: ImageRef, source: Path) -> None:
    """Raise ``SourceImageMissingError`` unless a file is at ``source``, where the run looks for
    the image ``ref`` records."""
    if not source.exists():
        raise SourceImageMissingError(
            f"The source image recorded for this run is missing: {_where(ref, source)}.",
            detail=(
                f"Put the image back at {os.path.realpath(source)!r}, or set `source_image.path` "
                f"in recipe.json to where it is now, then repeat the command."
            ),
        )


def require_unchanged(ref: ImageRef, source: Path, sha256: str) -> None:
    """Raise ``SourceImageChangedError`` unless the bytes at ``source`` (their ``sha256``) are the
    image ``ref`` records. The recipe is the source of truth for the image's identity, and every
    hash chains off it: pixels from another file would be cached under the recipe's name."""
    if sha256 != ref.sha256:
        raise SourceImageChangedError(
            f"Source image {_where(ref, source)} has changed on disk "
            f"(recipe sha256 {ref.sha256[:12]}…, current {sha256[:12]}…). Re-run `gelpy init` "
            f"against a fresh run folder, or point the recipe at the intended file."
        )
