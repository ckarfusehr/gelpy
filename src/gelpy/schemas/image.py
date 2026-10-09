"""Reference to a source image on disk, and how it is interpreted."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .common import CanonicalModel


class ImageRef(CanonicalModel):
    """Content-addressed pointer to a source image.

    The image is referenced by path + ``sha256`` rather than copied into the run
    folder by default (many run folders analyze the same large TIFF with different
    parameters). ``size``/``mtime`` are a cheap fast-path: an unchanged
    ``(path, size, mtime)`` triple lets the pipeline skip re-hashing the file bytes.
    """

    path: str = Field(description="Path to the source image as referenced by the run.")
    sha256: str = Field(description="SHA-256 of the raw file bytes.")
    size: int = Field(ge=0, description="File size in bytes.")
    mtime: float = Field(description="File modification time (epoch seconds).")
    height: int = Field(gt=0, description="Decoded grayscale image height in pixels.")
    width: int = Field(gt=0, description="Decoded grayscale image width in pixels.")
    dtype: str = Field(description="numpy dtype of the decoded grayscale array.")


class ImageMeta(CanonicalModel):
    """How the source image is interpreted to produce the canonical analysis array.

    Interpretation intent (channel + polarity + page) — editable and precious, a sibling
    to ``ImageRef`` (file identity). Detected once at init; frozen here to make the cached
    pipeline deterministic. These fields feed ``image_hash`` (``runfolder/stages.py``), so a
    change here recomputes exactly what depends on the pixels.
    """

    original_dtype: str = Field(default="", description="numpy dtype of the decoded source.")
    original_shape: tuple[int, ...] = Field(default=(), description="Decoded source shape.")
    source_channels: int = Field(default=1, ge=1, description="Source channel count (1=grayscale).")
    source_pages: int = Field(default=1, ge=1, description="Pages in the source (TIFF stacks).")
    page: int = Field(default=0, ge=0, description="Page index used from a multi-page source.")
    channel_strategy: str = Field(
        default="mean",
        description="How channels collapse to grayscale: gray|gray_rgb|mean|luminance|r|g|b|dominant|max.",
    )
    channel_index: int | None = Field(
        default=None, description="Chosen channel index for r/g/b/dominant strategies; else null."
    )
    polarity: Literal["bright_on_dark", "dark_on_light"] = Field(
        default="bright_on_dark", description="Signal polarity of the source image."
    )
    inverted: bool = Field(
        default=False, description="Whether 1-x was applied to make signal bright (canonical)."
    )
    polarity_source: Literal["auto", "manual"] = Field(
        default="auto", description="Whether polarity was auto-detected or set explicitly."
    )
    polarity_confidence: Literal["ok", "low"] = Field(
        default="ok", description="Detector confidence; 'low' raises an ignorable warning."
    )
    polarity_reason: str = Field(default="", description="Human reason when confidence is 'low'.")
