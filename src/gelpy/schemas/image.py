"""Reference to a source image on disk, and how it is interpreted."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import AfterValidator, Field

from .common import CanonicalModel


class ImageRef(CanonicalModel):
    """Content-addressed pointer to a source image.

    The image is referenced by path + ``sha256`` rather than copied into the run
    folder by default (many run folders analyze the same large TIFF with different
    parameters). ``size``/``mtime`` are a cheap fast-path: an unchanged
    ``(path, size, mtime)`` triple lets the pipeline skip re-hashing the file bytes.
    """

    path: str = Field(
        description="Path to the source image: relative to the run folder where possible "
        "(`RunPaths.store_path`), absolute otherwise."
    )
    sha256: str = Field(description="SHA-256 of the raw file bytes.")
    size: int = Field(ge=0, description="File size in bytes.")
    mtime: float = Field(description="File modification time (epoch seconds).")
    height: int = Field(gt=0, description="Decoded grayscale image height in pixels.")
    width: int = Field(gt=0, description="Decoded grayscale image width in pixels.")
    dtype: str = Field(description="numpy dtype of the decoded grayscale array.")


def _quarter_turn(turn: int) -> int:
    if turn % 90:
        raise ValueError(f"turn must be a multiple of 90 degrees, got {turn}")
    return turn % 360


def _ordered_box(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    if min(box) < 0:
        raise ValueError(f"crop box {list(box)} has a negative coordinate")
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"crop box {list(box)} is empty: x1 must be larger than x0, y1 than y0")
    return box


#: A quarter turn in degrees, counter-clockwise as displayed; stored as 0, 90, 180 or 270.
QuarterTurn = Annotated[int, AfterValidator(_quarter_turn)]
#: A fine rotation in degrees, counter-clockwise as displayed.
FineAngle = Annotated[float, Field(ge=-45.0, le=45.0, allow_inf_nan=False)]
#: [x0, y0, x1, y1]: the first column and row kept and one past the last, like a slice.
CropBox = Annotated[tuple[int, int, int, int], AfterValidator(_ordered_box)]


class ImageGeometry(CanonicalModel):
    """How the decoded image is oriented and cropped before analysis, in this order: ``turn``,
    ``flip``, ``angle``, ``crop`` (``core/geometry.py``).

    Turns, the mirror and the crop copy pixels exactly. The fine rotation interpolates
    bilinearly, so it never leaves the input's range; it rotates about the image centre onto a
    canvas just large enough for the whole source, padded evenly, and fills the corners that lie
    outside the source with the image's minimum (no signal, as gelpy's arrays are signal-bright).
    """

    turn: QuarterTurn = Field(
        default=0,
        description="Quarter turn in degrees, counter-clockwise as displayed: a multiple of 90 "
        "(-90 turns clockwise), stored as 0, 90, 180 or 270. Exact.",
    )
    flip: bool = Field(default=False, description="Mirror left-right, after the turn. Exact.")
    angle: FineAngle = Field(
        default=0.0,
        description="Fine rotation in degrees, counter-clockwise as displayed, after the turn and "
        "the flip: bilinear, about the image centre, onto a canvas that holds the whole source; "
        "the corners from outside the source take the image's minimum.",
    )
    crop: CropBox | None = Field(
        default=None,
        description="[x0, y0, x1, y1] in pixels of the oriented image (after turn, flip and "
        "angle): x0, y0 the first column and row kept, x1, y1 one past the last. null keeps the "
        "whole image. Exact.",
    )

    def is_identity(self) -> bool:
        """Whether the geometry leaves the image as decoded."""
        return self == ImageGeometry()

    def orientation(self) -> tuple[int, bool, float]:
        """(turn, flip, angle): the frame a crop box is chosen in."""
        return self.turn, self.flip, self.angle


class ImageMeta(CanonicalModel):
    """How the source image is interpreted to produce the canonical analysis array.

    Interpretation intent (channel + polarity + page + geometry) — editable and precious, a sibling
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
    geometry: ImageGeometry = Field(
        default_factory=ImageGeometry,
        description="Orientation and crop of the decoded image (turn, flip, angle, crop): part of "
        "how the source is read, so it feeds image_hash and every stage analyses the oriented, "
        "cropped image. Set with `image rotate` and `image crop`.",
    )
    reader_warnings: list[str] = Field(
        default_factory=list,
        description="What tifffile logged (warnings and errors) while it read the source image. A "
        "stack cut off in the middle is read with its missing pages as zeros and reported only "
        "there. At most five different messages, each without the file's folders, and a line "
        "that counts the rest; a record, not a cache key.",
    )
