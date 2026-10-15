"""Image decoding to a canonical 2D grayscale float array."""

from __future__ import annotations

import dataclasses
import logging
import math
import os
import re
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from skimage import color, io
from skimage.util import img_as_float

from .errors import (
    GelpyError,
    ImageDecodeError,
    ImageTooLargeError,
    InvalidValueError,
    NonFiniteImageError,
    UnsupportedImageError,
    shown,
)
from ..schemas.image import ImageMeta

_CHANNEL_INDEX = {"r": 0, "g": 1, "b": 2}
SKEW_FLOOR = 0.5
TAIL_ASYM_FLOOR = 0.1


def to_float01(arr: np.ndarray) -> np.ndarray:
    """Canonical float64 in [0,1]. Integer inputs scale by their dtype max; float inputs
    are min-max rescaled only if they fall outside [0,1]. Scaling is GLOBAL (one factor for
    the whole array), so an RGB image keeps its inter-channel ratios."""
    a = np.asarray(arr)
    if a.dtype == bool:
        return a.astype(np.float64)
    if np.issubdtype(a.dtype, np.floating):
        f = a.astype(np.float64, copy=False)
    else:
        f = np.asarray(img_as_float(a), dtype=np.float64)  # unsigned->[0,1], signed->[-1,1]
    lo, hi = float(f.min()), float(f.max())
    if lo < 0.0 or hi > 1.0:
        return (f - lo) / (hi - lo) if hi > lo else np.zeros_like(f)
    return f


def _drop_alpha(arr: np.ndarray) -> np.ndarray:
    """Strip a trailing alpha channel: RGBA -> RGB, and gray+alpha -> gray.

    Alpha is opacity, not signal. Only the 4-channel case used to be handled, so a 2-channel
    gray+alpha source averaged its alpha into the measurement: a uint8 (H, W, 2) image with
    gray 50 and alpha 255 canonicalized to (50/255 + 1.0)/2 = 0.598 instead of 0.196, i.e. the
    opacity dominated the intensity of every pixel.
    """
    if arr.ndim == 3 and arr.shape[2] in (2, 4):
        return arr[:, :, : arr.shape[2] - 1]
    return arr


def _collapse_with(arr: np.ndarray, strategy: str, index: int | None) -> np.ndarray:
    """Deterministic channel->grayscale collapse for a resolved (strategy, index)."""
    if arr.ndim == 2:
        return arr
    rgb = _drop_alpha(arr)
    if strategy in ("gray_rgb", "mean"):
        return rgb.mean(axis=2)
    if strategy == "luminance":
        return color.rgb2gray(rgb)
    if strategy == "max":
        return rgb.max(axis=2)
    if index is not None:
        return rgb[:, :, index]
    raise UnsupportedImageError(f"unknown channel strategy {shown(strategy)}")


def collapse_channels(
    arr: np.ndarray, strategy: str = "auto", *, gray_atol: float = 0.02
) -> tuple[np.ndarray, str, int | None]:
    """Collapse to 2-D grayscale, returning (gray, resolved_strategy, channel_index).

    A 2-D input is passthrough ('gray'). An RGB whose channels are equal within tolerance is
    'gray_rgb' (lossless mean). Otherwise 'auto' resolves to 'mean' (the least-biased default
    for a genuinely colored image, whose intensity->ground-truth mapping is unknowable from
    the image alone); 'r'/'g'/'b'/'dominant' pick a single channel (index recorded);
    'luminance'/'max' collapse.
    """
    if arr.ndim == 2:
        return arr, "gray", None
    rgb = _drop_alpha(arr)
    spread = rgb.max(axis=2) - rgb.min(axis=2)
    if float(np.percentile(spread, 99.9)) < gray_atol:
        return rgb.mean(axis=2), "gray_rgb", None
    resolved = "mean" if strategy == "auto" else strategy
    index: int | None = None
    if resolved == "dominant":
        spreads = [np.percentile(rgb[:, :, i], 99.5) - np.percentile(rgb[:, :, i], 0.5)
                   for i in range(rgb.shape[2])]
        index = int(np.argmax(spreads))
    elif resolved in _CHANNEL_INDEX:
        index = _CHANNEL_INDEX[resolved]
    return _collapse_with(rgb, resolved, index), resolved, index


@dataclass
class PolarityDecision:
    polarity: str          # "bright_on_dark" | "dark_on_light"
    inverted: bool
    confidence: str        # "ok" | "low"
    reason: str
    skew: float
    tail_asym: float


def detect_polarity(gray01: np.ndarray) -> PolarityDecision:
    """Detect signal polarity from two independent, deterministic cues; agreement = confidence.

    Background occupies most of the image area, so the LONG tail of the histogram is signal.
    Cue 1: sign of Fisher skewness. Cue 2: which tail reaches further from the median. Both
    positive -> bright-on-dark. They must agree AND be non-trivial to earn 'ok'; otherwise we
    do NOT invert (the reversible default) and flag 'low'. Inversion is applied only on a
    confident dark-on-light.
    """
    x = np.asarray(gray01, dtype=np.float64).ravel()
    mu, sd = float(x.mean()), float(x.std())
    g1 = float(((x - mu) ** 3).mean() / (sd**3)) if sd > 0 else 0.0
    p_lo, m, p_hi = (float(v) for v in np.percentile(x, [0.5, 50, 99.5]))
    bright_reach, dark_reach = p_hi - m, m - p_lo
    denom = bright_reach + dark_reach
    tail_asym = (bright_reach - dark_reach) / denom if denom > 0 else 0.0
    strong = abs(g1) >= SKEW_FLOOR or abs(tail_asym) >= TAIL_ASYM_FLOOR
    agree = (g1 > 0) == (tail_asym > 0)
    if agree and strong:
        if g1 > 0:
            return PolarityDecision("bright_on_dark", False, "ok", "", g1, tail_asym)
        return PolarityDecision("dark_on_light", True, "ok", "", g1, tail_asym)
    reason = (
        "polarity cues (histogram skew and tail asymmetry) disagree or are too weak "
        "(near-symmetric intensity); assuming bright-on-dark. Set --invert / --no-invert if wrong."
    )
    return PolarityDecision("bright_on_dark", False, "low", reason, g1, tail_asym)


#: Caps on what one decode may materialise, checked from the file header before any pixel data
#: is read. The environment variables raise or lower them for one process.
DEFAULT_MAX_PIXELS = 100_000_000
DEFAULT_MAX_PAGES = 64
MAX_PIXELS_ENV = "GELPY_MAX_PIXELS"
MAX_PAGES_ENV = "GELPY_MAX_PAGES"

_TIFF_MAGIC = (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")  # classic and BigTIFF


@dataclass(frozen=True)
class ImageProbe:
    """What a file's header says it holds, read without decoding pixel data."""

    pages: int  # every 2-D plane the decoder would materialise: stack axes multiply
    pixels: int  # width x height summed over those planes; a channel axis is not counted
    width: int  # of the first plane
    height: int
    notes: tuple[str, ...] = ()  # what tifffile logged (warnings and errors) while it read the header


def _limit(name: str, default: int, unit: str) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    # ASCII digits only, and at most 18 of them: int() alone would also take "1_000", "+5" and the
    # digits of other scripts, and refuses a string of over 4300 digits with an error of its own.
    if not re.fullmatch(r"[0-9]{1,18}", raw) or int(raw) < 1:
        raise InvalidValueError(
            f"{name}={shown(raw)} is not a positive whole number of {unit}.",
            detail=(
                f"Set {name} to a whole number of 1 to 18 digits written with the digits 0-9, "
                f"or unset it to use {default}."
            ),
        )
    return int(raw)


def image_limits() -> tuple[int, int]:
    """``(max_pixels, max_pages)``: from GELPY_MAX_PIXELS / GELPY_MAX_PAGES, else the defaults."""
    return (
        _limit(MAX_PIXELS_ENV, DEFAULT_MAX_PIXELS, "pixels"),
        _limit(MAX_PAGES_ENV, DEFAULT_MAX_PAGES, "pages"),
    )


_pillow_guard_lock = threading.Lock()
_pillow_guard_readers = 0
_pillow_guard_saved: int | None = None


@contextmanager
def _pillow_guard_off() -> Iterator[None]:
    """Switch Pillow's decompression-bomb guard off while gelpy reads a file, then put it back.

    gelpy's own caps (``check_limits``) decide what is too large, from the header, in gelpy's
    own figures. Pillow's guard is a second, differently sized opinion held in a process-wide
    setting (``Image.MAX_IMAGE_PIXELS``): it warns on stderr above 89 million pixels, refuses
    above twice that, quotes its own limit, and would refuse an image the caller raised
    ``GELPY_MAX_PIXELS`` to accept. The setting is restored on the way out, whether the read
    worked or not. The first reader in saves it and the last one out puts it back, so nested or
    concurrent reads cannot leave the guard off.
    """
    from PIL import Image

    global _pillow_guard_readers, _pillow_guard_saved
    with _pillow_guard_lock:
        if _pillow_guard_readers == 0:
            _pillow_guard_saved = Image.MAX_IMAGE_PIXELS
            Image.MAX_IMAGE_PIXELS = None
        _pillow_guard_readers += 1
    try:
        yield
    finally:
        with _pillow_guard_lock:
            _pillow_guard_readers -= 1
            if _pillow_guard_readers == 0:
                Image.MAX_IMAGE_PIXELS = _pillow_guard_saved


@contextmanager
def _tifffile_notes() -> Iterator[list[str]]:
    """Take tifffile's log records while a file is read; the warnings and errors are listed.

    tifffile reports a damaged TIFF by logging a warning (``invalid offset to first page`` for a
    file that is only a header) and then returns nothing, raises, or, for a stack cut off in the
    middle, returns what it could read with the missing pages as zeros. With no logging configured,
    Python prints that line on stderr ahead of gelpy's own error object, so stderr would no longer be
    exactly one JSON object. So the records never reach a handler: the warnings and errors are
    appended to the list this yields, for the caller to keep (``ImageMeta.reader_warnings``), and
    nothing is printed. Only the ``tifffile`` logger is touched, only while the file is read. Each
    call adds a filter of its own and removes that one, so nested or concurrent reads never restore
    each other's state, and no other logger or handler is affected.
    """
    logger = logging.getLogger("tifffile")
    notes: list[str] = []

    def take(record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            try:
                notes.append(record.getMessage())
            except Exception:  # a message that cannot be formatted is still worth the line
                notes.append(str(record.msg))
        return False

    logger.addFilter(take)
    try:
        yield notes
    finally:
        logger.removeFilter(take)


def _without_path(text: str, file: str | Path) -> str:
    """``text`` with every spelling of the path of ``file`` replaced by its name.

    A reader's message names the file it was given: the folders it lies in are private information in
    a message that is logged, pasted into a report and sent on, and add nothing to the file's name.
    """
    path = Path(file)
    if not path.name:
        return text
    spellings = {str(file), str(path), os.path.abspath(path)}
    try:
        spellings.add(str(path.resolve()))
    except OSError:  # a path that cannot be resolved has no further spelling
        pass
    for spelling in sorted(spellings, key=len, reverse=True):
        text = text.replace(spelling, path.name)
    return text


def _cause(exc: Exception, file: str | Path | None = None) -> str:
    """``ExceptionType: first line of its message``, without the path of ``file``.

    One line of the decoder's text only: imageio's "Could not find a backend" message runs to
    several lines and ends in "pip install" hints, which are no next step for this tool.
    """
    reason = next((line.strip() for line in str(exc).splitlines() if line.strip()), "")
    if file is not None:
        reason = _without_path(reason, file)
    return f"{type(exc).__name__}: {reason}" if reason else type(exc).__name__


#: How many different messages a read keeps; the rest are counted.
MAX_READER_NOTES = 5
_NOTE_LENGTH = 200


def _keep_notes(messages: Iterable[str], file: str | Path) -> list[str]:
    """The first ``MAX_READER_NOTES`` different messages, each without the file's path and cut at
    200 characters, then a line that counts the rest."""
    kept: list[str] = []
    for message in messages:
        text = _without_path(message, file).strip()
        if len(text) > _NOTE_LENGTH:
            text = text[:_NOTE_LENGTH] + "..."
        if text and text not in kept:
            kept.append(text)
    if len(kept) > MAX_READER_NOTES:
        kept = [*kept[:MAX_READER_NOTES], f"(and {len(kept) - MAX_READER_NOTES} more)"]
    return kept


def _probe_shape(shape: tuple[int, ...]) -> ImageProbe:
    """Pages and pixels of the array ``decode_image`` holds for a file of this shape.

    ``skimage.io.imread`` moves a planar channel axis last, (..., C, H, W) -> (..., H, W, C),
    when the last axis is not 3 or 4 and the third from last is. After that a last axis of 2, 3
    or 4 on three or more axes is the channel axis, as in ``decode_image``. Pixels are the
    product of the other axes and pages the product of those before the last two, so every 2-D
    plane counts, whatever the leading axes mean (pages, time points, z slices)."""
    if len(shape) > 2 and shape[-1] not in (3, 4) and shape[-3] in (3, 4):
        shape = (*shape[:-3], *shape[-2:], shape[-3])
    planes = shape[:-1] if len(shape) > 2 and shape[-1] in (2, 3, 4) else shape
    height, width = (planes[-2], planes[-1]) if len(planes) > 1 else (planes[0], 1)
    return ImageProbe(
        pages=math.prod(planes[:-2]), pixels=math.prod(planes), width=width, height=height
    )


def _probe_tiff(file: Path) -> ImageProbe:
    import tifffile

    with tifffile.TiffFile(file) as tif:
        series = tif.series
        if not series:  # a header with no page behind it
            raise ImageDecodeError(f"{file.name!r} is a TIFF file that holds no image.")
        first = series[0]
        if getattr(first, "kind", "") == "shaped" and callable(series):
            # tifffile 2026.5.2 and later report every series without its length-1 axes, but decode a
            # 'shaped' one (what a plain tifffile write is) with all the axes it was written with:
            # (4, 1, Y, X) is four pages of Y x X, and without the length-1 axis it would read as one
            # colour image of Y x X pixels. Earlier releases already report the axes of a shaped series.
            first = series(squeeze=False)[0]
        # The shape decoding will return, not the IFD count: tifffile stores an (H, W, 2) array
        # as H pages of W x 2 pixels, which is a single two-channel image.
        shape = tuple(int(size) for size in first.shape)
    return _probe_shape(shape)


def _probe_pillow(file: Path) -> ImageProbe:
    from PIL import Image

    with _pillow_guard_off(), Image.open(file) as image:
        width, height = image.size
        if image.format == "GIF":
            # Pillow reports the logical screen, grown to the first frame when that is larger, and
            # counts frames, but never sees a later frame larger than the screen, and the decoder
            # reads every frame at its own size: refuse any GIF with more than one frame.
            if image.is_animated:
                raise ImageDecodeError(
                    f"{file.name!r} is a GIF with more than one frame.",
                    detail="Export the frame you want as PNG or TIFF and use that file.",
                )
            pages = 1
        else:
            pages = int(getattr(image, "n_frames", 1))
    return ImageProbe(pages=pages, pixels=pages * width * height, width=width, height=height)


def probe_image(path: str) -> ImageProbe:
    """Read the header of ``path`` -- page count and pixel count -- without decoding pixels.

    TIFF files are read with tifffile, everything else with Pillow. A file neither can read
    raises ``ImageDecodeError``, and so does a GIF with more than one frame, whose later frames
    can be far larger than its header says.
    """
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"No image file at {shown(path)}.")
    with open(file, "rb") as handle:
        magic = handle.read(4)
    with _tifffile_notes() as notes:
        try:
            probe = _probe_tiff(file) if magic in _TIFF_MAGIC else _probe_pillow(file)
        except GelpyError:
            raise
        except Exception as exc:  # a header no reader understands
            raise ImageDecodeError(
                f"Could not read the header of {file.name!r} ({_cause(exc, file)})."
            ) from exc
    return dataclasses.replace(probe, notes=tuple(notes))


def check_limits(probe: ImageProbe, name: str) -> None:
    """Refuse a file over the page or pixel cap (``image_limits``)."""
    max_pixels, max_pages = image_limits()
    if probe.pages > max_pages:
        raise ImageTooLargeError(
            f"{name!r} has {probe.pages} pages; the limit is {max_pages}.",
            detail=f"Export the page you need as its own file, or raise the limit with {MAX_PAGES_ENV}.",
        )
    if probe.pixels > max_pixels:
        raise ImageTooLargeError(
            f"{name!r} would decode to {probe.pixels:,} pixels; the limit is {max_pixels:,}.",
            detail=f"Crop or downsample the image, or raise the limit with {MAX_PIXELS_ENV}.",
        )


def _drop_length_one_axes(a: np.ndarray) -> np.ndarray:
    """``a`` without its length-1 axes other than Y and X (and the channel axis of a colour image).

    The header probe counts a (4, 1, Y, X) file as four pages and (1, 1, Y, X) as one; without this the
    decoder would read both as a 1 x Y x X stack. A (Y, X, 1) gel is a plain Y x X image."""
    if a.ndim >= 3 and a.shape[-1] == 1:
        a = a[..., 0]
    keep_tail = 3 if a.ndim > 2 and a.shape[-1] in (2, 3, 4) else 2
    lead = a.shape[: a.ndim - keep_tail]
    if 1 in lead:
        a = a.reshape(tuple(n for n in lead if n != 1) + a.shape[a.ndim - keep_tail :])
    return a


def _read_image(path: str, *, page: int = 0) -> tuple[np.ndarray, int, list[str]]:
    """``decode_image``, and what tifffile logged (warnings and errors) while it read the file."""
    probe = probe_image(path)  # raises FileNotFoundError / ImageDecodeError
    check_limits(probe, Path(path).name)
    try:
        with _pillow_guard_off(), _tifffile_notes() as notes:
            a = np.asarray(io.imread(path))
    except MemoryError:
        raise
    except Exception as exc:  # every decoder failure becomes one error with a next step
        raise ImageDecodeError(
            f"Could not decode {Path(path).name!r} as an image ({_cause(exc, path)})."
        ) from exc
    logged = [*probe.notes, *notes]
    a = _drop_length_one_axes(a)
    if a.ndim <= 2:
        return a, 1, logged
    if a.ndim == 3 and a.shape[-1] in (2, 3, 4):
        return a, 1, logged
    if a.ndim in (3, 4):
        n = int(a.shape[0])
        return a[min(page, n - 1)], n, logged
    raise UnsupportedImageError(f"unsupported image ndim={a.ndim}, shape={a.shape}")


def decode_image(path: str, *, page: int = 0) -> tuple[np.ndarray, int]:
    """Decode to a single 2-D/3-D(HxWxC) frame, returning (array, source_pages).

    A multi-page/stacked source (ndim 3 whose last axis is not a channel count, or ndim 4)
    is treated as pages along axis 0; ``page`` selects one (clamped). The (P,H,W) vs (H,W,3)
    ambiguity is resolved in favor of channels when the last axis is in {2,3,4}. A path that
    is not a file raises ``FileNotFoundError``; anything the decoder cannot read raises
    ``ImageDecodeError`` instead of the decoder's own exception. The header is checked against
    the pixel and page caps (``image_limits``) before any pixel data is decoded, so an oversized
    or many-page file is refused, not loaded."""
    array, pages, _ = _read_image(path, page=page)
    return array, pages


def _check_finite(arr: np.ndarray) -> None:
    a = np.asarray(arr)
    if not np.isfinite(a).all():
        n = int((~np.isfinite(a)).sum())
        raise NonFiniteImageError(
            f"image contains {n} non-finite (NaN/Inf) pixels; clean the source before analysis."
        )


def canonicalize_image(
    path: str, *, channel: str = "auto", invert: bool | str = "auto", page: int = 0
) -> tuple[np.ndarray, ImageMeta]:
    """Decode + canonicalize to a signal-bright float64 [0,1] 2-D array, plus the frozen
    ``ImageMeta`` describing the interpretation. Detection runs here (init path)."""
    raw, source_pages, logged = _read_image(path, page=page)
    _check_finite(raw)
    original_dtype = str(np.asarray(raw).dtype)
    original_shape = tuple(int(s) for s in np.asarray(raw).shape)
    source_channels = 1 if raw.ndim == 2 else int(raw.shape[-1])
    gray, strategy, index = collapse_channels(to_float01(raw), channel)

    if invert is True:
        polarity, inverted, source, confidence, reason = "dark_on_light", True, "manual", "ok", ""
    elif invert is False:
        polarity, inverted, source, confidence, reason = "bright_on_dark", False, "manual", "ok", ""
    else:
        d = detect_polarity(gray)
        polarity, inverted, source = d.polarity, d.inverted, "auto"
        confidence, reason = d.confidence, d.reason

    canonical = 1.0 - gray if inverted else gray
    meta = ImageMeta(
        original_dtype=original_dtype, original_shape=original_shape,
        source_channels=source_channels, source_pages=source_pages, page=page,
        channel_strategy=strategy, channel_index=index, polarity=polarity, inverted=inverted,
        polarity_source=source, polarity_confidence=confidence, polarity_reason=reason,
        reader_warnings=_keep_notes(logged, path),
    )
    return np.ascontiguousarray(canonical, dtype=np.float64), meta


def apply_image_meta(path: str, meta: ImageMeta) -> np.ndarray:
    """Deterministic replay of a frozen ``ImageMeta`` — NO detection. Used by the cached
    pipeline so the same recipe always yields the same canonical pixels."""
    raw, _ = decode_image(path, page=meta.page)
    _check_finite(raw)
    gray = _collapse_with(to_float01(raw), meta.channel_strategy, meta.channel_index)
    canonical = 1.0 - gray if meta.inverted else gray
    return np.ascontiguousarray(canonical, dtype=np.float64)


def load_image(path: str) -> np.ndarray:
    """Backward-compatible convenience: auto-canonicalized array only (no metadata)."""
    return canonicalize_image(path)[0]
