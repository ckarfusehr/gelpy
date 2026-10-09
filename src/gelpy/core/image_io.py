"""Image decoding to a canonical 2D grayscale float array."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage import color, io
from skimage.util import img_as_float

from .errors import NonFiniteImageError, UnsupportedImageError
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
    raise UnsupportedImageError(f"unknown channel strategy {strategy!r}")


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


def decode_image(path: str, *, page: int = 0) -> tuple[np.ndarray, int]:
    """Decode to a single 2-D/3-D(HxWxC) frame, returning (array, source_pages).

    A multi-page/stacked source (ndim 3 whose last axis is not a channel count, or ndim 4)
    is treated as pages along axis 0; ``page`` selects one (clamped). The (P,H,W) vs (H,W,3)
    ambiguity is resolved in favor of channels when the last axis is in {2,3,4}."""
    a = np.asarray(io.imread(path))
    if a.ndim <= 2:
        return a, 1
    if a.ndim == 3 and a.shape[-1] in (2, 3, 4):
        return a, 1
    if a.ndim in (3, 4):
        n = int(a.shape[0])
        return a[min(page, n - 1)], n
    raise UnsupportedImageError(f"unsupported image ndim={a.ndim}, shape={a.shape}")


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
    raw, source_pages = decode_image(path, page=page)
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
