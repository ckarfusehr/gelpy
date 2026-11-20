"""Orientation and crop of a decoded 2-D image: quarter turns, mirror, fine rotation, crop.

Quarter turns, the mirror and the crop are pure pixel copies: same dtype, same values, no
interpolation. The fine rotation interpolates bilinearly (first order): every output pixel is a
weighted mean of at most four input pixels with non-negative weights, so no value falls below the
input's minimum or rises above its maximum (no ringing, no negative intensities). It rotates
about the image centre onto a canvas padded evenly on both sides, just large enough to hold every
source pixel centre; output pixels whose source position lies outside the source take ``fill``
(default: the image's minimum, which is "no signal" in gelpy's signal-bright arrays). The result
keeps the input's dtype: integers are rounded to the nearest value, ties to even (``numpy.rint``);
floats are computed in float64 and cast back.

Angles and turns are counter-clockwise as the image is displayed (row 0 at the top), as in
``numpy.rot90`` and ``scipy.ndimage.rotate``.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from ..schemas.image import ImageGeometry
from .errors import InvalidValueError

#: Slack for the canvas size: a padding that floating point gives as 2.0000000001 px is 2 px.
_EPS = 1e-9


def quarter_turn(image: np.ndarray, turn: int) -> np.ndarray:
    """``image`` turned by ``turn`` degrees (a multiple of 90) counter-clockwise: an exact copy."""
    if turn % 90:
        raise InvalidValueError(f"turn must be a multiple of 90 degrees, got {turn}")
    return np.rot90(image, (turn // 90) % 4).copy(order="C")


def mirror(image: np.ndarray) -> np.ndarray:
    """``image`` mirrored left-right: an exact copy."""
    return image[:, ::-1].copy(order="C")


def rotated_shape(height: int, width: int, angle: float) -> tuple[int, int]:
    """(height, width) of the canvas a fine rotation by ``angle`` writes: the source plus an even
    padding on each side, the least that holds every rotated source pixel centre."""
    t = math.radians(angle)
    c, s = abs(math.cos(t)), abs(math.sin(t))
    span_w = (width - 1) * c + (height - 1) * s
    span_h = (width - 1) * s + (height - 1) * c
    pad_w = max(0, math.ceil((span_w + 1 - width) / 2 - _EPS))
    pad_h = max(0, math.ceil((span_h + 1 - height) / 2 - _EPS))
    return height + 2 * pad_h, width + 2 * pad_w


def _rotation(height: int, width: int, angle: float) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """The (matrix, offset) that map an output (row, col) to its source (row, col), and the
    output shape: the inverse of a counter-clockwise rotation about the centre."""
    out = rotated_shape(height, width, angle)
    t = math.radians(angle)
    c, s = math.cos(t), math.sin(t)
    matrix = np.array([[c, s], [-s, c]])
    source_centre = np.array([(height - 1) / 2, (width - 1) / 2])
    out_centre = np.array([(out[0] - 1) / 2, (out[1] - 1) / 2])
    return matrix, source_centre - matrix @ out_centre, out


def fine_rotate(image: np.ndarray, angle: float, *, fill: float | None = None) -> np.ndarray:
    """``image`` (2-D) rotated by ``angle`` degrees counter-clockwise, bilinearly, onto the
    ``rotated_shape`` canvas; pixels from outside the source take ``fill`` (default the
    image's minimum). Same dtype as the input; an angle of 0 returns an exact copy."""
    a = np.asarray(image)
    if a.ndim != 2:
        raise InvalidValueError(f"a fine rotation needs a 2-D image, got shape {a.shape}")
    if angle == 0:
        return a.copy()
    lo, hi = float(a.min()), float(a.max())
    fill = lo if fill is None else float(fill)
    matrix, offset, shape = _rotation(a.shape[0], a.shape[1], angle)
    # order=1 is bilinear; mode="constant" gives a position outside the source grid exactly
    # `cval` and interpolates one inside it from source pixels only (scipy >= 1.6).
    out = ndimage.affine_transform(
        a.astype(np.float64), matrix, offset=offset, output_shape=shape,
        order=1, mode="constant", cval=fill, prefilter=False,
    )
    # A weighted mean stays within its inputs; the clip only removes floating-point last bits.
    np.clip(out, min(lo, fill), max(hi, fill), out=out)
    if np.issubdtype(a.dtype, np.integer):
        out = np.rint(out)
    return out.astype(a.dtype)


def source_mask(height: int, width: int, angle: float) -> np.ndarray:
    """Boolean mask on the ``rotated_shape`` canvas: True where a fine rotation by ``angle``
    takes the pixel from the source, False where it writes the fill."""
    ones = np.ones((height, width), dtype=np.uint8)
    return fine_rotate(ones, angle, fill=0).astype(bool)


def oriented_shape(height: int, width: int, geometry: ImageGeometry) -> tuple[int, int]:
    """(height, width) of a ``height`` x ``width`` image after ``geometry``'s turn, flip and
    angle, before the crop: the frame a crop box is given in."""
    if geometry.turn in (90, 270):
        height, width = width, height
    return rotated_shape(height, width, geometry.angle) if geometry.angle else (height, width)


def analysed_shape(height: int, width: int, geometry: ImageGeometry) -> tuple[int, int]:
    """(height, width) of a ``height`` x ``width`` image after the whole ``geometry``; a crop
    box outside the oriented image is refused (``check_crop``)."""
    shape = oriented_shape(height, width, geometry)
    if geometry.crop is None:
        return shape
    check_crop(geometry.crop, shape)
    x0, y0, x1, y1 = geometry.crop
    return y1 - y0, x1 - x0


def check_crop(box: tuple[int, int, int, int], shape: tuple[int, int]) -> None:
    """Refuse a crop box that does not lie inside an image of ``shape`` (height, width)."""
    x0, y0, x1, y1 = box
    height, width = shape
    problems = [
        *([f"x0 = {x0} is negative"] if x0 < 0 else []),
        *([f"y0 = {y0} is negative"] if y0 < 0 else []),
        *([f"x1 = {x1} is beyond the width {width}"] if x1 > width else []),
        *([f"y1 = {y1} is beyond the height {height}"] if y1 > height else []),
        *(["x1 must be larger than x0"] if x1 <= x0 else []),
        *(["y1 must be larger than y0"] if y1 <= y0 else []),
    ]
    if problems:
        raise InvalidValueError(
            f"crop box [{x0}, {y0}, {x1}, {y1}] does not fit the oriented image "
            f"({width} x {height} px): {'; '.join(problems)}.",
            detail="The box is [x0, y0, x1, y1] in pixels of the oriented image (after the turn, "
            "the flip and the angle): x0, y0 the first column and row kept, x1, y1 one past the "
            "last, so 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height. The coordinate grid "
            "(`gelpy image grid`) shows the oriented image with its pixel coordinates.",
        )


def crop(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """The pixels of ``image`` inside ``box`` [x0, y0, x1, y1] (x1, y1 exclusive): an exact copy."""
    check_crop(box, image.shape[:2])
    x0, y0, x1, y1 = box
    return image[y0:y1, x0:x1].copy(order="C")


def orient(image: np.ndarray, geometry: ImageGeometry, *, fill: float | None = None) -> np.ndarray:
    """``image`` after ``geometry``'s turn, flip and angle, in that order, without the crop.
    Always a new array."""
    out = quarter_turn(image, geometry.turn)
    if geometry.flip:
        out = mirror(out)
    if geometry.angle:
        out = fine_rotate(out, geometry.angle, fill=fill)
    return out


def apply_geometry(image: np.ndarray, geometry: ImageGeometry) -> np.ndarray:
    """``image`` after the whole ``geometry``: turn, flip, angle, then crop. Always a new array."""
    out = orient(image, geometry)
    return crop(out, geometry.crop) if geometry.crop is not None else out
