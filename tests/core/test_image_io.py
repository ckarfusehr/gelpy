import numpy as np
import pytest

from gelpy.core.image_io import collapse_channels, detect_polarity, to_float01


def test_to_float01_uint8_scales_by_255():
    arr = np.array([[0, 255], [128, 64]], dtype=np.uint8)
    out = to_float01(arr)
    assert out.dtype == np.float64
    assert out.min() == 0.0 and out.max() == 1.0


def test_to_float01_uint16_scales_by_65535():
    arr = np.array([[0, 65535]], dtype=np.uint16)
    assert np.allclose(to_float01(arr), [[0.0, 1.0]])


def test_to_float01_float_out_of_range_minmax_rescales():
    arr = np.array([[10.0, 20.0, 30.0]], dtype=np.float64)
    assert np.allclose(to_float01(arr), [[0.0, 0.5, 1.0]])


def test_to_float01_preserves_channel_ratios_global_scale():
    # A 2x1 RGB image; global (not per-channel) scaling keeps color ratios intact.
    arr = np.zeros((1, 1, 3), dtype=np.uint8)
    arr[0, 0] = [100, 200, 50]
    out = to_float01(arr)
    assert np.allclose(out[0, 0], [100 / 255, 200 / 255, 50 / 255])


def test_collapse_gray_rgb_is_lossless_mean():
    g = np.tile(np.linspace(0, 1, 10)[None, :, None], (4, 1, 3))  # R==G==B
    out, strategy, idx = collapse_channels(g)
    assert strategy == "gray_rgb" and idx is None
    assert np.allclose(out, g[:, :, 0])


def test_collapse_colored_defaults_to_mean():
    c = np.zeros((2, 2, 3))
    c[..., 0] = 0.9  # strong red, differs across channels
    out, strategy, idx = collapse_channels(c)
    assert strategy == "mean" and idx is None
    assert np.allclose(out, 0.3)


def test_collapse_channel_override_records_index():
    c = np.zeros((2, 2, 3))
    c[..., 2] = 0.8
    out, strategy, idx = collapse_channels(c, "b")
    assert strategy == "b" and idx == 2 and np.allclose(out, 0.8)


def test_collapse_2d_passthrough():
    g = np.linspace(0, 1, 12).reshape(3, 4)
    out, strategy, idx = collapse_channels(g)
    assert strategy == "gray" and idx is None and np.allclose(out, g)


def _bright_on_dark(width=400, height=200, sd=6.0):
    img = np.full((height, width), 0.05)
    x = np.arange(width)
    for c in (80, 200, 320):
        img += np.exp(-((x - c) ** 2) / (2 * sd**2))[None, :]
    return np.clip(img, 0, 1)


def test_detect_bright_on_dark():
    d = detect_polarity(_bright_on_dark())
    assert d.polarity == "bright_on_dark" and d.inverted is False and d.confidence == "ok"


def test_detect_dark_on_light_is_flagged_for_inversion():
    d = detect_polarity(1.0 - _bright_on_dark())
    assert d.polarity == "dark_on_light" and d.inverted is True and d.confidence == "ok"


def test_detect_near_symmetric_is_low_confidence_and_not_inverted():
    rng = np.random.default_rng(0)
    noise = np.clip(0.5 + 0.05 * rng.standard_normal((200, 400)), 0, 1)
    d = detect_polarity(noise)
    assert d.confidence == "low" and d.inverted is False


from skimage import io as _skio

from gelpy.core.errors import NonFiniteImageError
from gelpy.core.image_io import apply_image_meta, canonicalize_image


def _save(tmp_path, name, arr):
    p = tmp_path / name
    _skio.imsave(str(p), arr, check_contrast=False)
    return str(p)


def test_canonicalize_uint16_bright_on_dark(tmp_path):
    gel = (_bright_on_dark() * 65535).astype(np.uint16)
    img, meta = canonicalize_image(_save(tmp_path, "g.tif", gel))
    assert img.dtype == np.float64 and img.min() >= 0.0 and img.max() <= 1.0
    assert meta.polarity == "bright_on_dark" and meta.inverted is False
    assert meta.original_dtype == "uint16" and meta.channel_strategy == "gray"


def test_canonicalize_inverts_dark_on_light(tmp_path):
    gel = ((1.0 - _bright_on_dark()) * 255).astype(np.uint8)
    img, meta = canonicalize_image(_save(tmp_path, "d.png", gel))
    assert meta.inverted is True and meta.polarity == "dark_on_light"
    # after canonicalization the band columns are the bright ones again
    assert img[:, 80].mean() > img[:, 5].mean()


def test_manual_invert_override_sets_source_manual(tmp_path):
    gel = (_bright_on_dark() * 255).astype(np.uint8)
    _, meta = canonicalize_image(_save(tmp_path, "g.png", gel), invert=True)
    assert meta.inverted is True and meta.polarity_source == "manual"


def test_apply_image_meta_reproduces_canonical(tmp_path):
    gel = ((1.0 - _bright_on_dark()) * 255).astype(np.uint8)
    p = _save(tmp_path, "d.png", gel)
    img, meta = canonicalize_image(p)
    assert np.allclose(apply_image_meta(p, meta), img)


def test_nonfinite_raises(tmp_path):
    bad = _bright_on_dark().astype(np.float32)
    bad[0, 0] = np.nan
    with pytest.raises(NonFiniteImageError):
        canonicalize_image(_save(tmp_path, "nan.tif", bad))


def test_alpha_is_never_measured_as_signal(tmp_path):
    """A trailing alpha channel must be stripped, for 2-channel sources as well as RGBA.

    Alpha is opacity, not signal. Only the 4-channel case was handled, so a gray+alpha source
    averaged its alpha into every measurement: uint8 gray 50 with alpha 255 canonicalized to
    (50/255 + 1.0)/2 = 0.598 instead of 0.196, with the opacity dominating the intensity.
    """
    gel = (_bright_on_dark() * 255).astype(np.uint8)
    opaque = np.full_like(gel, 255)

    plain, _ = canonicalize_image(_save(tmp_path, "gray.tif", gel))
    gray_alpha, meta_ga = canonicalize_image(
        _save(tmp_path, "gray_alpha.tif", np.dstack([gel, opaque]))
    )
    rgba, meta_rgba = canonicalize_image(
        _save(tmp_path, "rgba.tif", np.dstack([gel, gel, gel, opaque]))
    )

    assert meta_ga.source_channels == 2
    assert meta_rgba.source_channels == 4
    # Adding an opaque alpha channel must not change a single pixel.
    assert np.allclose(plain, gray_alpha)
    assert np.allclose(plain, rgba)


def test_a_varying_alpha_channel_is_still_ignored(tmp_path):
    """Even a non-constant alpha must not leak into the measurement."""
    gel = (_bright_on_dark() * 255).astype(np.uint8)
    ramp = np.tile(np.linspace(0, 255, gel.shape[1], dtype=np.uint8), (gel.shape[0], 1))

    plain, _ = canonicalize_image(_save(tmp_path, "gray2.tif", gel))
    with_ramp, _ = canonicalize_image(
        _save(tmp_path, "gray_ramp_alpha.tif", np.dstack([gel, ramp]))
    )
    assert np.allclose(plain, with_ramp)
