"""Oversized and many-page images are refused from the header, before any pixel is decoded."""

import logging
import struct
import warnings
import zlib
from io import BytesIO

import numpy as np
import pytest
import tifffile
from PIL import Image
from skimage import io as skimage_io

from gelpy.core import image_io
from gelpy.core.errors import ImageDecodeError, ImageTooLargeError, InvalidValueError
from gelpy.core.image_io import (
    DEFAULT_MAX_PAGES,
    DEFAULT_MAX_PIXELS,
    decode_image,
    image_limits,
    probe_image,
)


def _png_header_only(path, width, height):
    """A PNG whose header declares width x height but whose data is a few bytes (a 'bomb')."""

    def chunk(kind, data):
        crc = struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + kind + data + crc

    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)  # 8-bit grayscale
    data = zlib.compress(b"\x00" * 16)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", data) + chunk(b"IEND", b"")
    )
    return path


def _gif_with_frames(screen, frames):
    """A GIF with this logical screen and one image descriptor per (width, height) in `frames`.

    The frame data is a stub: enough for a reader that walks the headers, not for a decoder.
    A frame may be far larger than the screen, which the format allows."""
    out = b"GIF89a" + struct.pack("<HHBBB", *screen, 0x80, 0, 0) + b"\x00\x00\x00\xff\xff\xff"
    for width, height in frames:
        out += b"\x2c" + struct.pack("<HHHHB", 0, 0, width, height, 0)  # image descriptor
        out += b"\x02" + b"\x02\x44\x01" + b"\x00"  # LZW code size, one data block, terminator
    return out + b"\x3b"


def _single_frame_gif(path, frame_size, screen, offset=(0, 0)):
    """A decodable one-frame GIF (LZW data from Pillow) whose screen and frame position are
    rewritten, so the frame can be larger than the screen or sit away from the corner."""
    buffer = BytesIO()
    Image.new("P", frame_size).save(buffer, format="GIF")
    data = bytearray(buffer.getvalue())
    struct.pack_into("<HH", data, 6, *screen)  # logical screen width, height
    position = 13 + (3 * 2 ** ((data[10] & 7) + 1) if data[10] & 0x80 else 0)
    while data[position] == 0x21:  # skip extension blocks: introducer, label, data blocks
        position += 2
        while data[position] != 0:
            position += data[position] + 1
        position += 1
    assert data[position] == 0x2C  # the image descriptor
    struct.pack_into("<HH", data, position + 1, *offset)  # frame left, top
    path.write_bytes(bytes(data))
    return path


@pytest.fixture(autouse=True)
def _default_limits(monkeypatch):
    monkeypatch.delenv("GELPY_MAX_PIXELS", raising=False)
    monkeypatch.delenv("GELPY_MAX_PAGES", raising=False)


@pytest.fixture
def decoded(monkeypatch):
    """What the decoder was called with: a refusal from the header must leave this empty."""
    calls = []
    monkeypatch.setattr(image_io.io, "imread", lambda *a, **k: calls.append(a))
    return calls


def test_the_defaults_apply_without_environment_variables():
    assert image_limits() == (DEFAULT_MAX_PIXELS, DEFAULT_MAX_PAGES)


def test_a_header_declaring_too_many_pixels_is_refused_before_decoding(tmp_path, monkeypatch):
    bomb = _png_header_only(tmp_path / "bomb.png", 40_000, 40_000)
    decoded = []
    monkeypatch.setattr(image_io.io, "imread", lambda *a, **k: decoded.append(a))
    with pytest.raises(ImageTooLargeError) as info:
        decode_image(str(bomb))
    assert decoded == []  # the decoder never ran
    assert "GELPY_MAX_PIXELS" in (info.value.detail or ImageTooLargeError.next_step)


def test_the_page_cap_counts_pages_from_the_header(tmp_path, monkeypatch, decoded):
    stack = tmp_path / "stack.tif"
    tifffile.imwrite(stack, np.zeros((6, 8, 8), "uint8"), photometric="minisblack")
    monkeypatch.setenv("GELPY_MAX_PAGES", "4")
    with pytest.raises(ImageTooLargeError) as info:
        decode_image(str(stack))
    assert decoded == []  # the decoder never ran
    assert "6 pages" in str(info.value)
    assert "GELPY_MAX_PAGES" in info.value.detail


def test_the_pixel_cap_is_configurable(tmp_path, monkeypatch):
    image = tmp_path / "small.png"
    Image.fromarray(np.zeros((20, 30), "uint8")).save(image)
    monkeypatch.setenv("GELPY_MAX_PIXELS", "599")
    with pytest.raises(ImageTooLargeError):
        decode_image(str(image))
    monkeypatch.setenv("GELPY_MAX_PIXELS", "600")
    array, pages = decode_image(str(image))
    assert array.shape == (20, 30) and pages == 1


def test_the_probe_reads_pages_and_size_without_pixels(tmp_path):
    stack = tmp_path / "stack.tif"
    tifffile.imwrite(stack, np.zeros((5, 10, 12), "uint16"), photometric="minisblack")
    probe = probe_image(str(stack))
    assert (probe.pages, probe.width, probe.height, probe.pixels) == (5, 12, 10, 600)


def test_a_two_channel_image_is_one_page(tmp_path):
    """tifffile stores an (H, W, 2) array as H pages of W x 2 pixels; decoded, it is one image."""
    path = tmp_path / "gray_alpha.tif"
    tifffile.imwrite(path, np.zeros((200, 300, 2), "uint8"))
    probe = probe_image(str(path))
    assert (probe.pages, probe.height, probe.width) == (1, 200, 300)
    array, pages = decode_image(str(path))
    assert array.shape == (200, 300, 2) and pages == 1


@pytest.mark.parametrize("name", ["GELPY_MAX_PIXELS", "GELPY_MAX_PAGES"])
@pytest.mark.parametrize(
    "value",
    ["lots", "0", "-3", "1_000", "+5", "1.5", "٣", "５", "1" * 19, "9" * 5000],
    ids=[
        "word", "zero", "negative", "underscore", "plus", "decimal", "arabic-indic", "fullwidth",
        "nineteen digits", "five thousand digits",
    ],
)
def test_a_malformed_limit_is_an_invalid_value(monkeypatch, name, value):
    """Only the ASCII digits 0-9 make a limit, and at most 18 of them: Python's own int() also takes
    `1_000`, `+5` and the digits of other scripts, and refuses a string of over 4300 digits with an
    error of its own that names neither the variable nor a next step."""
    monkeypatch.setenv(name, value)
    with pytest.raises(InvalidValueError) as info:
        image_limits()
    assert name in str(info.value)
    assert "18 digits" in info.value.detail


def test_a_limit_of_eighteen_digits_is_accepted(monkeypatch):
    monkeypatch.setenv("GELPY_MAX_PIXELS", "9" * 18)
    assert image_limits() == (10**18 - 1, DEFAULT_MAX_PAGES)


def test_a_limit_is_a_whole_number_in_ascii_digits(monkeypatch):
    monkeypatch.setenv("GELPY_MAX_PIXELS", " 0250 ")
    monkeypatch.setenv("GELPY_MAX_PAGES", "7")
    assert image_limits() == (250, 7)
    monkeypatch.setenv("GELPY_MAX_PIXELS", "")  # an empty variable counts as unset
    assert image_limits() == (DEFAULT_MAX_PIXELS, 7)


def test_a_file_with_only_a_tiff_header_is_a_decode_error(tmp_path):
    broken = tmp_path / "broken.tif"
    broken.write_bytes(b"II*\x00\x08\x00\x00\x00")  # a TIFF header pointing past the end
    with pytest.raises(ImageDecodeError) as info:
        decode_image(str(broken))
    assert "holds no image" in str(info.value)
    assert "IndexError" not in str(info.value)


def test_tifffile_is_silenced_only_while_a_file_is_read(tmp_path, caplog):
    """tifffile logs a warning for a damaged TIFF; gelpy's error is then the only report. The
    logger is back to normal once the read is over: nothing is disabled for the process."""
    broken = tmp_path / "broken.tif"
    broken.write_bytes(b"II*\x00\x08\x00\x00\x00")

    def tifffile_messages():
        return [record.getMessage() for record in caplog.records if record.name == "tifffile"]

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(ImageDecodeError):
            decode_image(str(broken))
        assert tifffile_messages() == []
        logging.getLogger("tifffile").warning("after the read")
    assert tifffile_messages() == ["after the read"]
    assert logging.getLogger("tifffile").filters == []


@pytest.mark.parametrize("stage", ["header", "decode"])
def test_a_readers_text_is_cut_to_its_first_line_at_either_stage(tmp_path, monkeypatch, stage):
    """The header is read first now, so a reader's multi-line text (imageio's ends in `pip
    install` hints) can arrive from the probe as well as from the decode."""
    image = tmp_path / "small.png"
    Image.fromarray(np.zeros((4, 4), "uint8")).save(image)

    def multi_line(*args, **kwargs):
        raise RuntimeError("first line\nsecond line\npip install something")

    if stage == "header":
        monkeypatch.setattr(image_io, "_probe_pillow", multi_line)
    else:
        monkeypatch.setattr(image_io.io, "imread", multi_line)
    with pytest.raises(ImageDecodeError) as info:
        decode_image(str(image))
    message = str(info.value)
    assert "small.png" in message and "RuntimeError: first line" in message
    assert "\n" not in message and "pip install" not in message


#: TIFF layouts with the pages and pixels the header must report: every plane of the stack counts,
#: and the channel axis does not, because that is what the decoder holds in memory. skimage's
#: imread moves a planar channel axis last, so (3, H, W) and (N, 3, H, W) are RGB, not stacks.
LAYOUTS = {
    "gray": ((50, 60), {}, 1, 3_000),
    "rgb interleaved": ((50, 60, 3), {"photometric": "rgb"}, 1, 3_000),
    "rgb planar": ((3, 400, 400), {"photometric": "rgb", "planarconfig": "separate"}, 1, 160_000),
    "gray pages": ((7, 50, 60), {"photometric": "minisblack"}, 7, 21_000),
    "rgb pages": ((7, 50, 60, 3), {"photometric": "rgb"}, 7, 21_000),
    "rgb planar pages": (
        (2, 3, 50, 60),
        {"photometric": "rgb", "planarconfig": "separate"},
        2,
        6_000,
    ),
    "imagej TZYX": ((3, 5, 50, 60), {"imagej": True, "metadata": {"axes": "TZYX"}}, 15, 45_000),
    "imagej ZCYX": ((4, 3, 50, 60), {"imagej": True, "metadata": {"axes": "ZCYX"}}, 4, 12_000),
    "ome ZCYX": ((2, 3, 50, 60), {"ome": True, "metadata": {"axes": "ZCYX"}}, 2, 6_000),
    "imagej TZCYX": ((2, 3, 4, 50, 60), {"imagej": True}, 6, 18_000),
    # a 'shaped' series (what a plain tifffile write is) is decoded with every axis it was written
    # with, a length-1 axis included, and tifffile 2026.5.2 and later reports its shape without that
    # axis: three pages of Y x X looked like one colour image of Y x X pixels
    "shaped 3 x 1 x Y x X": ((3, 1, 50, 60), {}, 3, 9_000),
    "shaped 4 x 1 x Y x X": ((4, 1, 50, 60), {}, 4, 12_000),
    "shaped 1 x 3 x Y x X": ((1, 3, 50, 60), {}, 1, 3_000),
    "shaped 2 x 1 x 3 x Y x X": ((2, 1, 3, 50, 60), {}, 2, 6_000),
    "shaped 1 x 1 x Y x X": ((1, 1, 50, 60), {}, 1, 3_000),
}


@pytest.mark.parametrize(
    ("shape", "options", "pages", "pixels"), LAYOUTS.values(), ids=list(LAYOUTS)
)
def test_the_header_counts_what_the_decoder_holds(tmp_path, shape, options, pages, pixels):
    path = tmp_path / "layout.tif"
    tifffile.imwrite(path, np.zeros(shape, "uint8"), **options)
    probe = probe_image(str(path))
    assert (probe.pages, probe.pixels) == (pages, pixels)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        held = skimage_io.imread(str(path))  # the array decode_image would be handed
    channels = held.shape[-1] if held.ndim >= 3 and held.shape[-1] in (2, 3, 4) else 1
    assert held.size // channels == probe.pixels


def test_a_planar_rgb_image_is_one_page_of_height_times_width_pixels(tmp_path):
    path = tmp_path / "planar.tif"
    tifffile.imwrite(
        path, np.zeros((3, 400, 400), "uint8"), photometric="rgb", planarconfig="separate"
    )
    probe = probe_image(str(path))
    assert (probe.pages, probe.pixels, probe.width, probe.height) == (1, 160_000, 400, 400)


def test_a_stack_over_the_pixel_cap_is_refused_whatever_its_axes_mean(tmp_path, monkeypatch, decoded):
    """ImageJ and OME stacks have four axes and no channel axis (T, Z, Y, X): the plane size
    is the last two axes, and every plane counts."""
    path = tmp_path / "tzyx.tif"
    tifffile.imwrite(
        path, np.zeros((3, 5, 50, 60), "uint8"), imagej=True, metadata={"axes": "TZYX"}
    )
    monkeypatch.setenv("GELPY_MAX_PIXELS", "44999")  # 3 x 5 x 50 x 60 = 45,000
    with pytest.raises(ImageTooLargeError) as info:
        decode_image(str(path))
    assert decoded == []  # the decoder never ran
    assert "45,000 pixels" in str(info.value)
    monkeypatch.setenv("GELPY_MAX_PIXELS", "45000")
    image_io.check_limits(probe_image(str(path)), path.name)  # exactly at the cap: accepted


def test_a_stack_over_the_page_cap_is_refused_whatever_its_axes_mean(tmp_path, monkeypatch, decoded):
    path = tmp_path / "tzyx.tif"
    tifffile.imwrite(
        path, np.zeros((3, 5, 50, 60), "uint8"), imagej=True, metadata={"axes": "TZYX"}
    )
    monkeypatch.setenv("GELPY_MAX_PAGES", "14")  # 3 x 5 planes
    with pytest.raises(ImageTooLargeError) as info:
        decode_image(str(path))
    assert decoded == []
    assert "15 pages" in str(info.value)
    monkeypatch.setenv("GELPY_MAX_PAGES", "15")
    image_io.check_limits(probe_image(str(path)), path.name)


def test_a_stack_of_four_pages_with_a_length_one_axis_is_refused_by_both_caps(tmp_path, monkeypatch, decoded):
    path = tmp_path / "four.tif"
    tifffile.imwrite(path, np.zeros((4, 1, 50, 60), "uint8"))  # four pages of 50 x 60
    monkeypatch.setenv("GELPY_MAX_PIXELS", "11999")  # 4 x 50 x 60 = 12,000
    with pytest.raises(ImageTooLargeError) as info:
        decode_image(str(path))
    assert decoded == [] and "12,000 pixels" in str(info.value)
    monkeypatch.setenv("GELPY_MAX_PIXELS", "12000")
    monkeypatch.setenv("GELPY_MAX_PAGES", "3")
    with pytest.raises(ImageTooLargeError) as info:
        decode_image(str(path))
    assert decoded == [] and "4 pages" in str(info.value)


def test_two_planar_rgb_pages_of_six_thousand_pixels_square_are_counted_in_full(monkeypatch):
    """Header-only: such a file is 216 million samples, which a four-axis count that dropped the
    last axis reported as 36,000 pixels."""
    probe = image_io._probe_shape((2, 3, 6000, 6000))
    assert (probe.pages, probe.pixels) == (2, 72_000_000)
    monkeypatch.setenv("GELPY_MAX_PIXELS", "50000000")
    with pytest.raises(ImageTooLargeError):
        image_io.check_limits(probe, "pages.tif")


def test_a_gif_whose_later_frames_are_larger_than_its_screen_is_refused_before_decoding(
    tmp_path, decoded
):
    """Pillow reports the logical screen and counts frames, but the decoder reads every frame at
    its own size, and a GIF may make a frame far larger than the screen."""
    path = tmp_path / "frames.gif"
    path.write_bytes(_gif_with_frames((1, 1), [(1, 1)] + [(4000, 4000)] * 3))
    with pytest.raises(ImageDecodeError) as info:
        decode_image(str(path))
    assert decoded == []  # the decoder never ran
    assert "more than one frame" in str(info.value)
    assert "PNG or TIFF" in info.value.detail


def test_any_gif_with_more_than_one_frame_is_refused(tmp_path, decoded):
    frames = [Image.fromarray(np.full((8, 8), 40 * i, "uint8")) for i in range(3)]
    path = tmp_path / "animated.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:])
    with pytest.raises(ImageDecodeError) as info:
        decode_image(str(path))
    assert decoded == [] and "more than one frame" in str(info.value)


def test_a_single_frame_gif_still_decodes(tmp_path):
    path = _single_frame_gif(tmp_path / "one.gif", (40, 30), screen=(40, 30))
    array, pages = decode_image(str(path))
    assert array.shape[:2] == (30, 40) and pages == 1  # a palette GIF arrives as RGB


@pytest.mark.parametrize(
    ("screen", "offset", "extent"),
    [((1, 1), (0, 0), (40, 30)), ((10, 10), (5, 7), (45, 37)), ((100, 80), (0, 0), (100, 80))],
    ids=["frame larger than the screen", "frame past the screen's corner", "screen larger"],
)
def test_a_single_frame_gif_is_counted_by_the_larger_of_its_screen_and_its_frame(
    tmp_path, screen, offset, extent
):
    path = _single_frame_gif(tmp_path / "one.gif", (40, 30), screen, offset)
    probe = probe_image(str(path))
    assert (probe.pages, probe.width, probe.height) == (1, *extent)
    assert probe.pixels == extent[0] * extent[1]


def test_a_single_frame_gif_over_the_cap_by_its_frame_is_refused(tmp_path, monkeypatch, decoded):
    path = _single_frame_gif(tmp_path / "one.gif", (40, 30), screen=(1, 1))
    monkeypatch.setenv("GELPY_MAX_PIXELS", "1000")  # the 1 x 1 screen is under it, the frame over
    with pytest.raises(ImageTooLargeError) as info:
        decode_image(str(path))
    assert decoded == [] and "1,200 pixels" in str(info.value)


def test_a_pixel_bomb_is_reported_with_gelpys_own_count_and_limit(tmp_path):
    """Pillow's own bomb error quotes twice its limit; gelpy's message names gelpy's."""
    bomb = _png_header_only(tmp_path / "bomb.png", 40_000, 40_000)
    with pytest.raises(ImageTooLargeError) as info:
        decode_image(str(bomb))
    assert str(info.value) == (
        "'bomb.png' would decode to 1,600,000,000 pixels; the limit is 100,000,000."
    )


def test_pillows_decompression_guard_is_out_of_the_way_during_a_read_and_restored_after(
    tmp_path, monkeypatch
):
    """gelpy's caps decide what is too large. Pillow's guard is a process-wide setting: it is
    switched off while gelpy reads a file and put back as it was, whether the read worked or not."""
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)  # Pillow alone would refuse over 200 pixels
    small = tmp_path / "small.png"
    Image.fromarray(np.zeros((20, 30), "uint8")).save(small)  # 600 pixels
    bomb = _png_header_only(tmp_path / "bomb.png", 40_000, 40_000)
    garbage = tmp_path / "garbage.png"
    garbage.write_bytes(b"not a png " * 20)
    array, _ = decode_image(str(small))
    assert array.shape == (20, 30)
    for path, error in ((bomb, ImageTooLargeError), (garbage, ImageDecodeError)):
        with pytest.raises(error):
            decode_image(str(path))
    assert Image.MAX_IMAGE_PIXELS == 100


def test_nested_reads_put_pillows_guard_back_once(monkeypatch):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)
    with image_io._pillow_guard_off():
        assert Image.MAX_IMAGE_PIXELS is None
        with image_io._pillow_guard_off():
            assert Image.MAX_IMAGE_PIXELS is None
        assert Image.MAX_IMAGE_PIXELS is None  # the inner exit must not restore it early
    assert Image.MAX_IMAGE_PIXELS == 100
