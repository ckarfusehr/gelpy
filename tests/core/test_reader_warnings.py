"""What tifffile says while a file is read is kept: a stack it could only half read is no silent success.

tifffile reports a damaged file by logging, and then often returns what it could read: a stack cut
off in the middle comes back with its missing pages as zeros. gelpy keeps the log records of a read
that succeeded in ``ImageMeta.reader_warnings``, so that ``init``, ``image set`` and ``status`` can
say so, and still keeps them off stderr, which is one error object or nothing.
"""

import logging
import struct

import numpy as np
import pytest
import tifffile

from gelpy.core import image_io
from gelpy.core.image_io import canonicalize_image
from gelpy.runfolder import commands
from gelpy.schemas.requests import InitRunRequest, SetImageRequest, StatusRequest

TIFFFILE = logging.getLogger("tifffile")


@pytest.fixture
def small_tiff(tmp_path):
    path = tmp_path / "gel.tif"
    tifffile.imwrite(path, np.random.default_rng(1).random((50, 60)).astype("float32"))
    return path


@pytest.fixture
def noisy_read(monkeypatch):
    """A read that succeeds while tifffile logs these records (level, text)."""

    def set_records(*records):
        real = image_io.io.imread

        def imread(*args, **kwargs):
            for level, text in records:
                TIFFFILE.log(level, text)
            return real(*args, **kwargs)

        monkeypatch.setattr(image_io.io, "imread", imread)

    return set_records


def test_a_clean_read_keeps_nothing(small_tiff):
    assert canonicalize_image(str(small_tiff))[1].reader_warnings == []


def test_a_warning_tifffile_logs_during_a_successful_read_is_kept(small_tiff, noisy_read, capsys, caplog):
    noisy_read((logging.WARNING, "<tifffile.TiffPages @8> invalid page offset 12"))
    with caplog.at_level(logging.DEBUG):
        _, meta = canonicalize_image(str(small_tiff))
    assert meta.reader_warnings == ["<tifffile.TiffPages @8> invalid page offset 12"]
    assert capsys.readouterr().err == ""  # kept, not printed: stderr stays one error object or nothing
    assert [r for r in caplog.records if r.name == "tifffile"] == []


def test_an_error_record_is_kept_and_a_note_is_not(small_tiff, noisy_read):
    noisy_read((logging.INFO, "a note"), (logging.ERROR, "a failure"), (logging.DEBUG, "a detail"))
    assert canonicalize_image(str(small_tiff))[1].reader_warnings == ["a failure"]


def test_a_warning_logged_while_the_header_is_read_is_kept_too(small_tiff, monkeypatch):
    real = image_io._probe_tiff

    def noisy(file):
        TIFFFILE.warning("a problem in the page list")
        return real(file)

    monkeypatch.setattr(image_io, "_probe_tiff", noisy)
    assert canonicalize_image(str(small_tiff))[1].reader_warnings == ["a problem in the page list"]


def test_the_same_message_is_kept_once(small_tiff, noisy_read):
    noisy_read(*[(logging.WARNING, "again")] * 20)
    assert canonicalize_image(str(small_tiff))[1].reader_warnings == ["again"]


def test_at_most_five_messages_are_kept_and_the_rest_are_counted(small_tiff, noisy_read):
    noisy_read(*[(logging.WARNING, f"problem {i}") for i in range(8)])
    kept = canonicalize_image(str(small_tiff))[1].reader_warnings
    assert kept == [f"problem {i}" for i in range(5)] + ["(and 3 more)"]


def test_a_long_message_is_cut(small_tiff, noisy_read):
    noisy_read((logging.WARNING, "x" * 5000))
    (kept,) = canonicalize_image(str(small_tiff))[1].reader_warnings
    assert len(kept) < 300 and kept.startswith("xxx")


def test_the_path_of_the_file_is_not_kept(small_tiff, noisy_read):
    noisy_read((logging.WARNING, f"cannot read {small_tiff} past byte 12"))
    (kept,) = canonicalize_image(str(small_tiff))[1].reader_warnings
    assert kept == "cannot read gel.tif past byte 12"


def test_replaying_a_recipe_collects_nothing_and_leaves_the_logger_alone(small_tiff, noisy_read):
    _, meta = canonicalize_image(str(small_tiff))
    noisy_read((logging.WARNING, "again"))
    image_io.apply_image_meta(str(small_tiff), meta)  # the cached pipeline's replay: no detection, no notes
    assert TIFFFILE.filters == []


# --- what a person sees --------------------------------------------------------------------------
def test_init_says_that_tifffile_reported_a_problem(small_tiff, noisy_read, tmp_path):
    noisy_read((logging.ERROR, "<tifffile.TiffPages @8> invalid page offset 65541"))
    run = str(tmp_path / "run")
    summary = commands.init_run(InitRunRequest(run_dir=run, image_path=str(small_tiff), n_lanes=3, width=10))
    (warning,) = [w for w in summary.warnings if "tifffile" in w]
    assert "invalid page offset 65541" in warning
    assert "truncated" in warning and "source_pages" in warning  # what it may mean, and where to look
    assert summary.image["reader_warnings"] == ["<tifffile.TiffPages @8> invalid page offset 65541"]
    # status says it again: it is not a one-time remark
    assert [w for w in commands.status(StatusRequest(run_dir=run)).warnings if "tifffile" in w] == [warning]


def test_image_set_reads_the_file_again_and_says_so_too(small_tiff, noisy_read, tmp_path):
    run = str(tmp_path / "run")
    commands.init_run(InitRunRequest(run_dir=run, image_path=str(small_tiff), n_lanes=3, width=10))
    noisy_read((logging.WARNING, "a problem the second read has"))
    summary = commands.set_image(SetImageRequest(run_dir=run, invert=False))
    assert any("a problem the second read has" in w for w in summary.warnings)


def test_a_clean_file_has_no_such_warning(small_tiff, tmp_path):
    run = str(tmp_path / "run")
    summary = commands.init_run(InitRunRequest(run_dir=run, image_path=str(small_tiff), n_lanes=3, width=10))
    assert not [w for w in summary.warnings if "tifffile" in w]


def _stack_with_a_broken_page_list(path, pages=6, readable=3):
    """A stack of ``pages`` pages whose page list breaks after ``readable`` of them, the pixels intact:
    the next-page pointer of the last readable page leads past the end of the file."""
    with tifffile.TiffWriter(path) as writer:
        for i in range(pages):
            writer.write(
                np.full((40, 50), 1000 * (i + 1), "uint16"),
                photometric="minisblack",
                metadata=None,
                contiguous=False,
            )
    raw = bytearray(path.read_bytes())
    here = struct.unpack_from("<I", raw, 4)[0]
    for _ in range(readable - 1):
        entries = struct.unpack_from("<H", raw, here)[0]
        here = struct.unpack_from("<I", raw, here + 2 + 12 * entries)[0]
    entries = struct.unpack_from("<H", raw, here)[0]
    struct.pack_into("<I", raw, here + 2 + 12 * entries, len(raw) + 4096)
    path.write_bytes(bytes(raw))


def test_a_stack_whose_page_list_breaks_is_reported(tmp_path):
    """The case the warning is for: tifffile returns the pages it can still reach (three of six) and
    says so only in its log, so that nothing else tells the person that half the stack is missing."""
    path = tmp_path / "broken.tif"
    _stack_with_a_broken_page_list(path)
    _, meta = canonicalize_image(str(path))
    if not meta.reader_warnings:
        pytest.skip("this tifffile does not log anything for a broken page list")
    assert any("page" in text for text in meta.reader_warnings)
    whole = tmp_path / "whole.tif"
    tifffile.imwrite(whole, np.zeros((6, 40, 50), "uint16"), photometric="minisblack")
    assert canonicalize_image(str(whole))[1].reader_warnings == []
