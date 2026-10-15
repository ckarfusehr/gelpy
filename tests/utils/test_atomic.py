"""The one atomic writer: every file gelpy puts in a run folder or a report folder goes through it."""

import os
import stat
import subprocess
import sys

import pytest

from gelpy.utils import atomic
from gelpy.utils.atomic import atomic_write

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes and symlinks")


def test_it_writes_bytes_and_makes_the_folder(tmp_path):
    atomic_write(tmp_path / "a" / "b" / "file.bin", b"abc")
    assert (tmp_path / "a" / "b" / "file.bin").read_bytes() == b"abc"


def test_it_hands_a_binary_file_to_a_writer(tmp_path):
    def write(fh):
        fh.write(b"one ")
        fh.write(b"two")

    atomic_write(tmp_path / "file.bin", write)
    assert (tmp_path / "file.bin").read_bytes() == b"one two"


def test_it_replaces_what_is_there(tmp_path):
    (tmp_path / "file.txt").write_bytes(b"old")
    atomic_write(tmp_path / "file.txt", b"new")
    assert (tmp_path / "file.txt").read_bytes() == b"new"
    assert [p.name for p in tmp_path.iterdir()] == ["file.txt"]  # no temp file is left


def test_a_failing_writer_keeps_the_old_file_and_leaves_no_temp_file(tmp_path):
    (tmp_path / "file.txt").write_bytes(b"old")

    def write(fh):
        fh.write(b"half")
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError):
        atomic_write(tmp_path / "file.txt", write)
    assert (tmp_path / "file.txt").read_bytes() == b"old"
    assert [p.name for p in tmp_path.iterdir()] == ["file.txt"]


@posix_only
def test_a_link_at_the_destination_is_replaced_and_its_target_is_untouched(tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_bytes(b"precious")
    (tmp_path / "out.png").symlink_to(victim)
    atomic_write(tmp_path / "out.png", b"png")
    assert victim.read_bytes() == b"precious"
    assert not (tmp_path / "out.png").is_symlink()
    assert (tmp_path / "out.png").read_bytes() == b"png"


@posix_only
def test_a_link_planted_at_the_name_of_a_temp_file_is_not_followed(tmp_path):
    """The temp name is not predictable: gelpy 1.0 wrote ``.name.tmp.<pid>``, which another user
    could plant as a link to a file of the victim's."""
    victim = tmp_path / "victim.txt"
    victim.write_bytes(b"precious")
    (tmp_path / f".out.png.tmp.{os.getpid()}").symlink_to(victim)
    atomic_write(tmp_path / "out.png", b"png")
    assert victim.read_bytes() == b"precious"


@posix_only
@pytest.mark.parametrize("mask, mode", [(0o000, 0o666), (0o002, 0o664), (0o022, 0o644), (0o027, 0o640), (0o077, 0o600)])
def test_the_file_gets_0o666_minus_the_umask(tmp_path, mask, mode):
    """The kernel applies the umask when the temp file is created: a process sets its own in a
    subprocess, so that this one's is never touched."""
    code = (
        "import os, sys; os.umask(int(sys.argv[2], 8)); "
        "from gelpy.utils.atomic import atomic_write; atomic_write(sys.argv[1], b'x')"
    )
    target = tmp_path / "file.txt"
    subprocess.run([sys.executable, "-c", code, str(target), f"{mask:o}"], check=True)
    assert stat.S_IMODE(target.stat().st_mode) == mode


def test_a_write_neither_changes_a_mode_nor_reads_the_umask(tmp_path, monkeypatch):
    """A file system may refuse a mode change, and the umask can only be read by setting it, which
    changes it for every other thread of the process: a write needs neither."""

    def refuse(*args, **kwargs):
        raise PermissionError(1, "Operation not permitted")

    for name in ("chmod", "fchmod", "umask"):
        if hasattr(os, name):
            monkeypatch.setattr(os, name, refuse)
    atomic_write(tmp_path / "file.txt", b"x")
    atomic_write(tmp_path / "file.bin", lambda handle: handle.write(b"y"))
    assert (tmp_path / "file.txt").read_bytes() == b"x"
    assert (tmp_path / "file.bin").read_bytes() == b"y"


def test_the_temp_file_is_named_after_the_file_with_a_random_part(tmp_path):
    seen = []

    def write(handle):
        seen.extend(p.name for p in tmp_path.iterdir() if p.name != "file.bin")
        handle.write(b"x")

    atomic_write(tmp_path / "file.bin", write)
    atomic_write(tmp_path / "file.bin", write)
    first, second = seen
    for name in seen:
        assert atomic.is_temp_name(name), name
        assert name.startswith(".file.bin.") and name.endswith(".tmp")
    assert first != second  # not a name another process could predict
    assert [p.name for p in tmp_path.iterdir()] == ["file.bin"]


@pytest.mark.parametrize(
    "name, debris",
    [
        (".index.html.0123456789ab.tmp", True),
        (".lane_000.png.ffffffffffff.tmp", True),
        ("..gelpy-report.json.0123456789ab.tmp", True),
        ("index.html.0123456789ab.tmp", False),  # not a dotfile
        (".index.html.0123456789AB.tmp", False),  # the random part is lower-case hexadecimal
        (".index.html.0123456789a.tmp", False),  # and twelve characters long
        (".index.html.0123456789abc.tmp", False),
        (".index.html.tmp", False),
        (".index.html.0123456789ab", False),
        (".notes.tmp", False),
        (".tmp", False),
        ("", False),
    ],
)
def test_what_is_a_temp_name(name, debris):
    assert atomic.is_temp_name(name) is debris


def test_a_name_that_is_taken_is_not_used_another_is_tried(tmp_path, monkeypatch):
    names = iter(["aaaaaaaaaaaa", "aaaaaaaaaaaa", "bbbbbbbbbbbb"])
    monkeypatch.setattr(atomic.secrets, "token_hex", lambda n=None: next(names))
    taken = tmp_path / ".file.bin.aaaaaaaaaaaa.tmp"
    taken.write_bytes(b"someone else's")
    atomic_write(tmp_path / "file.bin", b"x")
    assert (tmp_path / "file.bin").read_bytes() == b"x"
    assert taken.read_bytes() == b"someone else's"  # created exclusively: never written over


def test_when_every_name_is_taken_it_fails_and_changes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(atomic.secrets, "token_hex", lambda n=None: "aaaaaaaaaaaa")
    taken = tmp_path / ".file.bin.aaaaaaaaaaaa.tmp"
    taken.write_bytes(b"someone else's")
    (tmp_path / "file.bin").write_bytes(b"old")
    with pytest.raises(FileExistsError):
        atomic_write(tmp_path / "file.bin", b"new")
    assert (tmp_path / "file.bin").read_bytes() == b"old"
    assert sorted(p.name for p in tmp_path.iterdir()) == [".file.bin.aaaaaaaaaaaa.tmp", "file.bin"]
