"""The cache and the precious files go through the same atomic writer as the plots."""

import sys

import numpy as np
import pytest

from gelpy.runfolder import cache
from gelpy.schemas.fit_config import FitConfigStore

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="symlinks need extra privileges")


@posix_only
def test_write_json_does_not_follow_a_link_planted_at_a_predictable_temp_name(tmp_path):
    import os

    victim = tmp_path / "victim.txt"
    victim.write_bytes(b"precious")
    (tmp_path / f".fit_config.json.tmp.{os.getpid()}").symlink_to(victim)
    cache.write_json(tmp_path / "fit_config.json", FitConfigStore())
    assert victim.read_bytes() == b"precious"
    assert (tmp_path / "fit_config.json").read_text(encoding="utf-8").startswith("{")


@posix_only
def test_the_array_caches_replace_a_link_at_their_path(tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_bytes(b"precious")
    (tmp_path / "a.npy").symlink_to(victim)
    (tmp_path / "b.npz").symlink_to(victim)
    cache.get_or_compute_npy(tmp_path / "a.npy", lambda: np.arange(3))
    cache.get_or_compute_npz(tmp_path / "b.npz", lambda: {"x": np.arange(3)})
    assert victim.read_bytes() == b"precious"
    assert np.load(tmp_path / "a.npy").tolist() == [0, 1, 2]


def test_a_failed_compute_leaves_no_file_behind(tmp_path):
    def broken():
        raise RuntimeError("no result")

    with pytest.raises(RuntimeError):
        cache.get_or_compute_json(tmp_path / "x.json", broken)
    assert list(tmp_path.iterdir()) == []
