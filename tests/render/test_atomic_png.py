"""Every file a renderer writes lands atomically, and every label is drawn as plain text.

``savefig(path)`` writes through a link at ``path`` and leaves a truncated file when it fails;
a label such as ``$\\frac$`` is parsed as mathtext and fails the whole figure.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.cbook import is_math_text

from gelpy import render
from gelpy.render._figure import plain_text
from gelpy.schemas.contrast import ContrastConfig
from gelpy.schemas.image import ImageRef
from gelpy.schemas.lanes import LaneDefinition, LaneLayout
from gelpy.schemas.peaks import BandFit, GaussianPeakParams, LaneFitResult
from gelpy.schemas.recipe import Recipe
from gelpy.schemas.results import AnalysisResult

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="symlinks need extra privileges")
PNG = b"\x89PNG\r\n\x1a\n"
MATH = r"$\frac$"


def _layout(label="L0"):
    lanes = [LaneDefinition(index=0, label=label, x_center_px=40.0)]
    return LaneLayout(lanes=lanes, profile_width_px=30)


def _result(label="L0"):
    band = BandFit(
        band_number=0, component_class="sample", area=20.0, area_fraction=1.0,
        maximum_position=100.0, params=GaussianPeakParams(amplitude=1.0, center=100.0, sigma=8.0),
    )
    return LaneFitResult(lane_index=0, label=label, coverage=1.0, signal_area=20.0, bands=[band])


def _gel(out):
    return render.render_gel_image(
        np.random.rand(120, 200), out, contrast=ContrastConfig(), layout=_layout(MATH)
    )


def _detection(out):
    return render.render_lane_detection(
        np.random.rand(120, 200), np.random.rand(200), np.array([40]), [40.0], out
    )


def _profiles(out):
    return render.render_profiles_overview([(MATH, np.random.rand(100))], out)


def _fit(out):
    x = np.arange(200, dtype=float)
    profile = 0.2 + np.exp(-((x - 100) ** 2) / (2 * 8**2))
    return render.render_lane_fit(profile, np.full_like(profile, 0.2), _result(MATH), out)


def _diagnostic(out):
    gx, gy = np.meshgrid(np.arange(200), np.arange(120))
    plane = 0.01 * gx + 0.02 * gy + 1.0
    return render.render_background_diagnostic(plane, plane, [(20, 10), (90, 10)], out)


def _surface(out):
    image = np.random.rand(120, 200)
    return render.render_background_surface(image, image * 0.9, image, image - 0.9 * image, out)


RENDERERS = [
    pytest.param(_gel, id="gel image"),
    pytest.param(_detection, id="lane detection"),
    pytest.param(_profiles, id="profiles overview"),
    pytest.param(_fit, id="lane fit"),
    pytest.param(_diagnostic, id="background diagnostic"),
    pytest.param(_surface, id="background surface"),
]
#: The ones that draw a label.
LABELLED = [RENDERERS[0], RENDERERS[2], RENDERERS[3]]


@pytest.fixture(autouse=True)
def _no_open_figures():
    plt.close("all")
    yield
    plt.close("all")


@posix_only
@pytest.mark.parametrize("draw", RENDERERS)
def test_a_link_at_the_destination_is_replaced_never_written_through(draw, tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_bytes(b"precious")
    out = tmp_path / "plot.png"
    out.symlink_to(victim)
    draw(out)
    assert victim.read_bytes() == b"precious"
    assert not out.is_symlink() and out.read_bytes().startswith(PNG)


@pytest.mark.parametrize("draw", RENDERERS)
def test_a_failed_save_keeps_the_old_plot_and_closes_the_figure(draw, tmp_path, monkeypatch):
    out = tmp_path / "plot.png"
    out.write_bytes(b"old plot")

    def interrupted(self, fname, *args, **kwargs):
        sink = fname if hasattr(fname, "write") else open(fname, "wb")
        sink.write(b"half a plot")
        raise RuntimeError("interrupted while saving")

    monkeypatch.setattr(plt.Figure, "savefig", interrupted)
    with pytest.raises(RuntimeError):
        draw(out)
    assert out.read_bytes() == b"old plot"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["plot.png"]
    assert plt.get_fignums() == []


@pytest.mark.parametrize("draw", LABELLED)
def test_a_label_that_looks_like_mathtext_is_drawn_as_text(draw, tmp_path):
    assert draw(tmp_path / "plot.png").read_bytes().startswith(PNG)


@pytest.mark.parametrize(
    "label", ["plain", "$", "$$", "$5", "a$b$c", r"$\frac$", r"\$", r"a\$b", r"\\$x$", "$a$ $b$", ""]
)
def test_plain_text_is_never_mathtext(label):
    drawn = plain_text(label)
    assert not is_math_text(drawn)
    assert drawn.replace(r"\$", "$") == label.replace("$", "$")  # matplotlib shows `\$` as `$`


def test_plain_text_leaves_a_label_without_dollars_alone():
    assert plain_text("lane 1 (µg)") == "lane 1 (µg)"


@posix_only
def test_report_md_replaces_a_link_too(tmp_path):
    ref = ImageRef(path="x.tif", sha256="a" * 64, size=1, mtime=1.0, height=120, width=200, dtype="f8")
    recipe = Recipe(source_image=ref, lanes=_layout())
    victim = tmp_path / "victim.txt"
    victim.write_bytes(b"precious")
    out = tmp_path / "report.md"
    out.symlink_to(victim)
    render.render_report(recipe, AnalysisResult(lanes=[_result()]), out)
    assert victim.read_bytes() == b"precious"
    assert not out.is_symlink() and "gelpy analysis report" in out.read_text(encoding="utf-8")
