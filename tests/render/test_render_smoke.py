"""Smoke tests: each renderer produces a non-trivial file. No pixel diffing."""

import numpy as np
import pytest

from gelpy import render
from gelpy.schemas.contrast import ContrastConfig
from gelpy.schemas.image import ImageRef
from gelpy.schemas.lanes import LaneDefinition, LaneLayout
from gelpy.schemas.peaks import BandFit, GaussianPeakParams, LaneFitResult
from gelpy.schemas.recipe import Recipe
from gelpy.schemas.results import AnalysisResult


def _nontrivial(path):
    return path.exists() and path.stat().st_size > 1000


def _band():
    return BandFit(
        band_number=0,
        component_class="sample",
        area=20.0,
        area_fraction=1.0,
        maximum_position=100.0,
        params=GaussianPeakParams(amplitude=1.0, center=100.0, sigma=8.0),
    )


@pytest.fixture
def layout():
    lanes = [LaneDefinition(index=i, label=f"L{i}", x_center_px=40 + i * 60) for i in range(3)]
    return LaneLayout(lanes=lanes, profile_width_px=30)


def test_render_gel_image(tmp_path, layout):
    image = np.random.rand(120, 200)
    out = render.render_gel_image(
        image, tmp_path / "img.png", contrast=ContrastConfig(), layout=layout, title="test"
    )
    assert _nontrivial(out)


def test_render_gel_image_accepts_signed_residuals(tmp_path, layout):
    """A background-corrected image keeps signed residuals; rendering must not raise.

    ``background remove`` deliberately does not rectify (clipping negatives biases every
    downstream area), so the corrected image legitimately contains negative pixels. Display
    contrast is the only thing that cannot cope with them, so the clamp belongs here.
    """
    image = np.random.rand(120, 200) - 0.3
    assert image.min() < 0
    out = render.render_gel_image(
        image, tmp_path / "corrected.png", contrast=ContrastConfig(), layout=layout, mode="non_linear"
    )
    assert _nontrivial(out)


def test_render_gel_image_raw_mode_applies_no_contrast(tmp_path):
    """``mode="raw"`` must show the canonical array as gelpy sees it.

    Rendering the same array as 'raw' and as 'non_linear' has to produce different pixels;
    if they agree, 'raw' is silently gamma-adjusted and the kind name is a lie.
    """
    image = np.linspace(0.0, 0.4, 120 * 200).reshape(120, 200)
    raw = render.render_gel_image(image, tmp_path / "raw.png", contrast=ContrastConfig(), mode="raw")
    gamma = render.render_gel_image(
        image, tmp_path / "gamma.png", contrast=ContrastConfig(), mode="non_linear"
    )
    assert _nontrivial(raw) and _nontrivial(gamma)
    assert raw.read_bytes() != gamma.read_bytes()


def test_render_profiles_overview(tmp_path):
    profiles = [("a", np.random.rand(100)), ("b", np.random.rand(100))]
    out = render.render_profiles_overview(profiles, tmp_path / "profiles.png")
    assert _nontrivial(out)


def test_render_lane_fit(tmp_path):
    x = np.arange(200, dtype=float)
    profile = 0.2 + np.exp(-((x - 100) ** 2) / (2 * 8**2))
    baseline = np.full_like(profile, 0.2)
    result = LaneFitResult(
        lane_index=0, label="L0", coverage=1.0, signal_area=20.0, bands=[_band()]
    )
    out = render.render_lane_fit(profile, baseline, result, tmp_path / "fit.png")
    assert _nontrivial(out)


def test_render_lane_fit_failed(tmp_path):
    profile = np.zeros(100)
    result = LaneFitResult(lane_index=1, label="x", status="failed", error="no peaks detected")
    out = render.render_lane_fit(profile, profile * 0.0, result, tmp_path / "fit_failed.png")
    assert _nontrivial(out)


def test_render_background_diagnostic(tmp_path):
    gx, gy = np.meshgrid(np.arange(200), np.arange(120))
    image = 0.01 * gx + 0.02 * gy + 1.0
    plane = 0.01 * gx + 0.02 * gy + 1.0
    out = render.render_background_diagnostic(
        image, plane, [(20, 10), (90, 10)], tmp_path / "bg.png"
    )
    assert _nontrivial(out)


def test_render_report(tmp_path):
    ref = ImageRef(
        path="x.tif", sha256="abc", size=10, mtime=1.0, height=120, width=200, dtype="float64"
    )
    layout = LaneLayout(
        lanes=[LaneDefinition(index=0, label="L0", x_center_px=40.0)], profile_width_px=30
    )
    recipe = Recipe(source_image=ref, lanes=layout)
    result = AnalysisResult(
        lanes=[LaneFitResult(lane_index=0, label="L0", coverage=1.0, bands=[_band()])]
    )
    out = render.render_report(recipe, result, tmp_path / "report.md")
    text = out.read_text()
    assert "gelpy analysis report" in text
    assert "lane_index" in text


def test_the_report_names_the_image_by_file_name_and_hash_not_by_path(tmp_path):
    """A path recorded relative to the run folder means nothing to a reader of a copied report."""
    ref = ImageRef(
        path="../data/gel.tif", sha256="a" * 64, size=10, mtime=1.0, height=120, width=200,
        dtype="float64",
    )
    layout = LaneLayout(
        lanes=[LaneDefinition(index=0, label="Größe", x_center_px=40.0)], profile_width_px=30
    )
    result = AnalysisResult(
        lanes=[LaneFitResult(lane_index=0, label="Größe", coverage=1.0, bands=[_band()])]
    )
    out = render.render_report(Recipe(source_image=ref, lanes=layout), result, tmp_path / "r.md")
    text = out.read_text(encoding="utf-8")
    assert f"**Source image:** `gel.tif` (sha256 `{'a' * 64}`)" in text
    assert "../data" not in text
    assert "Größe" in text
