import numpy as np

from gelpy.core.background import background_registry
from gelpy.core.background.plane_ransac import default_stripes, fit_plane
from gelpy.schemas.background import PlaneRansacParams


def test_registry_has_plane_ransac():
    assert "plane_ransac" in background_registry.names()
    assert background_registry.config_model("plane_ransac") is PlaneRansacParams


def test_default_stripes():
    stripes = default_stripes(100)
    assert stripes[0] == (20, 10)
    assert stripes[1] == (85, 10)


def test_fit_plane_recovers_known_plane():
    # z = 0.5*x + 0.2*y + 3  -> RANSAC should recover coefficients closely
    h, w = 100, 120
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    image = 0.5 * gx + 0.2 * gy + 3.0
    params, plane, inliers = fit_plane(image, default_stripes(h))
    assert np.isclose(params["a"], 0.5, atol=1e-6)
    assert np.isclose(params["b"], 0.2, atol=1e-6)
    assert np.isclose(params["c"], 3.0, atol=1e-4)
    assert plane.shape == image.shape
    assert 0.0 <= inliers <= 1.0


def test_remove_subtracts_and_clips():
    h, w = 100, 120
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    image = 0.5 * gx + 0.2 * gy + 3.0
    model = background_registry.get("plane_ransac")()
    result = model.remove(image, PlaneRansacParams())
    # background is (approximately) the whole image -> corrected near zero, non-negative
    assert result.corrected_image.min() >= 0
    assert np.median(result.corrected_image) < 1.0
    assert result.inlier_fraction is not None
