"""2D plane background fit via RANSAC (ports the legacy ``PlaneFit2d``).

Two horizontal, background-only stripes of the image are sampled and a 2D plane is
fit through them with RANSAC (to reject band-signal outliers). The reconstructed
plane is subtracted from the whole image and negative values are clipped to zero.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor

from ...schemas.background import PlaneRansacParams
from .base import BackgroundFitResult, BackgroundModel, background_registry


def default_stripes(height: int) -> list[tuple[int, int]]:
    """Heuristic background stripes: a thin band near the top and near the bottom."""
    return [(20, 10), (max(0, height - 15), 10)]


def _stripe_samples(image: np.ndarray, position: int, stripe_height: int):
    stripe = image[position : position + stripe_height, :]
    xs, ys = np.meshgrid(
        np.arange(stripe.shape[1]),
        np.arange(position, position + stripe.shape[0]),
    )
    return stripe.ravel(), xs.ravel(), ys.ravel()


def fit_plane(image: np.ndarray, stripes: list[tuple[int, int]]):
    """RANSAC-fit a plane ``z = a*x + b*y + c`` to the sampled stripes.

    Returns ``(params_dict, background_plane, inlier_fraction)``.
    """
    zs, xs, ys = [], [], []
    for position, stripe_height in stripes:
        z, x, y = _stripe_samples(image, position, stripe_height)
        zs.append(z)
        xs.append(x)
        ys.append(y)
    fit_data = np.concatenate(zs)
    data = np.c_[np.concatenate(xs), np.concatenate(ys)]

    ransac = RANSACRegressor(estimator=LinearRegression(fit_intercept=True), random_state=0)
    ransac.fit(data, fit_data)
    a, b = ransac.estimator_.coef_
    c = ransac.estimator_.intercept_
    inlier_fraction = float(np.mean(ransac.inlier_mask_))

    h, w = image.shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    background_plane = a * grid_x + b * grid_y + c
    params = {"a": float(a), "b": float(b), "c": float(c)}
    return params, background_plane, inlier_fraction


@background_registry.register(
    "plane_ransac",
    config_model=PlaneRansacParams,
    description="RANSAC-fit 2D plane through two background stripes, then subtract it.",
)
class PlaneRansac(BackgroundModel):
    name = "plane_ransac"

    def remove(self, image: np.ndarray, params: PlaneRansacParams) -> BackgroundFitResult:
        stripes = params.stripes or default_stripes(image.shape[0])
        plane_params, background_plane, inlier_fraction = fit_plane(image, stripes)
        corrected = image - background_plane
        corrected[corrected < 0] = 0  # clip negatives
        return BackgroundFitResult(
            corrected_image=corrected,
            background_plane=background_plane,
            params={**plane_params, "inlier_fraction": inlier_fraction},
            inlier_fraction=inlier_fraction,
        )
