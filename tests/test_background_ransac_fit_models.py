
import pytest
import numpy as np
from gelpy.background_ransac_fit_models import PlaneFit2d

def test_get_stripe_data():
    image = np.random.rand(100, 100)
    model = PlaneFit2d(image, ((10, 10), (80, 10)))
    stripe, X, Y = model.get_stripe_data(10, 10)
    assert stripe.shape == (10, 100)
    assert X.shape == (10, 100)
    assert Y.shape == (10, 100)

def test_extract_fit_data_from_image():
    image = np.random.rand(100, 100)
    model = PlaneFit2d(image, ((10, 10), (80, 10)))
    model.extract_fit_data_from_image()
    assert model.fit_data.shape == (2000,)
    assert model.fit_data_X.shape == (2000,)
    assert model.fit_data_Y.shape == (2000,)

def test_compute_new_image():
    image = np.random.rand(100, 100)
    model = PlaneFit2d(image, ((10, 10), (80, 10)))
    X, Y = np.meshgrid(np.arange(100), np.arange(100))
    params = [0.1, 0.2, 0.3]
    new_image, bg_plane = model.compute_new_image(image, params, X, Y)
    assert new_image.shape == image.shape
    assert bg_plane.shape == image.shape

def test_compute_overlay():
    image = np.random.rand(100, 100)
    model = PlaneFit2d(image, ((10, 10), (80, 10)))
    overlay = model.compute_overlay()
    assert overlay.shape == (100, 100, 4)


