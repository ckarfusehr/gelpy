import numpy as np
import pytest
from skimage import io


def _gaussian(x, amp, mean, sd):
    return amp * np.exp(-((x - mean) ** 2) / (2 * sd**2))


@pytest.fixture
def gel_image_path(tmp_path):
    """A small synthetic gel: 4 lanes, 2 bands each, on a tilted background plane."""
    h, w = 250, 300
    image = np.zeros((h, w), dtype=np.float64)
    y = np.arange(h)
    centers = [40, 110, 180, 250]
    band_profile = _gaussian(y, 1.0, 80, 6) + _gaussian(y, 0.7, 160, 6)
    for cx in centers:
        for xoff in range(-15, 16):
            x = cx + xoff
            if 0 <= x < w:
                image[:, x] += band_profile
    # tilted background plane
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    image += 0.02 * gx + 0.01 * gy + 5.0
    image = image / image.max()

    path = tmp_path / "synthetic_gel.tif"
    io.imsave(str(path), image.astype(np.float32))
    return str(path), centers, w, h


@pytest.fixture
def run_dir(tmp_path):
    return str(tmp_path / "run")
