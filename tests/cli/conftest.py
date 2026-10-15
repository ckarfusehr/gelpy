import numpy as np
import pytest
from skimage import io


@pytest.fixture(autouse=True)
def _plain_cli_output(monkeypatch):
    """Every CLI test sees the same plain, 100-column output, whatever environment runs it.

    rich and typer switch colour on under GitHub Actions (`GITHUB_ACTIONS`, `FORCE_COLOR`) and
    wrap at the terminal width, which splits `--n-lanes` with escape codes. In-process runners
    and `subprocess` calls both read os.environ, so one autouse fixture covers them all.
    """
    for name in ("FORCE_COLOR", "CLICOLOR_FORCE", "GITHUB_ACTIONS", "TTY_COMPATIBLE", "TTY_INTERACTIVE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("TERM", "dumb")
    monkeypatch.setenv("COLUMNS", "100")
    monkeypatch.setenv("LINES", "50")


def _gaussian(x, amp, mean, sd):
    return amp * np.exp(-((x - mean) ** 2) / (2 * sd**2))


@pytest.fixture
def gel_image(tmp_path):
    h, w = 250, 300
    image = np.zeros((h, w), dtype=np.float64)
    y = np.arange(h)
    band = _gaussian(y, 1.0, 80, 6) + _gaussian(y, 0.7, 160, 6)
    for cx in (40, 110, 180, 250):
        for x in range(cx - 15, cx + 16):
            if 0 <= x < w:
                image[:, x] += band
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    image += 0.02 * gx + 0.01 * gy + 5.0
    image = image / image.max()
    path = tmp_path / "gel.tif"
    io.imsave(str(path), image.astype(np.float32))
    return str(path)


@pytest.fixture
def run_dir(tmp_path):
    return str(tmp_path / "run")
