import numpy as np
from skimage import io

from gelpy.runfolder.commands import init_run, set_image, status
from gelpy.runfolder.layout import RunPaths
from gelpy.runfolder.recipe_io import load_recipe
from gelpy.schemas.requests import InitRunRequest, SetImageRequest, StatusRequest


def _png(tmp_path, arr):
    p = tmp_path / "gel.png"
    io.imsave(str(p), arr, check_contrast=False)
    return str(p)


def _bands(width=400):
    x = np.arange(width); b = np.zeros(width)
    for c in (80, 200, 320):
        b += np.exp(-((x - c) ** 2) / (2 * 6.0**2))
    return np.tile(b, (200, 1))


def test_init_detects_and_records_polarity(tmp_path):
    dark = np.clip(0.9 - _bands(), 0, 1)
    run = str(tmp_path / "run")
    init_run(InitRunRequest(run_dir=run, image_path=_png(tmp_path, (dark * 255).astype("uint8")), n_lanes=3))
    recipe = load_recipe(RunPaths(run))
    assert recipe.image.polarity == "dark_on_light" and recipe.image.inverted is True


def test_init_manual_invert_and_summary_image_block(tmp_path):
    bright = np.clip(_bands(), 0, 1)
    run = str(tmp_path / "run2")
    summary = init_run(InitRunRequest(
        run_dir=run, image_path=_png(tmp_path, (bright * 255).astype("uint8")), n_lanes=3, invert=True))
    assert summary.image["inverted"] is True
    assert load_recipe(RunPaths(run)).image.polarity_source == "manual"


def test_init_low_confidence_emits_warning(tmp_path):
    rng = np.random.default_rng(0)
    flat = np.clip(0.5 + 0.05 * rng.standard_normal((200, 400)), 0, 1)
    run = str(tmp_path / "run3")
    summary = init_run(InitRunRequest(run_dir=run, image_path=_png(tmp_path, (flat * 255).astype("uint8")), n_lanes=3))
    assert any("low-confidence" in w or "polarity" in w for w in summary.warnings)


def test_set_image_forces_inversion_and_status_reflects_it(tmp_path):
    bright = np.clip(_bands(), 0, 1)
    run = str(tmp_path / "run4")
    init_run(InitRunRequest(run_dir=run, image_path=_png(tmp_path, (bright * 255).astype("uint8")), n_lanes=3))
    set_image(SetImageRequest(run_dir=run, invert=True))
    recipe = load_recipe(RunPaths(run))
    assert recipe.image.inverted is True and recipe.image.polarity_source == "manual"
    st = status(StatusRequest(run_dir=run))
    assert st.polarity == "dark_on_light"
