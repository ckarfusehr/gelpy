import numpy as np
from skimage import io

from gelpy.runfolder.layout import RunPaths
from gelpy.runfolder.manifest import RunManifest
from gelpy.runfolder.pipeline import Pipeline
from gelpy.runfolder.recipe_io import build_image_ref
from gelpy.schemas.image import ImageMeta
from gelpy.schemas.lanes import LaneLayout
from gelpy.schemas.recipe import Recipe


def _bright_on_dark_png(tmp_path):
    x = np.arange(400)
    band = np.zeros(400)
    for c in (80, 200, 320):
        band += np.exp(-((x - c) ** 2) / (2 * 6.0**2))
    img = np.clip(0.05 + np.tile(band, (200, 1)), 0, 1)  # bright bands on dark bg
    p = tmp_path / "gel.png"
    io.imsave(str(p), (img * 255).astype("uint8"), check_contrast=False)
    return str(p)


def test_ensure_image_replays_recipe_meta_not_fresh_autodetect(tmp_path):
    """The source is bright-on-dark (auto-detect would NOT invert), but the frozen
    recipe.image forces inverted=True. If ensure_image re-detects instead of replaying
    the recipe's meta, this override would be silently dropped and bands would stay
    bright; asserting the reverse proves the frozen meta -- not fresh detection -- drove
    the cached array."""
    path = _bright_on_dark_png(tmp_path)
    ref = build_image_ref(path)
    recipe = Recipe(
        source_image=ref, lanes=LaneLayout(lanes=[], profile_width_px=10),
        image=ImageMeta(inverted=True, polarity="dark_on_light", polarity_source="manual",
                         channel_strategy="gray"),
    )
    paths = RunPaths(str(tmp_path / "run")); paths.ensure_dirs()
    img = Pipeline(paths, recipe, RunManifest(recipe_hash="x")).ensure_image()
    assert img.dtype == np.float64 and 0.0 <= img.min() and img.max() <= 1.0
    assert img[:, 80].mean() < img[:, 5].mean()  # forced inversion makes bands dark
