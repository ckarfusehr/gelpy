import pytest
from pydantic import ValidationError

from gelpy.schemas.image import ImageMeta
from gelpy.schemas.recipe import Recipe
from gelpy.schemas.run import RunStatus, RunUpdateSummary


def test_image_meta_defaults_are_legacy_safe():
    m = ImageMeta()
    assert m.polarity == "bright_on_dark"
    assert m.inverted is False
    assert m.channel_strategy == "mean"
    assert m.page == 0


def test_image_meta_forbids_unknown_field():
    with pytest.raises(ValidationError):
        ImageMeta(polarty="bright_on_dark")  # typo -> extra=forbid


def test_recipe_has_default_image_meta():
    # A recipe dict predating this field still validates (default_factory supplies it).
    data = {
        "source_image": {
            "path": "x.tif", "sha256": "0" * 64, "size": 1, "mtime": 0.0,
            "height": 2, "width": 2, "dtype": "float64",
        },
        "lanes": {"lanes": [], "profile_width_px": 10, "label_rotation": 45.0},
    }
    r = Recipe.model_validate(data)
    assert r.image.polarity == "bright_on_dark"


def test_summary_and_status_carry_warnings():
    s = RunUpdateSummary(warnings=["w"], image={"polarity": "dark_on_light"})
    assert s.warnings == ["w"] and s.image["polarity"] == "dark_on_light"
    st = RunStatus(
        run_dir="r", source_image="x.tif", image_width=2, image_height=2,
        polarity="dark_on_light", polarity_confidence="low", warnings=["w"],
    )
    assert st.polarity_confidence == "low"
