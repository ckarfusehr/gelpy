from gelpy.runfolder.hashing import canonical_json, content_hash, hash_fields
from gelpy.runfolder.stages import background_hash, image_hash
from gelpy.schemas.fit_config import FitConfig
from gelpy.schemas.image import ImageMeta, ImageRef
from gelpy.schemas.lanes import LaneLayout
from gelpy.schemas.recipe import Recipe


def test_content_hash_is_deterministic():
    a = FitConfig(method="grow_select", seed=1)
    b = FitConfig(method="grow_select", seed=1)
    assert content_hash(a) == content_hash(b)


def test_content_hash_changes_with_field():
    a = FitConfig(seed=1)
    b = FitConfig(seed=0)
    assert content_hash(a) != content_hash(b)


def test_canonical_json_key_order_independent():
    # Same logical dict, different insertion order -> identical canonical form.
    assert hash_fields(a=1, b=2) == hash_fields(b=2, a=1)


def test_type_coercion_resolved_before_hash():
    # Validated models coerce "0" -> 0, so these must hash identically.
    a = FitConfig(seed="0")
    b = FitConfig(seed=0)
    assert content_hash(a) == content_hash(b)


def test_canonical_json_compact():
    payload = canonical_json({"b": 1, "a": 2})
    assert payload == '{"a":2,"b":1}'


def _recipe(**meta_kw):
    ref = ImageRef(path="x.tif", sha256="a" * 64, size=1, mtime=0.0, height=2, width=2, dtype="f8")
    return Recipe(
        source_image=ref,
        lanes=LaneLayout(lanes=[], profile_width_px=10),
        image=ImageMeta(**meta_kw),
    )


def test_image_hash_changes_when_inverted_flips():
    assert image_hash(_recipe(inverted=False)) != image_hash(_recipe(inverted=True))


def test_image_hash_stable_for_equal_meta():
    assert image_hash(_recipe(channel_strategy="mean")) == image_hash(_recipe(channel_strategy="mean"))


def test_invert_flip_propagates_to_background_hash():
    assert background_hash(_recipe(inverted=False)) != background_hash(_recipe(inverted=True))
