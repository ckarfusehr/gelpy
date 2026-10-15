"""The lane count is capped at 1000 wherever a request or a stored config carries it.

A request is untyped JSON from a person or an agent, and a recipe can be edited by hand: a count of
ten million would otherwise run for minutes and fill the disk. The cap is a field constraint, so the
failure is an ordinary ``validation_error`` that names the field and the limit, and the published
tool schemas carry it.
"""

import json

import pytest
from pydantic import ValidationError

from gelpy.core.errors import to_error_response
from gelpy.schemas import (
    DetectLanesRequest,
    ExtractProfilesRequest,
    FitConfigStore,
    FitRunRequest,
    InitRunRequest,
    LaneLayout,
    Recipe,
    SetLanesRequest,
)

CAP = 1000
BASE = {"run_dir": "run"}
INIT = {**BASE, "image_path": "gel.tif"}


def _lane(i):
    return {"index": i, "label": str(i), "x_center_px": float(i)}


#: (model, the fields of a request, how many it holds in the field under test, the field's name)
CASES = [
    pytest.param(InitRunRequest, INIT, lambda n: {"n_lanes": n}, "n_lanes", id="init n_lanes"),
    pytest.param(InitRunRequest, INIT, lambda n: {"labels": ["x"] * n}, "labels", id="init labels"),
    pytest.param(SetLanesRequest, BASE, lambda n: {"n_lanes": n}, "n_lanes", id="lanes_set n_lanes"),
    pytest.param(
        SetLanesRequest, BASE, lambda n: {"linspace": [0.0, 10.0, n]}, "linspace", id="lanes_set linspace"
    ),
    pytest.param(
        SetLanesRequest, BASE, lambda n: {"x_centers": [1.0] * n}, "x_centers", id="lanes_set x_centers"
    ),
    pytest.param(SetLanesRequest, BASE, lambda n: {"labels": ["x"] * n}, "labels", id="lanes_set labels"),
    pytest.param(DetectLanesRequest, BASE, lambda n: {"n_lanes": n}, "n_lanes", id="lanes_detect n_lanes"),
    pytest.param(DetectLanesRequest, BASE, lambda n: {"labels": ["x"] * n}, "labels", id="lanes_detect labels"),
    pytest.param(FitRunRequest, BASE, lambda n: {"lanes": list(range(n))}, "lanes", id="fit_run lanes"),
    pytest.param(
        ExtractProfilesRequest, BASE, lambda n: {"lanes": ["a"] * n}, "lanes", id="profiles_extract lanes"
    ),
]


@pytest.mark.parametrize(("model", "base", "field", "name"), CASES)
def test_a_count_of_1000_passes_validation(model, base, field, name):
    model.model_validate({**base, **field(CAP)})


@pytest.mark.parametrize(("model", "base", "field", "name"), CASES)
def test_a_count_of_1001_is_a_validation_error_that_names_the_field_and_the_limit(model, base, field, name):
    with pytest.raises(ValidationError) as refused:
        model.model_validate({**base, **field(CAP + 1)})

    response = to_error_response(refused.value)
    assert response.code == "validation_error"
    assert name in response.error  # the field is named ...
    errors = json.loads(response.detail)
    limits = [e for e in errors if e["type"] in ("less_than_equal", "too_long")]
    assert limits, errors
    assert limits[0]["loc"][0] == name
    assert limits[0]["ctx"].get("le", limits[0]["ctx"].get("max_length")) == CAP  # ... and so is the limit


def test_a_count_far_over_the_cap_is_refused_as_quickly_as_one_over_it():
    """A count of ten million never reaches the code that would loop over it."""
    with pytest.raises(ValidationError):
        InitRunRequest(run_dir="run", image_path="gel.tif", n_lanes=10_000_000)
    with pytest.raises(ValidationError):
        SetLanesRequest(run_dir="run", linspace=(0.0, 10.0, 10_000_000))


def test_the_tool_schemas_carry_the_cap():
    """An agent reads the limit from the schema before it sends anything."""
    init = InitRunRequest.model_json_schema()["properties"]
    assert init["n_lanes"]["maximum"] == CAP
    assert init["labels"]["anyOf"][0]["maxItems"] == CAP
    lanes = SetLanesRequest.model_json_schema()["properties"]
    assert lanes["x_centers"]["anyOf"][0]["maxItems"] == CAP
    assert lanes["linspace"]["anyOf"][0]["prefixItems"][2]["maximum"] == CAP


def test_the_cap_has_one_home():
    from gelpy.schemas.lanes import MAX_LANES

    assert MAX_LANES == CAP


# --- stored records -------------------------------------------------------------------------------
def test_a_layout_of_1001_lanes_does_not_load_and_one_of_1000_does():
    LaneLayout.model_validate({"lanes": [_lane(i) for i in range(CAP)], "profile_width_px": 3})
    with pytest.raises(ValidationError) as refused:
        LaneLayout.model_validate({"lanes": [_lane(i) for i in range(CAP + 1)], "profile_width_px": 3})
    response = to_error_response(refused.value)
    assert response.code == "validation_error" and "lanes" in response.error
    assert "at most 1000 items" in response.detail


def test_a_recipe_of_1001_lanes_fails_to_load_with_a_structured_error():
    """A hand-edited recipe.json: it is refused at once, never worked through."""
    source = {"path": "gel.tif", "sha256": "0" * 64, "size": 1, "mtime": 1.0, "height": 10, "width": 10, "dtype": "float32"}
    text = json.dumps(
        {"source_image": source, "lanes": {"lanes": [_lane(i) for i in range(CAP + 1)], "profile_width_px": 3}}
    )
    with pytest.raises(ValidationError) as refused:
        Recipe.model_validate_json(text)
    assert to_error_response(refused.value).code == "validation_error"


def test_a_fit_config_store_of_1001_lanes_fails_to_load_and_one_of_1000_loads():
    FitConfigStore.model_validate({"lanes": {str(i): {} for i in range(CAP)}})
    with pytest.raises(ValidationError) as refused:
        FitConfigStore.model_validate({"lanes": {str(i): {} for i in range(CAP + 1)}})
    response = to_error_response(refused.value)
    assert response.code == "validation_error" and "lanes" in response.error
