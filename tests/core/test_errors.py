"""The exception -> ErrorResponse mapping (shared by the CLI and the MCP server).

Each domain error must surface a stable machine-readable ``code`` so an agent can
branch on the failure kind without string-matching the message.
"""

import errno
import json
import os
from typing import Literal, Union

import numpy as np
import pytest
from pydantic import BaseModel, ValidationError, create_model

from gelpy.core.errors import (
    MAX_ECHOED_INPUT,
    NON_DOMAIN_CODES,
    ImageDecodeError,
    InvalidValueError,
    LaneSelectionError,
    NonFiniteImageError,
    RegistryError,
    SourceImageChangedError,
    SourceImageMissingError,
    UnsupportedImageError,
    to_error_response,
)
from gelpy.schemas.fit_config import FitConfig


def test_lane_selection_maps_to_code_and_available():
    r = to_error_response(LaneSelectionError("no match", available=["a", "b"]))
    assert r.code == "lane_selection"
    assert r.available == ["a", "b"]


def test_registry_maps_to_code():
    assert to_error_response(RegistryError("bad", available=["fit"])).code == "registry"


def test_source_image_changed_maps_to_code():
    # Previously fell through to a generic message; now distinctly identifiable.
    assert to_error_response(SourceImageChangedError("drift")).code == "source_image_changed"


def test_file_not_found_maps_to_code():
    assert to_error_response(FileNotFoundError("nope")).code == "not_found"


def test_validation_error_maps_to_code_with_detail():
    r = None
    try:
        FitConfig(bogus=1)
    except ValidationError as exc:
        r = to_error_response(exc)
    assert r is not None
    assert r.code == "validation_error"
    assert r.detail  # carries the pydantic error list as a JSON string


def test_unknown_exception_is_internal_error():
    assert to_error_response(RuntimeError("boom")).code == "internal_error"


def test_nonfinite_and_unsupported_map_to_codes():
    assert to_error_response(NonFiniteImageError("bad")).code == "nonfinite_image"
    assert to_error_response(UnsupportedImageError("bad")).code == "unsupported_image"


def _domain_error_classes():
    from gelpy.core import errors

    return [
        obj
        for obj in vars(errors).values()
        if isinstance(obj, type) and issubclass(obj, errors.GelpyError)
    ]


@pytest.mark.parametrize("cls", _domain_error_classes(), ids=lambda c: c.__name__)
def test_every_domain_error_has_its_own_code_and_next_step(cls):
    assert "code" in vars(cls) and "next_step" in vars(cls)
    response = to_error_response(cls("something failed"))
    assert (response.code, response.detail) == (cls.code, cls.next_step)
    assert response.detail.strip()


def test_codes_are_unique_across_domain_and_non_domain_errors():
    codes = [cls.code for cls in _domain_error_classes()]
    assert len(codes) == len(set(codes))
    assert not set(codes) & set(NON_DOMAIN_CODES)


def test_a_missing_file_that_knows_what_to_do_about_it_says_so():
    missing = SourceImageMissingError("the image is missing", detail="Put it back at /data/gel.tif.")
    response = to_error_response(missing)
    assert (response.code, response.error, response.detail) == (
        "not_found",
        "the image is missing",
        "Put it back at /data/gel.tif.",
    )
    assert isinstance(missing, FileNotFoundError)  # Python callers that catch it keep working


def test_a_specific_detail_replaces_the_default_next_step():
    response = to_error_response(InvalidValueError("2 labels for 20 lanes", detail="Pass 20 labels."))
    assert (response.code, response.detail) == ("invalid_value", "Pass 20 labels.")


def test_a_bare_value_error_is_invalid_value_not_internal_error():
    response = to_error_response(ValueError("n_lanes required"))
    assert (response.code, response.error) == ("invalid_value", "n_lanes required")
    assert response.detail == InvalidValueError.next_step


def test_invalid_value_error_is_still_a_value_error():
    """Python callers that catch ValueError keep working."""
    assert issubclass(InvalidValueError, ValueError)


@pytest.mark.parametrize(
    ("exc", "code"),
    [
        (FileNotFoundError("no such file"), "not_found"),
        (PermissionError("denied"), "os_error"),
        (MemoryError(), "out_of_memory"),
        (RuntimeError("boom"), "internal_error"),
    ],
)
def test_non_domain_failures_have_codes_and_next_steps(exc, code):
    response = to_error_response(exc)
    assert response.code == code
    assert response.detail == NON_DOMAIN_CODES[code][1]


def test_internal_error_names_the_exception_type():
    assert to_error_response(KeyError("x")).error == "KeyError: 'x'"


def test_validation_error_names_the_field_and_keeps_the_field_list():
    with pytest.raises(ValidationError) as info:
        FitConfig(seed=-1)
    response = to_error_response(info.value)
    assert response.code == "validation_error"
    assert "seed" in response.error and "gelpy schema FitConfig" in response.error
    assert json.loads(response.detail)[0]["loc"] == ["seed"]


def test_a_validation_error_without_a_location_does_not_repeat_the_title():
    """Invalid JSON has an empty `loc`; the message used to read "Recipe: Recipe: Invalid JSON"."""
    from gelpy.schemas.recipe import Recipe

    with pytest.raises(ValidationError) as info:
        Recipe.model_validate_json('{"source_image": ')
    response = to_error_response(info.value)
    assert response.code == "validation_error"
    assert response.error.startswith("Recipe: Invalid JSON")
    assert "Recipe: Recipe" not in response.error
    assert response.error.endswith("`gelpy schema Recipe` prints the contract.")
    assert json.loads(response.detail)[0]["loc"] == []


@pytest.mark.parametrize(
    "exc",
    [
        UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"),
        np.linalg.LinAlgError("Singular matrix"),
    ],
    ids=["a file that is not utf-8", "LinAlgError"],
)
def test_the_generic_invalid_value_detail_does_not_claim_the_message_names_a_value(exc):
    """These messages name no value, so the detail must not tell the caller to correct one."""
    response = to_error_response(exc)
    assert response.code == "invalid_value"
    assert response.detail == (
        "Check the value or file the message names; if it names none, this is a bug in gelpy: "
        "report it."
    )


# --- a validation_error names the allowed values of a Literal field ---------------------------------
def _validation_error(model, **fields) -> ValidationError:
    with pytest.raises(ValidationError) as info:
        model(**fields)
    return info.value


def _literal_model(*values):
    return create_model("Choice", value=(Literal[values], ...))


def test_a_literal_field_lists_its_values_as_available():
    from gelpy.schemas.requests import RenderImageRequest

    response = to_error_response(_validation_error(RenderImageRequest, run_dir="r", kind="sepia"))
    assert response.code == "validation_error"
    assert response.available == ["raw", "linear", "non_linear", "corrected"]
    assert "kind" in response.error


def test_a_nested_literal_field_lists_its_values():
    from gelpy.schemas.requests import FitRunRequest

    error = _validation_error(FitRunRequest, run_dir="r", profile={"normalization": "bogus"})
    response = to_error_response(error)
    assert response.available == ["area", "min_max", "safe"]
    assert "profile.normalization" in response.error


@pytest.mark.parametrize(
    "values",
    [
        ("x", "y", "z"),
        ("only",),
        ("a, b", "this or that", "it's", 'say "hi"'),  # a comma, " or ", both kinds of quote
        ("back\\slash", "tab\tnew\nline", "naïve"),  # escapes and non-ASCII characters
        ("'quoted'", '"double"'),
    ],
    ids=["three", "one", "commas-or-and-quotes", "escapes", "quote-characters"],
)
def test_the_allowed_values_of_a_literal_are_read_back_exactly(values):
    response = to_error_response(_validation_error(_literal_model(*values), value="nope"))
    assert response.available == list(values)


def test_non_string_literal_values_are_listed_as_json_would_write_them():
    response = to_error_response(_validation_error(_literal_model(1, 2, 3), value=9))
    assert response.available == ["1", "2", "3"]
    response = to_error_response(_validation_error(_literal_model(True, None, "s"), value="nope"))
    assert response.available == ["true", "null", "s"]


@pytest.mark.parametrize(
    "expected",
    ["<Color.RED: 1> or <Color.BLUE: 2>", "b'x' or b'y'", "'unterminated", "", "or"],
    ids=["enum-members", "bytes", "unterminated", "empty", "only-a-separator"],
)
def test_an_unreadable_list_of_values_is_no_available_rather_than_a_wrong_one(expected):
    error = ValidationError.from_exception_data(
        "Choice",
        [{"type": "literal_error", "loc": ("value",), "input": "q", "ctx": {"expected": expected}}],
    )
    response = to_error_response(error)
    assert response.code == "validation_error"
    assert response.available is None


def test_one_branch_of_a_union_is_not_the_list_of_valid_values():
    """`lanes` takes the literal 'all', an index, a label or a list of them. The error of the
    'all' branch alone would say that 'all' is the only choice."""
    from gelpy.schemas.requests import FitRunRequest

    error = _validation_error(FitRunRequest, run_dir="r", lanes={"a": 1})
    first = error.errors()[0]
    assert first["type"] == "literal_error" and "literal[" in str(first["loc"][-1])
    assert to_error_response(error).available is None


def test_a_literal_without_a_location_lists_its_values():
    """Validating a bare Literal reports an empty location; there is no last part to inspect."""
    from pydantic import TypeAdapter

    with pytest.raises(ValidationError) as info:
        TypeAdapter(Literal["a", "b"]).validate_python("c")
    assert info.value.errors()[0]["loc"] == ()
    assert to_error_response(info.value).available == ["a", "b"]


def test_available_belongs_to_the_field_the_message_names():
    """`error` names the first failing field, so `available` is that field's list or none."""
    from gelpy.schemas.requests import RenderImageRequest

    response = to_error_response(_validation_error(RenderImageRequest, kind="sepia"))  # no run_dir
    assert "run_dir" in response.error
    assert response.available is None
    assert "sepia" in response.detail  # the field list still carries the second error


# --- ... and echoes no more than MAX_ECHOED_INPUT characters of any input ----------------------------
def _echoed(**fields) -> list:
    """The field errors a StatusRequest with these (partly unknown) fields makes."""
    from gelpy.schemas.requests import StatusRequest

    return json.loads(to_error_response(_validation_error(StatusRequest, **fields)).detail)


def test_a_long_input_is_cut_and_says_so():
    big = "x" * 5000
    errors = _echoed(run_dir="r", run=big, other=big)
    assert [error["loc"] for error in errors] == [["run"], ["other"]]  # the list is intact
    assert {error["type"] for error in errors} == {"extra_forbidden"}
    for error in errors:
        assert error["input"] == "x" * MAX_ECHOED_INPUT + "... [cut: 5000 characters in all]"


def test_an_input_at_the_limit_is_echoed_whole():
    at = "x" * MAX_ECHOED_INPUT
    errors = _echoed(run_dir="r", run=at, other=at + "y")
    assert errors[0]["input"] == at
    assert errors[1]["input"].endswith(f"[cut: {MAX_ECHOED_INPUT + 1} characters in all]")


def test_a_short_structured_input_stays_structured():
    assert _echoed(run_dir="r", run={"a": [1, 2]})[0]["input"] == {"a": [1, 2]}


def test_a_long_structured_input_is_cut_as_json_text():
    echoed = _echoed(run_dir="r", run={"a": ["v" * 500]})[0]["input"]
    assert echoed.startswith('{"a": ["vvv')
    assert echoed.endswith("characters in all]") and len(echoed) < MAX_ECHOED_INPUT + 40


def test_a_whole_truncated_file_is_not_echoed():
    from gelpy.schemas.recipe import Recipe

    text = '{"source_image": ' + '"' + "y" * 100_000
    with pytest.raises(ValidationError) as info:
        Recipe.model_validate_json(text)
    response = to_error_response(info.value)
    assert response.code == "validation_error"
    assert len(response.detail) < 1000
    assert json.loads(response.detail)[0]["type"] == "json_invalid"


def test_an_integer_of_thousands_of_digits_is_still_an_error_object():
    """json cannot write an integer of more than 4300 digits (Python 3.11+), which used to make
    the error mapping itself fail."""
    errors = _echoed(run_dir=10**6000)
    assert errors[0]["type"] == "string_type"
    assert len(json.dumps(errors)) < 1000


# --- one field named by several branches: only an agreement is a list ------------------------------
class _TagA(BaseModel):
    kind: Literal["a"]


class _TagB(BaseModel):
    kind: Literal["b"]


class _Item(BaseModel):
    item: Union[_TagA, _TagB]


def test_a_union_of_models_with_different_literal_values_lists_none_of_them():
    """Its branches are models, so no location names a branch of a literal: two errors name `kind`,
    each with the values of its own branch. Either list alone would say the other is invalid."""
    error = _validation_error(_Item, item={"kind": "c"})
    first, second = error.errors()
    assert (first["type"], first["loc"][-1]) == ("literal_error", "kind")
    assert first["ctx"]["expected"] != second["ctx"]["expected"]
    assert to_error_response(error).available is None


def test_a_union_of_models_that_agree_on_the_values_still_lists_them():
    one = create_model("One", kind=(Literal["x", "y"], ...), first=(int, 1))
    two = create_model("Two", kind=(Literal["x", "y"], ...), second=(int, 2))
    holder = create_model("Holder", item=(Union[one, two], ...))
    assert to_error_response(_validation_error(holder, item={"kind": "z"})).available == ["x", "y"]


# --- every echo of what the caller sent is bounded, not only an error's input -----------------------
class _Boom:
    """An object that cannot be turned into text."""

    def __str__(self):
        raise RuntimeError("str boom")

    __repr__ = __str__


def test_a_one_megabyte_unknown_key_is_cut_in_the_message_and_in_the_field_list():
    from gelpy.schemas.requests import StatusRequest

    key = "k" * 1_000_000
    response = to_error_response(_validation_error(StatusRequest, run_dir="r", **{key: 1}))
    assert key[:MAX_ECHOED_INPUT] in response.error and key not in response.error
    assert len(response.error) < 500 and len(response.detail) < 1000
    [field] = json.loads(response.detail)  # the one error, as pydantic reported it
    assert (field["type"], field["input"]) == ("extra_forbidden", 1)
    assert field["loc"] == ["k" * MAX_ECHOED_INPUT + "... [cut: 1000000 characters in all]"]


@pytest.mark.parametrize("data", [{"run_dir": _Boom()}, {_Boom(): 1, "run_dir": "r"}], ids=["value", "key"])
def test_an_input_that_cannot_be_turned_into_text_is_still_an_error_object(data):
    from gelpy.schemas.requests import StatusRequest

    with pytest.raises(ValidationError) as info:
        StatusRequest.model_validate(data)
    response = to_error_response(info.value)
    assert response.code == "validation_error"
    assert json.loads(response.detail)  # still the field list, as JSON


def test_a_value_quoted_in_a_message_is_cut_like_every_other_echo():
    from gelpy.core.errors import shown

    assert shown("run") == "'run'" and shown(5) == "5"  # as repr wrote it
    assert shown("x" * 1000) == repr("x" * 1000)[:MAX_ECHOED_INPUT] + "... [cut: 1002 characters in all]"
    assert shown(_Boom()) == "<_Boom>"  # not a traceback


@pytest.mark.parametrize(
    ("number", "code"), [(errno.ENAMETOOLONG, "os_error"), (errno.ENOENT, "not_found")]
)
def test_a_long_file_name_in_an_operating_system_error_is_cut(number, code):
    name = "d" * 5000
    response = to_error_response(OSError(number, os.strerror(number), name))  # ENOENT makes a FileNotFoundError
    assert response.code == code
    assert name not in response.error and len(response.error) < 400
    assert response.error.startswith(f"[Errno {number}] {os.strerror(number)}: 'dddd")


def test_a_lane_selector_list_over_the_limit_names_the_field_and_the_limit_in_the_message():
    from pydantic import ValidationError

    from gelpy.schemas.requests import FitRunRequest

    with pytest.raises(ValidationError) as raised:
        FitRunRequest.model_validate({"run_dir": "run", "lanes": list(range(1001))})
    error = to_error_response(raised.value).error
    assert "lanes" in error and "1000" in error
