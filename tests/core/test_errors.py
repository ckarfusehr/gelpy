"""The exception -> ErrorResponse mapping (shared by the CLI and a future MCP server).

Each domain error must surface a stable machine-readable ``code`` so an agent can
branch on the failure kind without string-matching the message.
"""

from pydantic import ValidationError

from gelpy.core.errors import (
    LaneSelectionError,
    NonFiniteImageError,
    RegistryError,
    SourceImageChangedError,
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
