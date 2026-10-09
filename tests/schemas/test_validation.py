import pytest
from pydantic import ValidationError

from gelpy.schemas import (
    ContrastConfig,
    FitConfig,
    IntensityRange,
    LineProfileConfig,
)
from gelpy.schemas.peaks import BandFit, EmgPeakParams, GaussianPeakParams, LaneFitResult


def test_extra_field_is_forbidden():
    with pytest.raises(ValidationError):
        FitConfig(bogus=1)


def test_fit_config_defaults():
    c = FitConfig()
    assert c.method == "grow_select"
    assert c.baseline.method == "poly"
    assert "gaussian" in c.shapes and "emg" in c.shapes


def test_seed_negative_rejected():
    with pytest.raises(ValidationError):
        FitConfig(seed=-1)


def test_intensity_range_bounds_checked():
    with pytest.raises(ValidationError):
        IntensityRange(low=0.9, high=0.1)
    with pytest.raises(ValidationError):
        IntensityRange(low=0.0, high=2.0, mode="percentile")
    IntensityRange(low=10, high=200, mode="absolute")


def test_line_profile_slice_validation():
    with pytest.raises(ValidationError):
        LineProfileConfig(slice_start=100, slice_end=50)
    LineProfileConfig(slice_start=50, slice_end=None)


def test_contrast_defaults():
    c = ContrastConfig()
    assert c.gamma.gamma == 0.1
    assert c.intensity_range.mode == "percentile"


def test_emg_params_signed_tau():
    e = EmgPeakParams(amplitude=1, center=2, sigma=3, tau=-4)
    assert e.tau == -4
    assert EmgPeakParams.model_validate_json(e.model_dump_json()).tau == -4


def test_lane_fit_result_to_rows_columns():
    lane = LaneFitResult(
        lane_index=12,
        label="600 NaCl",
        coverage=1.0,
        bands=[
            BandFit(
                band_number=0,
                component_class="sample",
                area=15.0,
                area_fraction=0.62,
                maximum_position=101.0,
                params=GaussianPeakParams(amplitude=1.0, center=100.0, sigma=5.0),
            )
        ],
    )
    rows = lane.to_rows()
    assert rows[0]["lane_index"] == 12
    assert set(rows[0]) >= {
        "lane_index",
        "label",
        "band_number",
        "component_class",
        "area",
        "area_fraction",
        "maximum_position",
        "shape",
        "param_amplitude",
        "param_center",
        "param_sigma",
    }


def test_failed_lane_is_preserved():
    lane = LaneFitResult(lane_index=3, label="x", status="failed", error="no peaks detected")
    assert lane.status == "failed"
    rows = lane.to_rows()
    assert len(rows) == 1
    assert rows[0]["lane_index"] == 3
    assert rows[0]["status"] == "failed"
    assert rows[0]["error"] == "no peaks detected"
    assert "band_number" not in rows[0]


def test_fit_config_store_int_key_roundtrip():
    from gelpy.schemas.fit_config import FitConfigStore, PerLaneFitConfig

    store = FitConfigStore(lanes={12: PerLaneFitConfig(fit=FitConfig(seed=3))})
    reloaded = FitConfigStore.model_validate_json(store.model_dump_json())
    assert set(reloaded.lanes) == {12}
    assert reloaded.lanes[12].fit.seed == 3


def test_fit_config_store_forbids_extra():
    from gelpy.schemas.fit_config import PerLaneFitConfig

    with pytest.raises(ValidationError):
        PerLaneFitConfig(bogus=1)


def test_lane_status_can_represent_every_fit_status():
    """`gelpy status` must be able to report any outcome a fit can actually produce.

    LaneFitResult.status allows "empty" (no significant signal detected, a legitimate
    outcome that is preserved rather than dropped). If LaneStatus cannot hold that value,
    `status` raises validation_error and exits 1 on a perfectly valid run folder.
    """
    from typing import Union, get_args, get_origin

    from gelpy.schemas.peaks import LaneFitResult
    from gelpy.schemas.run import LaneStatus

    def _literals(model, field):
        ann = model.model_fields[field].annotation
        if get_origin(ann) is Union:
            return {v for arg in get_args(ann) for v in get_args(arg)}
        return set(get_args(ann))

    fit_states = _literals(LaneFitResult, "status")
    status_states = _literals(LaneStatus, "status")
    missing = fit_states - status_states
    assert not missing, f"LaneStatus.status cannot represent {sorted(missing)}"


def test_every_model_is_discoverable_via_schema_command():
    """Every contract in gelpy.schemas must be printable by `gelpy schema <Name>`.

    `gelpy schema` looks names up in `schemas.__all__`, so a model that is defined but not
    exported is invisible to the discovery surface an agent is told to rely on — it gets a
    registry error and concludes the contract does not exist.
    """
    import importlib
    import pkgutil

    import gelpy.schemas as schemas
    from gelpy.schemas.common import CanonicalModel

    defined: set[str] = set()
    for info in pkgutil.iter_modules(schemas.__path__):
        module = importlib.import_module(f"gelpy.schemas.{info.name}")
        for name, obj in vars(module).items():
            if name.startswith("_") or getattr(obj, "__module__", None) != module.__name__:
                continue
            if isinstance(obj, type) and issubclass(obj, CanonicalModel):
                defined.add(name)

    missing = defined - set(schemas.__all__)
    assert not missing, f"defined but not discoverable via `gelpy schema`: {sorted(missing)}"
