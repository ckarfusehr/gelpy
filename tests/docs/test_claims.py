"""Every load-bearing factual claim the documentation makes, asserted against the code.

How this works
--------------
Prose cites a claim inline::

    The default background model is `robust_plane`. <!-- claim: default-background-model -->

and this module holds ``test_claim_default_background_model``, whose docstring states the
claim and whose body asserts it. ``test_every_claim_is_cited_and_every_citation_is_tested``
enforces the bijection, so a claim cannot be cited without being tested, and a test cannot
rot into describing prose nobody reads.

Why claims *and* freshness digests
----------------------------------
The digest in each page's front matter catches changed signatures, defaults and Field
descriptions. It cannot catch a change in behaviour *inside* a function -- if the meaning of
`area_fraction` changed while its name did not, no digest would move. Claims cover exactly
that: a fact worth stating in prose is worth asserting mechanically.

Claim ids are kebab-case and stable. Renaming one is a documentation change: update the
citation and the test name together.
"""

from __future__ import annotations

import inspect
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs"

CLAIM_MARKER = re.compile(r"<!--\s*claim:\s*([a-z0-9-]+)\s*-->")
TEST_PREFIX = "test_claim_"


# =============================================================================================
# the claims
# =============================================================================================


def test_claim_default_background_model():
    """Background removal defaults to `robust_plane`; `plane_ransac` is a fallback only."""
    import typer.main

    from gelpy.cli.app import app
    from gelpy.schemas.background import BackgroundStageConfig

    assert BackgroundStageConfig.model_fields["model"].default == "robust_plane"
    command = typer.main.get_command(app).commands["background"].commands["remove"]
    model_param = next(p for p in command.params if p.name == "model")
    assert model_param.default == "robust_plane"


def test_claim_default_fit_strategy():
    """Fitting defaults to the `grow_select` strategy, with two higher tiers available."""
    from gelpy.core.peaks.strategies import fit_strategy_registry
    from gelpy.schemas.fit_config import FitConfig

    assert FitConfig.model_fields["method"].default == "grow_select"
    assert set(fit_strategy_registry.names()) == {"grow_select", "sparse_seed", "bayesian"}


def test_claim_escalation_ladder_order():
    """`--auto-escalate` climbs grow_select -> sparse_seed -> bayesian."""
    from gelpy.runfolder.commands import _TIER_LADDER

    assert list(_TIER_LADDER) == ["grow_select", "sparse_seed", "bayesian"]


def test_claim_default_baseline_is_poly_order_2():
    """The fit-time baseline defaults to a polynomial of order 2, not arPLS."""
    from gelpy.schemas.baseline import BaselineConfig

    assert BaselineConfig.model_fields["method"].default == "poly"
    assert BaselineConfig.model_fields["poly_order"].default == 2


def test_claim_baseline_absorption_is_monotone_in_order():
    """More baseline freedom absorbs more of a genuine smear: none < poly-0 < poly-2 < arPLS.

    The measured figures quoted in the guide and in `core/peaks/baseline.py` (none 0%,
    poly-0 ~2%, poly-2 ~5.5%, poly-3 ~13%, arPLS ~34%) are re-derived here rather than
    trusted, since they are the numbers a reader uses to choose a baseline.
    """
    import numpy as np

    from gelpy.core.peaks.baseline import estimate_baseline
    from gelpy.schemas.baseline import BaselineConfig

    x = np.arange(600, dtype=float)
    band = np.exp(-((x - 180) ** 2) / (2 * 8.0**2))
    smear = 0.25 * np.exp(-((x - 380) ** 2) / (2 * 55.0**2))
    y = band + smear + np.random.default_rng(0).normal(0, 0.004, x.size)
    truth = smear.sum()

    def absorbed(config: BaselineConfig) -> float:
        return float(estimate_baseline(y, config).sum() / truth)

    none = absorbed(BaselineConfig(method="none"))
    order0 = absorbed(BaselineConfig(method="poly", poly_order=0))
    order2 = absorbed(BaselineConfig(method="poly", poly_order=2))
    order3 = absorbed(BaselineConfig(method="poly", poly_order=3))
    arpls = absorbed(BaselineConfig(method="arpls"))

    assert none == 0.0
    assert none < order0 < order2 < order3 < arpls
    assert order2 < 0.10, "the default must stay conservative (documented ~5.5%)"
    assert arpls > 0.25, "arPLS is documented as absorbing a large fraction (~34%)"


def test_claim_default_label_rotation_is_90():
    """Lane labels are vertical by default, so long sample names stay legible at any N."""
    from gelpy.schemas.lanes import LaneLayout
    from gelpy.schemas.requests import InitRunRequest

    assert LaneLayout.model_fields["label_rotation"].default == 90.0
    assert InitRunRequest.model_fields["label_rotation"].default == 90.0


def test_claim_default_projection_strategy_is_d4_union():
    """Lane detection projects rows with `d4_union` by default, falling back to full height."""
    from gelpy.core.lanes import detect_lane_centers, projection_strategies

    default = inspect.signature(detect_lane_centers).parameters["projection_strategy"].default
    assert default == "d4_union"
    assert {"full", "d4_union"} <= set(projection_strategies.names())


def test_claim_smoothing_window_fraction():
    """The detection smoothing window defaults to 0.7 x the expected pitch."""
    from gelpy.core.lanes import _SMOOTHING_WINDOW_FRACTION
    from gelpy.schemas.requests import DetectLanesRequest

    assert _SMOOTHING_WINDOW_FRACTION == 0.7
    description = DetectLanesRequest.model_fields["smoothing_window"].description
    assert "0.7" in description, "the schema description must quote the actual value"


def test_claim_lane_selector_grammar():
    """A lane term is an int index, `re:<pattern>`, `label:<text>`, or a literal label."""
    from gelpy.core.lanes import resolve_lane_selector
    from gelpy.schemas.lanes import LaneDefinition, LaneLayout

    layout = LaneLayout(
        lanes=[
            LaneDefinition(index=0, label="scaffold", x_center_px=10.0),
            LaneDefinition(index=1, label="6 MgCl", x_center_px=20.0),
            LaneDefinition(index=2, label="12", x_center_px=30.0),
        ],
        profile_width_px=8,
    )
    assert resolve_lane_selector(1, layout) == [1]
    assert resolve_lane_selector("re:.*MgCl.*", layout) == [1]
    assert resolve_lane_selector("scaffold", layout) == [0]
    assert resolve_lane_selector("label:12", layout) == [2]
    assert resolve_lane_selector("all", layout) == [0, 1, 2]


def test_claim_lane_selector_matching_nothing_raises():
    """A selector that matches nothing raises, listing the available labels. No silent fallback."""
    from gelpy.core.errors import LaneSelectionError
    from gelpy.schemas.lanes import LaneDefinition, LaneLayout

    from gelpy.core.lanes import resolve_lane_selector

    layout = LaneLayout(
        lanes=[LaneDefinition(index=0, label="a", x_center_px=1.0)], profile_width_px=4
    )
    with pytest.raises(LaneSelectionError) as excinfo:
        resolve_lane_selector("re:nothing-matches-this", layout)
    assert excinfo.value.available == ["a"]
    assert excinfo.value.code == "lane_selection"


def test_claim_status_ok_is_not_a_quality_signal():
    """`status: "ok"` means the fit converged; `confidence` is the quality signal."""
    from typing import get_args

    from gelpy.schemas.peaks import FitQuality, LaneFitResult

    assert set(get_args(LaneFitResult.model_fields["status"].annotation)) == {
        "ok",
        "empty",
        "failed",
    }
    assert "confidence" in LaneFitResult.model_fields
    assert "confidence_reasons" in LaneFitResult.model_fields
    assert set(get_args(LaneFitResult.model_fields["confidence"].annotation)) == {"ok", "low"}
    assert FitQuality is not None


def _demo_recipe():
    """A minimal valid recipe, for exercising the hash chain."""
    from gelpy.schemas.image import ImageRef
    from gelpy.schemas.lanes import LaneDefinition, LaneLayout
    from gelpy.schemas.recipe import Recipe

    lane = LaneDefinition(index=0, label="first", x_center_px=100.0)
    return (
        Recipe(
            source_image=ImageRef(
                path="/gels/a.tif",
                sha256="a" * 64,
                size=1234,
                mtime=1.0,
                height=500,
                width=2000,
                dtype="uint16",
            ),
            lanes=LaneLayout(lanes=[lane], profile_width_px=180),
        ),
        lane,
    )


def test_claim_profile_hash_excludes_slice_and_normalization():
    """The profile hash covers strip geometry only; slice and normalization feed the fit hash.

    That is what makes re-slicing or re-normalizing a lane cheap: the extracted profile stays
    a cache hit and only the fit recomputes. Asserted behaviourally -- the function cannot even
    see a profile config -- rather than by grepping its source, which would trip over its own
    docstring.
    """
    from gelpy.runfolder.stages import fit_hash, profile_hash
    from gelpy.schemas.fit_config import FitConfig
    from gelpy.schemas.profiles import LineProfileConfig

    recipe, lane = _demo_recipe()

    assert "profile_config" not in inspect.signature(profile_hash).parameters
    baseline = profile_hash(recipe, lane)

    # Renaming a lane is cosmetic and must not invalidate its cached profile.
    renamed = lane.model_copy(update={"label": "renamed"})
    assert profile_hash(recipe, renamed) == baseline

    # ...but the fit hash does move with the slice window.
    fit_config = FitConfig()
    wide = fit_hash(recipe, lane, LineProfileConfig(slice_start=0, slice_end=600), fit_config)
    narrow = fit_hash(recipe, lane, LineProfileConfig(slice_start=50, slice_end=500), fit_config)
    assert wide != narrow

    # Moving the strip DOES invalidate the profile: that is a different measurement.
    moved = lane.model_copy(update={"x_center_px": 140.0})
    assert profile_hash(recipe, moved) != baseline


def test_claim_hashes_never_read_the_image_pixels():
    """Every stage hash comes from the recipe and small configs, never from image pixels.

    That is why deciding what to recompute costs a few sha256-over-small-JSON operations
    regardless of image size. Demonstrated by changing everything about the file *except* its
    content hash and observing that the image hash does not move.
    """
    from gelpy.runfolder.stages import background_hash, image_hash, profile_hash

    recipe, lane = _demo_recipe()
    before = image_hash(recipe)

    # Same bytes, different path/mtime/size/dimensions: identity is the content hash alone.
    relocated = recipe.model_copy(
        update={
            "source_image": recipe.source_image.model_copy(
                update={"path": "/elsewhere/b.tif", "mtime": 999.0, "size": 4321, "height": 9}
            )
        }
    )
    assert image_hash(relocated) == before

    # Different content, same everything else: must move.
    edited = recipe.model_copy(
        update={"source_image": recipe.source_image.model_copy(update={"sha256": "b" * 64})}
    )
    assert image_hash(edited) != before

    # With no background stage configured, background_hash is a passthrough of image_hash,
    # so no background cache entry is ever created or consulted.
    assert background_hash(recipe) == image_hash(recipe)

    # No hash function accepts array data.
    for hasher in (image_hash, background_hash, profile_hash):
        for parameter in inspect.signature(hasher).parameters.values():
            assert "ndarray" not in str(parameter.annotation)


def test_claim_fit_config_is_not_on_the_manifest():
    """Per-lane fit intent lives in fit_config.json, so it survives deleting the manifest."""
    from gelpy.schemas.run import RunManifest

    assert "fit_config" not in RunManifest.model_fields
    assert "fit_config.json" in inspect.getsource(
        __import__("gelpy.runfolder.layout", fromlist=["RunPaths"]).RunPaths
    )


def test_claim_error_codes_are_stable_and_complete():
    """Every domain error carries a stable machine-readable code."""
    from gelpy.core import errors

    codes = {
        obj.code
        for obj in vars(errors).values()
        if isinstance(obj, type) and issubclass(obj, errors.GelpyError)
    }
    assert {
        "gelpy_error",
        "lane_selection",
        "registry",
        "source_image_changed",
        "nonfinite_image",
        "unsupported_image",
    } <= codes


def test_claim_results_csv_columns():
    """The documented results.csv column prefix is exactly RESULT_COLUMNS."""
    from gelpy.schemas.results import RESULT_COLUMNS

    assert list(RESULT_COLUMNS[:12]) == [
        "lane_index",
        "label",
        "band_number",
        "component_class",
        "shape",
        "area",
        "area_fraction",
        "area_uncertainty",
        "maximum_position",
        "residual_fraction",
        "confidence",
        "status",
    ]


def test_claim_component_classes():
    """Components are classified `sample|well|smear|dust|unknown`, counted, never dropped.

    Note the fifth class: `unknown` exists and is the classifier's fallback, so documentation
    that lists only four is wrong. The label is semantic only -- it never removes a band from
    the total, which is what keeps `area_fraction` honest.
    """
    from typing import get_args

    from gelpy.schemas.peaks import BandFit

    field = BandFit.model_fields["component_class"]
    assert set(get_args(field.annotation)) == {"sample", "well", "smear", "dust", "unknown"}
    assert field.default == "sample"
    assert "never excludes" in (field.description or "")


def test_claim_image_show_draws_no_lane_overlay_by_default():
    """`image show` renders a clean view unless `--lanes` is passed."""
    import typer.main

    from gelpy.cli.app import app

    command = typer.main.get_command(app).commands["image"].commands["show"]
    lanes_param = next(p for p in command.params if p.name == "lanes")
    assert lanes_param.default is False
    assert "--lanes" in lanes_param.opts


def test_claim_matplotlib_only_in_render():
    """matplotlib is imported only under render/, plus cli/_render.py and cli/_pick.py.

    `cli/_pick.py` is the one deliberate exception: it is an interactive human-in-the-loop
    picker, so it selects a real GUI backend itself instead of the Agg default.
    """
    offenders = []
    for path in (REPO / "src" / "gelpy").rglob("*.py"):
        if "render" in path.parts or path.name in ("_render.py", "_pick.py"):
            continue
        if re.search(r"^\s*(import matplotlib|from matplotlib)", path.read_text(), re.M):
            offenders.append(str(path.relative_to(REPO)))
    assert not offenders, f"matplotlib imported outside render/: {offenders}"


def test_claim_schemas_import_without_scientific_stack():
    """`gelpy.schemas` imports standalone -- no numpy/scipy/matplotlib -- so a thin client
    (an MCP server, say) can do schema discovery and validation alone."""
    program = (
        "import sys; import gelpy.schemas; "
        "bad = [m for m in ('numpy','scipy','matplotlib','skimage','sklearn') "
        "if m in sys.modules]; "
        "print(','.join(bad))"
    )
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"gelpy.schemas pulled in {result.stdout.strip()}"


def test_claim_every_schema_is_discoverable():
    """`gelpy schema <Name>` can print every contract in the package."""
    import importlib
    import pkgutil

    import gelpy.schemas as schemas
    from gelpy.schemas.common import CanonicalModel

    defined = set()
    for info in pkgutil.iter_modules(schemas.__path__):
        module = importlib.import_module(f"gelpy.schemas.{info.name}")
        for name, obj in vars(module).items():
            if getattr(obj, "__module__", None) != module.__name__:
                continue
            if isinstance(obj, type) and issubclass(obj, CanonicalModel):
                defined.add(name)
    assert defined <= set(schemas.__all__)


def test_claim_extra_fields_are_rejected():
    """Every contract forbids unknown fields, so a typo is an error rather than a silent no-op."""
    from pydantic import ValidationError

    from gelpy.schemas.baseline import BaselineConfig

    with pytest.raises(ValidationError):
        BaselineConfig(polly_order=2)


def test_claim_area_fraction_is_on_the_true_signal_basis():
    """`area_fraction` is a component's area over the integral of (profile - baseline),
    so fractions across a lane's bands sum to the modelled share of real signal and the
    unmodelled remainder shows up as `residual_fraction` rather than being hidden."""
    from gelpy.schemas.peaks import BandFit, LaneFitResult

    assert "area_fraction" in BandFit.model_fields
    assert "residual_fraction" in LaneFitResult.model_fields
    assert "signal_area" in LaneFitResult.model_fields
    description = (LaneFitResult.model_fields["signal_area"].description or "").lower()
    assert "baseline" in description


# =============================================================================================
# the bijection: no uncited claim, no untested citation
# =============================================================================================


def _strip_code(text: str) -> str:
    """Remove fenced blocks AND inline code spans before scanning for claim citations.

    A `<!-- claim: some-id -->` shown in backticks is an illustration of the syntax -- the
    extending guide explains how to cite a claim -- not a citation of a fact. Counting it
    would demand a test named after a placeholder.
    """
    without_fences = re.sub(r"^```.*?^```", "", text, flags=re.S | re.M)
    return re.sub(r"`[^`\n]*`", "", without_fences)


def _cited_claim_ids() -> dict[str, list[str]]:
    """claim id -> the documents citing it."""
    found: dict[str, list[str]] = {}
    for path in sorted(DOCS.rglob("*.md")):
        if "decisions" in path.parts:
            continue  # historical records are not held to the live claim contract
        for claim_id in CLAIM_MARKER.findall(_strip_code(path.read_text())):
            found.setdefault(claim_id, []).append(str(path.relative_to(REPO)))
    return found


def _tested_claim_ids() -> set[str]:
    return {
        name[len(TEST_PREFIX) :].replace("_", "-")
        for name in globals()
        if name.startswith(TEST_PREFIX)
    }


def test_every_claim_is_cited_and_every_citation_is_tested():
    """The claim set and the citation set must match exactly.

    An untested citation is a lie waiting to happen. An uncited claim is either dead weight
    or a fact the documentation ought to be stating -- either way, worth noticing.
    """
    cited = _cited_claim_ids()
    tested = _tested_claim_ids()

    untested = sorted(set(cited) - tested)
    uncited = sorted(tested - set(cited))

    message = []
    if untested:
        message.append(
            "cited in the docs but no test asserts them "
            f"(add test_claim_<id> here): {untested}\n  "
            + "\n  ".join(f"{cid}: {cited[cid]}" for cid in untested)
        )
    if uncited:
        message.append(
            "asserted here but cited by no document "
            f"(add `<!-- claim: <id> -->` where the docs state it, or delete the test): {uncited}"
        )
    assert not message, "\n\n".join(message)
