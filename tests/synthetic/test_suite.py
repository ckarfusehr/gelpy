"""Guardrails for the synthetic suite + harness (heuristic, not ground truth).

Keeps the generator deterministic, the ground truth well-formed, and the baseline behaving as
the harness assumes — so a design branch is measured against a stable reference. Fast: a few
seeds only.
"""

from __future__ import annotations

import numpy as np
import pytest

from .conditions import SUITE, by_name
from .generator import generate, with_seed
from .harness import aggregate, baseline_detect, evaluate


def test_generator_is_deterministic():
    spec = by_name("narrowband_mid")
    a = generate(spec).image
    b = generate(spec).image
    assert np.array_equal(a, b)


def test_seed_changes_the_image_not_the_geometry():
    spec = by_name("narrowband_mid")
    a = generate(spec)
    b = generate(with_seed(spec, spec.seed + 1))
    assert not np.array_equal(a.image, b.image)  # noise differs
    assert np.allclose(a.centers, b.centers)  # ground-truth geometry is seed-independent


@pytest.mark.parametrize("spec", SUITE, ids=[s.name for s in SUITE])
def test_ground_truth_is_uniform_pitch(spec):
    gel = generate(spec)
    assert len(gel.centers) == spec.n_lanes
    if spec.n_lanes > 1:
        gaps = np.diff(gel.centers)
        assert np.allclose(gaps, gaps[0]), "lane centers must lie on a uniform grid"
        assert np.isclose(gaps[0], gel.pitch)
    assert np.isfinite(gel.image).all()


def test_baseline_trivial_floor_is_confident_and_accurate():
    from .harness import score_condition

    s = score_condition(by_name("control_clean_thin"), baseline_detect, n_seeds=3)
    assert s.pass_rate == 1.0 and s.flag_correct_rate == 1.0


def test_baseline_flags_partial_fill_low():
    # The gel filling only part of the crop is the designed low-confidence case: it must fail
    # AND correctly flag low (never a silent wrong answer).
    from .harness import score_condition

    s = score_condition(by_name("partial_fill"), baseline_detect, n_seeds=3)
    assert s.pass_rate < 0.5
    assert s.false_ok_rate == 0.0


def test_baseline_never_silently_wrong_and_has_headroom():
    res = evaluate({"baseline": baseline_detect}, n_seeds=3)
    agg = aggregate(res["baseline"])
    assert agg.false_ok_rate == 0.0, "baseline must never say 'ok' on a failed run"
    assert agg.pass_rate > 0.8, "suite should be mostly solvable"
    assert agg.false_low_rate > 0.05, "suite must leave confidence headroom for the designs"
