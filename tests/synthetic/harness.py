"""Evaluation harness: score detection functions over the synthetic suite.

Ranks branches on BOTH axes the baseline showed matter (see ``conditions.py``):

  - **Accuracy**: max per-lane center error vs. ground truth, in units of pitch. A run
    *passes* when that error <= ``TOL`` (a quarter pitch — beyond it the lane strip would
    straddle the wrong lane).
  - **Confidence correctness**: the ``confidence`` flag is *correct* when ``"ok"`` coincides
    with a passing run and ``"low"`` with a failing one. Two error kinds are tracked
    separately because they cost differently:
      * ``false_ok``  — says "ok" but the run failed: a **silent wrong answer** (worst).
      * ``false_low`` — accurate but flagged "low": **needless manual fallback** (wasteful).

A detection function has signature ``fn(image, n_lanes) -> DetectResult`` where DetectResult
exposes ``.centers`` (list/array) and ``.confidence`` ("ok"/"low"). ``detect_lane_centers``
satisfies this directly; design branches wrap their variant to match.

``evaluate({name: fn})`` returns structured per-condition + aggregate results and can print a
table. It draws ``n_seeds`` deterministic seed-variants per condition so numbers are stable and
reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Protocol

import numpy as np

from gelpy.core.lanes import detect_lane_centers

from .conditions import SUITE
from .generator import GelSpec, generate

TOL = 0.25          # pass threshold: max center error in units of pitch
FALSE_OK_PENALTY = 1.0   # weight of a silent-wrong-answer in the composite score
N_SEEDS = 8


class DetectResult(Protocol):
    centers: object
    confidence: str


DetectFn = Callable[[np.ndarray, int], DetectResult]


def baseline_detect(image: np.ndarray, n_lanes: int) -> DetectResult:
    """The full-height detector, as the reference every design must beat or match.

    ``projection_strategy`` is pinned explicitly and must stay pinned. Letting it inherit the
    module default silently turns this reference into *whatever ships*, so every
    design-vs-baseline comparison compares a strategy against itself and cannot fail. That is
    exactly what happened when ``d4_union`` became the default: both
    ``test_d4_harness_composite_at_least_matches_baseline`` and
    ``test_baseline_never_silently_wrong_and_has_headroom`` became vacuous.
    """
    return detect_lane_centers(image, n_lanes, projection_strategy="full")


@dataclass
class RunResult:
    err_xpitch: float
    passed: bool
    conf_ok: bool

    @property
    def flag_correct(self) -> bool:
        return self.conf_ok == self.passed

    @property
    def false_ok(self) -> bool:
        return self.conf_ok and not self.passed

    @property
    def false_low(self) -> bool:
        return (not self.conf_ok) and self.passed


@dataclass
class ConditionScore:
    name: str
    median_err: float
    pass_rate: float
    flag_correct_rate: float
    false_ok_rate: float
    false_low_rate: float
    composite: float
    n: int


def _seed_variants(spec: GelSpec, n_seeds: int) -> list[GelSpec]:
    return [replace(spec, seed=spec.seed + 1000 * s) for s in range(n_seeds)]


def _score_run(spec: GelSpec, fn: DetectFn) -> RunResult:
    gel = generate(spec)
    res = fn(gel.image, spec.n_lanes)
    centers = np.sort(np.asarray(res.centers, dtype=np.float64))
    err = float(np.abs(centers - np.sort(gel.centers)).max() / gel.pitch)
    passed = err <= TOL
    return RunResult(err_xpitch=err, passed=passed, conf_ok=(res.confidence == "ok"))


def _composite(pass_rate: float, flag_correct_rate: float, false_ok_rate: float) -> float:
    """Transparent scalar: reward accuracy + a correct flag, punish silent wrong answers."""
    return 0.5 * pass_rate + 0.5 * flag_correct_rate - FALSE_OK_PENALTY * false_ok_rate


def score_condition(spec: GelSpec, fn: DetectFn, n_seeds: int = N_SEEDS) -> ConditionScore:
    runs = [_score_run(s, fn) for s in _seed_variants(spec, n_seeds)]
    n = len(runs)
    pass_rate = np.mean([r.passed for r in runs])
    flag = np.mean([r.flag_correct for r in runs])
    fok = np.mean([r.false_ok for r in runs])
    flow = np.mean([r.false_low for r in runs])
    return ConditionScore(
        name=spec.name,
        median_err=float(np.median([r.err_xpitch for r in runs])),
        pass_rate=float(pass_rate),
        flag_correct_rate=float(flag),
        false_ok_rate=float(fok),
        false_low_rate=float(flow),
        composite=float(_composite(pass_rate, flag, fok)),
        n=n,
    )


def evaluate(
    detect_fns: dict[str, DetectFn],
    specs: tuple[GelSpec, ...] = SUITE,
    n_seeds: int = N_SEEDS,
) -> dict[str, list[ConditionScore]]:
    """Score each named detection function over ``specs``. Returns {fn_name: [ConditionScore]}."""
    return {
        name: [score_condition(spec, fn, n_seeds) for spec in specs]
        for name, fn in detect_fns.items()
    }


def aggregate(scores: list[ConditionScore]) -> ConditionScore:
    """Suite-level roll-up (means across conditions), reported under the name ``ALL``."""
    return ConditionScore(
        name="ALL",
        median_err=float(np.median([s.median_err for s in scores])),
        pass_rate=float(np.mean([s.pass_rate for s in scores])),
        flag_correct_rate=float(np.mean([s.flag_correct_rate for s in scores])),
        false_ok_rate=float(np.mean([s.false_ok_rate for s in scores])),
        false_low_rate=float(np.mean([s.false_low_rate for s in scores])),
        composite=float(np.mean([s.composite for s in scores])),
        n=sum(s.n for s in scores),
    )


def format_table(results: dict[str, list[ConditionScore]]) -> str:
    """Render a per-condition table per function, plus the aggregate row."""
    lines = []
    hdr = (
        f"{'condition':24s} {'medErr':>7s} {'pass':>6s} {'flagOK':>7s} "
        f"{'falseOK':>8s} {'falseLo':>8s} {'score':>7s}"
    )
    for name, scores in results.items():
        lines.append(f"\n=== {name} (n_seeds per condition) ===")
        lines.append(hdr)
        for s in scores:
            lines.append(
                f"{s.name:24s} {s.median_err:6.2f}p {s.pass_rate:6.0%} "
                f"{s.flag_correct_rate:7.0%} {s.false_ok_rate:8.0%} "
                f"{s.false_low_rate:8.0%} {s.composite:7.2f}"
            )
        agg = aggregate(scores)
        lines.append("-" * len(hdr))
        lines.append(
            f"{agg.name:24s} {agg.median_err:6.2f}p {agg.pass_rate:6.0%} "
            f"{agg.flag_correct_rate:7.0%} {agg.false_ok_rate:8.0%} "
            f"{agg.false_low_rate:8.0%} {agg.composite:7.2f}"
        )
    return "\n".join(lines)


def shipped_detect(image: np.ndarray, n_lanes: int) -> DetectResult:
    """Whatever gelpy currently ships, i.e. all defaults taken."""
    return detect_lane_centers(image, n_lanes)


if __name__ == "__main__":
    # Both rows, always: "full" is the fixed reference a new design must beat, "shipped" is
    # what a user actually gets. Printing only one of them is how the previous snapshot in
    # README.md came to describe a detector that had stopped being the default.
    res = evaluate({"full_reference": baseline_detect, "shipped_default": shipped_detect})
    print(format_table(res))
