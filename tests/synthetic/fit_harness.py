"""Score peak-fit strategies against the synthetic SUITE (ground-truth areas).

Primary metric is **area-fraction error**: the total-variation distance between the true and
fitted area-fraction distributions (true components matched to fitted bands by position;
unmatched mass on either side counts as error). Also: N-recovery, class accuracy, and
confidence-flag correctness (``false_ok`` = said ok but wrong = worst; ``false_low`` = said
low but was fine = merely wasteful), plus timing. Mirrors the lane-detection harness ethos.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from statistics import median

from gelpy.schemas.fit_config import FitConfig

from . import fit_conditions
from .profiles import GeneratedProfile, ProfileSpec, generate

_MATCH_TOL_FRAC = 0.03  # match a fit band to a true component within this fraction of length
_PASS_THRESH = 0.10  # area-fraction error below this = a "pass"


def default_fit(profile):
    """The default strategy (grow_select) as a plain ``profile -> LaneFitResult`` callable."""
    from gelpy.core.peaks.strategies import fit_strategy_registry

    return fit_strategy_registry.get("grow_select")(profile, FitConfig(), lane_index=0, label="x")


def _match(truth_fracs, bands, tol):
    used: set[int] = set()
    pairs = []
    for center, frac, cls in sorted(truth_fracs, key=lambda z: -z[1]):
        best, best_d = None, tol
        for j, b in enumerate(bands):
            if j in used:
                continue
            d = abs(b.maximum_position - center)
            if d < best_d:
                best, best_d = j, d
        if best is not None:
            used.add(best)
            pairs.append(((center, frac, cls), bands[best]))
        else:
            pairs.append(((center, frac, cls), None))
    extra = [b for j, b in enumerate(bands) if j not in used]
    return pairs, extra


@dataclass
class ProfileScore:
    name: str
    area_frac_error: float
    n_true: int
    n_fit: int
    class_acc: float
    coverage: float
    confidence: str
    passed: bool
    false_ok: bool
    false_low: bool
    time_ms: float


def score(gen: GeneratedProfile, result, tol: float | None = None) -> ProfileScore:
    length = gen.spec.length
    tol = tol if tol is not None else _MATCH_TOL_FRAC * length
    truth = gen.true_fractions()
    bands = result.bands if result.status == "ok" else []
    pairs, extra = _match(truth, bands, tol)

    err = 0.0
    correct = 0
    matched = 0
    for (center, frac, cls), band in pairs:
        if band is None:
            err += frac
        else:
            err += abs(frac - band.area_fraction)
            matched += 1
            if band.component_class == cls:
                correct += 1
    err += sum(b.area_fraction for b in extra)
    class_acc = correct / matched if matched else 0.0
    passed = err < _PASS_THRESH
    conf = result.confidence
    return ProfileScore(
        name=gen.spec.name,
        area_frac_error=err,
        n_true=len(truth),
        n_fit=len(bands),
        class_acc=class_acc,
        coverage=result.coverage or 0.0,
        confidence=conf,
        passed=passed,
        false_ok=(conf == "ok" and not passed),
        false_low=(conf == "low" and passed),
        time_ms=0.0,
    )


def evaluate(methods: dict, specs=None, repeats: int = 3):
    """Run each method over the SUITE with ``repeats`` seeds; return per-method aggregates."""
    specs = list(specs if specs is not None else fit_conditions.SUITE)
    out: dict[str, dict] = {}
    for mname, fn in methods.items():
        rows: list[ProfileScore] = []
        for spec in specs:
            for r in range(repeats):
                gen = generate(ProfileSpec(**{**spec.__dict__, "seed": spec.seed + r}))
                t0 = time.perf_counter()
                result = fn(gen.profile)
                dt = (time.perf_counter() - t0) * 1000.0
                s = score(gen, result)
                s.time_ms = dt
                rows.append(s)
        n = len(rows)
        out[mname] = {
            "rows": rows,
            "area_err": sum(r.area_frac_error for r in rows) / n,
            "pass": sum(r.passed for r in rows) / n,
            "false_ok": sum(r.false_ok for r in rows) / n,
            "false_low": sum(r.false_low for r in rows) / n,
            "class_acc": sum(r.class_acc for r in rows) / n,
            "med_ms": median(r.time_ms for r in rows),
            "max_ms": max(r.time_ms for r in rows),
        }
    return out


def format_table(results: dict) -> str:
    lines = [
        f"{'method':16s} {'areaErr':>8s} {'pass':>6s} {'falseOK':>8s} {'falseLow':>9s} "
        f"{'classAcc':>9s} {'medMs':>7s} {'maxMs':>7s}"
    ]
    for mname, agg in results.items():
        lines.append(
            f"{mname:16s} {agg['area_err']:8.3f} {agg['pass']:6.0%} {agg['false_ok']:8.0%} "
            f"{agg['false_low']:9.0%} {agg['class_acc']:9.0%} {agg['med_ms']:7.0f} {agg['max_ms']:7.0f}"
        )
    return "\n".join(lines)


def per_condition_table(method_name: str, results: dict) -> str:
    """Per-condition breakdown for one method (aggregated across repeats by name)."""
    rows = results[method_name]["rows"]
    by_name: dict[str, list[ProfileScore]] = {}
    for r in rows:
        by_name.setdefault(r.name, []).append(r)
    lines = [f"{'condition':20s} {'areaErr':>8s} {'nTrue':>6s} {'nFit':>5s} {'clsAcc':>7s} {'conf':>5s}"]
    for name, rs in by_name.items():
        ae = sum(r.area_frac_error for r in rs) / len(rs)
        nt = rs[0].n_true
        nf = sum(r.n_fit for r in rs) / len(rs)
        ca = sum(r.class_acc for r in rs) / len(rs)
        conf = "low" if any(r.confidence == "low" for r in rs) else "ok"
        flag = "  <-" if ae >= _PASS_THRESH else ""
        lines.append(f"{name:20s} {ae:8.3f} {nt:6d} {nf:5.1f} {ca:7.0%} {conf:>5s}{flag}")
    return "\n".join(lines)
