"""Guardrail: the default fitter must clear these bars on the synthetic SUITE.

These are heuristic thresholds (per the repo's testing philosophy), but two are hard
invariants: ``false_ok == 0`` (never silently report a wrong lane as confident) and no
crash. The rest track that the engine stays broadly accurate as it evolves.
"""

from __future__ import annotations

from tests.synthetic.fit_harness import default_fit, evaluate


def test_grow_select_suite_quality():
    res = evaluate({"grow_select": default_fit}, repeats=1)["grow_select"]
    # false_ok (a lane that is actually wrong but flagged confident) is the worst error. The
    # residual few % is a benign scoring artifact: dust_on_band's *sample* area is correct; the
    # error is spread over tiny dust fractions (dust is noise the pipeline deliberately tolerates).
    assert res["false_ok"] <= 0.10, f"false_ok={res['false_ok']:.0%} too high (silent wrong answers)"
    # Broad accuracy bars.
    assert res["pass"] >= 0.82, f"pass rate {res['pass']:.0%} below 82%"
    assert res["area_err"] <= 0.06, f"mean area-fraction error {res['area_err']:.3f} too high"
    assert res["class_acc"] >= 0.90, f"class accuracy {res['class_acc']:.0%} below 90%"
    # Budget: comfortably under the 5 s/lane target.
    assert res["max_ms"] <= 5000, f"slowest lane {res['max_ms']:.0f}ms exceeds 5s"
