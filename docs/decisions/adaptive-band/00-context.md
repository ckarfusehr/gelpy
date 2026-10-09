# Adaptive y-band lane detection — shared context for design branches D1–D4

> **HISTORICAL CONTEXT -- SUPERSEDED IN PART.** Written when full-height projection was the
> default; `d4_union` is the default, so the baseline numbers below are
> the full-height REFERENCE, not what ships. The scaffold described as living on a feature
> branch was merged to `main` -- branch off `main`.
> Current measured accuracy: `docs/guide/lane-placement.md`.


**Read this once before implementing any DX.** It gives the problem, the exact code seam, the
shared "Phase-0 scaffold" every design reuses, how to score with the harness, and the acceptance
criteria. Each `DX.md` then specifies only that design's projection strategy. Keep changes inside
`core/` (+ small `schemas`/`render`/`cli` touches); do not touch unrelated code.

## Problem
`detect_lane_centers` collapses the 2-D image to a 1-D column projection by averaging **every**
row (`core/lanes.column_projection`, full height by default). A real band is wide-in-x, thin-in-y
at a migration depth, so full-height averaging dilutes it with empty rows and loses SNR. Goal:
adaptively emphasize the informative rows. **Must never be worse than full height** (fallback).

## What the synthetic+real foundation already established (don't re-derive)
- The uniform-grid + `width/N` prior is strong: center **accuracy** rarely breaks even at low
  SNR. The measurable win is **reducing false-low confidence** (baseline false-low 18%,
  concentrated in `snr_015`, `snr_020`, `narrowband_hard`, `gradient_bg`, `empty_edge_and_interior`)
  and **geometry/illumination** cases. So target the confidence flag, not just center error.
- **Cy3 caveat (critical):** on real gels different lanes carry bands at different y (Cy3: lanes
  0–1 ladder/scaffold high, lanes 2–10 samples low). **No single y-band covers all N lanes.** A
  greedy "brightest window" that drops peak coverage for some lanes must be caught by the
  fallback. The comb-SNR scorer below is computed over **all N grid nodes**, so a band that
  abandons lanes scores *lower* — that is what protects Cy3. Verify this explicitly.

## The code seam (`src/gelpy/core/lanes.py`)
Today `detect_lane_centers` (≈L129–184) does: `column_projection` → normalize → `uniform_filter1d`
→ `find_peaks` → `_fit_uniform_grid` → `_detection_confidence` → return `LaneDetection`.
The only production caller of `column_projection` is this function (L135). `LaneDetection`
(dataclass L80) carries `projection`, `peak_indices`, `centers`, `confidence`, …

## Phase-0 scaffold — ALREADY BUILT (on a feature branch)
**Do not re-implement this.** The scaffold below is committed on that feature branch.
Branch your design off the scaffold and go straight to your
strategy function. The pieces you consume:
- `projection_strategies` Registry in `core/lanes.py` — register your strategy with
  `@projection_strategies.register("dX_name")`; signature
  `fn(image, n_lanes, *, expected_pitch, y_start, y_end) -> AdaptiveProjection`.
- `AdaptiveProjection(projection, row_weights=None, y_band=None)` — your return type. `y_band`
  may be a single `(lo, hi)` or a list of them (D4).
- The fallback + `_comb_snr` + surfacing are done. Call
  `detect_lane_centers(img, n, projection_strategy="dX_name")`; adoption is automatic.
- `_detect_from_projection` and `_comb_snr` are importable helpers if you need them.

Reference (what the scaffold does, for context only):

1. **Extract** the body L135–184 into
   `_detect_from_projection(proj, n_lanes, *, expected_pitch, image_width, smoothing_window,
   min_prominence, min_distance, min_width) -> _CoreDetection` where `_CoreDetection` bundles
   `centers, pitch, profile_width, peak_indices, n_detected, band_edge_frac, confidence, reason,
   blurred`. Pure function of a 1-D projection. (This is a mechanical move of existing lines.)

2. **Add** `_comb_snr(proj, centers, pitch) -> float`:
   ```
   p = minmax_normalize(proj)                     # 0..1
   on  = mean( interp(p, centers) )               # value at each of the N grid nodes
   off = mean( interp(p, centers ± pitch/2) )     # anti-phase midpoints (clip to [0,width-1])
   return on - off                                # higher = crisper, well-covered lanes
   ```
   Computed over **all N nodes**, so dropping lanes lowers it (protects Cy3).

3. **Strategy registry** (reuse `core/_registry.Registry`): name → `fn(image, n_lanes, *,
   expected_pitch, y_start, y_end) -> AdaptiveProjection(projection, row_weights|None, y_band|None)`.
   Register the baseline `"full"` = `column_projection(image, y_start, y_end)` with
   `row_weights=None, y_band=None`. Each DX registers one more.

4. **Rewire `detect_lane_centers`** with a new param `projection_strategy: str = "full"`:
   compute the `"full"` projection AND the strategy projection; run `_detect_from_projection` on
   each; score `_comb_snr` on each; **adopt the strategy only if**
   `comb_snr_strategy > comb_snr_full + _ADAPTIVE_MARGIN` **and** its confidence is not worse than
   full's (never trade a correct `low` on `partial_fill` for a false `ok`). Else keep full.
   Record which was used.

5. **Extend `LaneDetection`** with `row_weights: np.ndarray|None = None`, `y_band: tuple|None =
   None`, `projection_source: str = "full"`, `comb_snr: float = 0.0`. Keep `projection` = the
   chosen one.

6. **Surfacing (small):** add optional `y_band`/`projection_source` to
   `schemas/lanes.LaneDetectionResult`; `runfolder/commands.detect_lanes` passes them through;
   `render/lanes.render_lane_detection` already shades a band via `axhspan` — extend it to draw
   the chosen `y_band` and (optional) a row-weight strip. `cli/_render.render_lanes` forwards.
   Keep matplotlib only in `render/` + `cli/_render.py`.

`_ADAPTIVE_MARGIN` (small positive, e.g. 0.02 in normalized comb units) is a module constant;
tune it so controls (`control_*`) and `partial_fill` keep full-height and never regress.

## Integration decision (from the user)
**Default-on with fallback.** The winning design eventually becomes the default strategy. During
each DX branch, wire your strategy as the candidate and compare to `"full"` via the fallback and
the harness. Do not remove the `"full"` path.

## How to score (the harness is already built — `tests/synthetic/`)
```python
from tests.synthetic.harness import evaluate, format_table, baseline_detect
from gelpy.core.lanes import detect_lane_centers
mine = lambda img, n: detect_lane_centers(img, n, projection_strategy="dX_name")
print(format_table(evaluate({"baseline": baseline_detect, "dX": mine})))
```
Run on real images too:
```python
from tests.synthetic.real import load_labeled_images, score_real
for g in load_labeled_images():
    print(score_real(g, mine))
```
Metric recap (`harness.py`): a run **passes** if max center error ≤ 0.25×pitch; `confidence` is
**correct** when `ok`⇔pass; `false_ok` (silent wrong) is the worst, `false_low` (needless manual)
lesser. Composite = `0.5·pass + 0.5·flag_correct − false_ok`.

## Baseline to beat (8 seeds/condition)
Aggregate: **medErr 0.04p · pass 94% · flagOK 82% · falseOK 0% · falseLow 18% · score 0.88**.
Real: folding_screen 0.019×pitch `ok`; Cy3 ~0.00×pitch `ok`.

## Acceptance criteria (every DX)
1. `false_ok_rate` stays **0** in aggregate (never introduce a silent wrong answer).
2. `partial_fill` stays failed-and-flagged-low; `control_*` stay pass+ok (fallback holds).
3. **Real images do not regress**: folding_screen and Cy3 stay `ok` with ≤ baseline error.
   (Cy3 is the guard that your strategy didn't drop the ladder/scaffold lanes.)
4. Net improvement: aggregate `false_low_rate` **down** vs baseline (the headroom) and composite
   **up**, with no condition's composite dropping materially.
5. `pytest tests/synthetic tests/core/test_lanes.py` green; existing detection tests unchanged.
6. New unit test for your strategy fn (weights/band on a crafted array) + a harness-delta check.

## Branch/commit
Start a feature branch from the scaffold branch (which already contains the
Phase-0 scaffold + the frozen `tests/synthetic/` suite). Your commit adds only the strategy
function (+ its registration) and a unit test — keep the diff to `core/lanes.py` + your test.

## Reference files
- `src/gelpy/core/lanes.py` (seam), `src/gelpy/core/_registry.py` (registry pattern)
- `src/gelpy/schemas/lanes.py` (`LaneDetectionResult`), `src/gelpy/schemas/requests.py`
  (`DetectLanesRequest`), `src/gelpy/render/lanes.py`, `src/gelpy/cli/_render.py`
- `tests/synthetic/` (generator, conditions, harness, real) — DO NOT modify; it's the ground truth.
