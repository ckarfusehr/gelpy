# D1 — Row-weighted projection by horizontal informativeness

> **HISTORICAL DESIGN RECORD -- NEVER BUILT.** This design was specified and not
> implemented; D4 was chosen instead and shipped. Nothing in gelpy behaves as described
> below. Current behaviour: `docs/guide/lane-placement.md`.


**Read `00-context.md` first.** This is the simplest design and the floor the others must beat.
No pitch dependency, single pass.

## Idea
Instead of a plain mean down columns, weight each row by how much horizontal *structure* it
carries, so band rows dominate and empty/flat rows contribute ~nothing. Dust rows get only a
little weight (one narrow spike barely lifts a row's horizontal contrast), band rows get a lot
(a band lights up many columns at once).

## Strategy function `d1_rowweight`
`fn(image, n_lanes, *, expected_pitch, y_start, y_end) -> AdaptiveProjection`:
1. Restrict to `image[y_start:y_end]` (respect the existing manual band args).
2. Per row `r`, weight `w[r] = max(0, contrast(r) - baseline)` where `contrast(r)` is a robust
   horizontal-structure measure. Start with **row max − row median** (per row); alternative is
   x-variance. `baseline` = the median of `contrast` over all rows (subtract the flat-row floor).
   Clip negatives to 0.
3. Optionally smooth `w` lightly over y (e.g. `uniform_filter1d(w, 3)`) to avoid single-row
   spikiness, then normalize `w` to sum 1 (guard all-zero → fall back to uniform weights).
4. `projection = (w[:,None] * band).sum(axis=0)` (weighted column mean).
5. Return `AdaptiveProjection(projection, row_weights=w, y_band=None)` (D1 produces weights, not a
   hard band; leave `y_band=None`).

## Why it should help / where
- Recovers SNR on `snr_015/020`, `narrowband_*`, `gradient_bg` (empty rows down-weighted → the
  gradient ramp and read-noise contribute less; the thin band dominates).
- Handles multi-band lanes and `spread_migration`/Cy3 **for free**: every band-bearing row (at
  any y) gets weight, so no lane is dropped — this is D1's key robustness advantage over the
  window designs. Verify Cy3 stays `ok`.

## Risks / mitigations
- A bright non-lane horizontal artifact (edge glow) could get weight → mitigated: uniform glow has
  low row max−median; and the comb-SNR fallback rejects it if it hurts.
- `row max − row median` is sensitive to a single hot pixel → the light y-smoothing + using median
  (not mean) as the floor guards it; consider a high-percentile (row 95th − 50th) if needed.

## Acceptance (beyond the shared list)
- Aggregate `false_low_rate` clearly below baseline's 18%; composite up.
- Cy3 and example_gel_1 stay `ok`, error ≤ baseline (proves no lane dropped).
- Unit test: craft a 2-D array with signal rows + empty rows + a one-row dust spike; assert `w`
  is high on signal rows, ~0 on empty rows, and small on the dust row.

## Files
`core/lanes.py` (strategy fn + register). No new schema knobs needed. Optional: render the
`row_weights` strip (D1 is the natural first consumer of that overlay).
