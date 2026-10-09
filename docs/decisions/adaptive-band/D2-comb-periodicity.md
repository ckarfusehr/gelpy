# D2 — Comb-periodicity row weighting (recommended core)

> **HISTORICAL DESIGN RECORD -- NEVER BUILT.** This design was specified and not
> implemented; D4 was chosen instead and shipped. Nothing in gelpy behaves as described
> below. Current behaviour: `docs/guide/lane-placement.md`.


**Read `00-context.md` first.** Strongest discriminator; encodes the physics directly (a band
row lights up N lanes at the lane pitch; dust is a single tooth, not a comb). Uses the pitch prior.

## Idea
Weight each row by how **comb-like** its x-content is at the lane pitch. A row crossing the band
region lights up ~N columns spaced at the pitch → high comb response. A dust row (one narrow
spike) or a background row → low response. Then project with those weights.

## Strategy function `d2_comb`
`fn(image, n_lanes, *, expected_pitch, y_start, y_end) -> AdaptiveProjection`:
1. **Pass 1 pitch:** build the full-height smoothed projection and get the pitch from the existing
   `_matched_filter_pitch(blurred, n_lanes, expected_pitch)` (anchored by the `width/N` prior, so
   usable even from a noisy projection). Reuse it; do not re-implement.
2. **Row comb-score:** for each row `r` of `image[y_start:y_end]`, compute the matched-filter
   response of an N-tooth comb at that pitch against the row. Reuse the interp-at-grid machinery
   from `_matched_filter_pitch` (sample the row at `offset + k·pitch`, best offset). Subtract a
   non-periodic baseline (e.g. the row's response at a half-pitch-shifted anti-phase comb, or the
   row mean) so a single wide smear / DC row doesn't score high. Clip at 0 → `w[r]`.
3. Normalize `w` (sum 1; guard all-zero → uniform), optional light y-smoothing.
4. `projection = (w[:,None]*band).sum(0)`; return `AdaptiveProjection(projection, row_weights=w,
   y_band=None)`.
5. **Optional EM (1–2 iters):** re-estimate pitch from the sharpened projection, re-score rows.
   Stop when pitch change < ~1px. Keep it optional/bounded; measure whether it helps before adding.

## Why it should help / where
- Best on dust cases (`dusty`, `dusty_in_band`, `dusty_faint`): dust rows never form a comb → near
  zero weight, so dust stops injecting false peaks. This is D2's edge over D1.
- Recovers confidence on the low-SNR/geometry cases like D1, and keeps multi-y coverage
  (`spread_migration`, Cy3): rows at *any* y that are comb-like get weight, so lanes aren't dropped.

## Risks / mitigations
- Depends on pass-1 pitch. Mitigated by the `width/N` prior anchoring + the comb-SNR fallback (bad
  pitch → weights don't help → fallback to full). Optionally the EM step.
- Chicken-and-egg on very low SNR: if pass-1 pitch is wrong the weights mislead. The fallback
  catches it; verify `partial_fill` (wrong-pitch regime) still falls back to full and stays low.

## Acceptance (beyond the shared list)
- Largest `false_low` reduction of the four expected; dust conditions' composite ≥ baseline.
- Cy3/folding_screen stay `ok`. Verify comb-SNR(strategy) on Cy3 is NOT falsely high from the
  bright samples alone (all-N-node scoring should keep it honest).
- Unit test: craft rows = {a comb at pitch P across N columns} vs {random spikes} vs {flat};
  assert comb rows score high, spike/flat rows low.

## Files
`core/lanes.py` (strategy fn + register; reuse `_matched_filter_pitch`). No new schema knobs.
