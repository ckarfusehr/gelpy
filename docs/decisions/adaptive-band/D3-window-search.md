# D3 — Best-y-window search (explicit band; most interpretable)

> **HISTORICAL DESIGN RECORD -- NEVER BUILT.** This design was specified and not
> implemented; D4 was chosen instead and shipped. Nothing in gelpy behaves as described
> below. Current behaviour: `docs/guide/lane-placement.md`.


**Read `00-context.md` first.** Returns a concrete `[y_start, y_end]` the user can see and tweak —
it slots straight into the existing `y_start/y_end` args and the `axhspan` overlay.

## Idea
Search over contiguous y-windows; for each, run detection and score the internal comb-SNR; pick
the best window. Directly optimizes the actual objective and yields a legible band.

## Strategy function `d3_window`
`fn(image, n_lanes, *, expected_pitch, y_start, y_end) -> AdaptiveProjection`:
1. Candidate windows over a coarse grid within `[y_start:y_end]`: positions × heights, with a
   **minimum height** (e.g. ≥ ~8% of image height) so a razor-thin bright band can't win alone.
   Keep it coarse (e.g. ~6 starts × ~5 heights) — cheap.
2. For each window `w`: `proj_w = column_projection(image, w0, w1)`; run
   `_detect_from_projection`; compute `_comb_snr` over **all N** grid nodes.
3. **Score = comb_snr + coverage_bonus − height_penalty**, where coverage counts how many of the N
   grid nodes have a real detected peak nearby (reuse the coverage idea in `_detection_confidence`).
   The coverage term is essential: it stops the search from picking the bright sample window on
   Cy3 and dropping lanes 0–1 (that window has low all-N coverage → loses).
4. Pick the best window; `projection = proj_best`; return `AdaptiveProjection(projection,
   row_weights=None, y_band=(w0, w1))`.

## Why it should help / where
- Explicit and inspectable; great for the narrow-band cases and for surfacing *where* the signal
  is. `y_band` flows to the result + overlay so the user sees the chosen region.
- The coverage-gated score is what makes it safe on multi-y gels.

## Risks / mitigations
- **Overfitting to one strong band / dropping lanes** — the deepest risk (Cy3). Mitigated by: the
  min-height floor, the all-N coverage term in the score, and the comb-SNR fallback (if no window
  beats full-height by the margin, keep full). On Cy3 the correct outcome is likely "no single
  window wins → fall back to full" — verify that explicitly; that is a *success*, not a failure.
- Cost: keep the grid coarse; each candidate is a cheap projection + fit.

## Acceptance (beyond the shared list)
- Narrow-band / low-SNR conditions improve (false_low down); `y_band` populated and sensible on
  those (brackets the true band region — compare to `SynthGel.band_y_lo/hi`).
- Cy3 and folding_screen stay `ok` with error ≤ baseline (via correct window OR fallback).
- Unit test: on a synthetic gel with a known thin band, assert the chosen `y_band` overlaps the
  true band region; on Cy3-like multi-y (use `spread_migration`), assert it does not drop lanes
  (falls back or picks a covering window).

## Files
`core/lanes.py` (strategy fn + register). Surface `y_band` through `LaneDetectionResult` +
overlay (the `axhspan` already exists — pass the chosen band). This design is the main consumer of
the `y_band` surfacing in Phase-0.
