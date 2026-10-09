# D4 — Multi-band union (handles multi-band / multi-y; most complex)

> **HISTORICAL DESIGN RECORD -- SHIPPED** as `d4_union`, the default projection strategy,
> adopted per image only when it beats full height. Implementation details
> diverge from this document. Current behaviour: `docs/guide/lane-placement.md`.


**Read `00-context.md` first.** The design built for the Cy3 reality: different lanes carry bands
at different y, so the informative region is a **union of y-clusters**, not one window. Do this
one last; consider it only if D2/D3 underperform on the multi-band/spread cases.

## Idea
Find *all* informative y-clusters (not just the brightest), keep those that add lane coverage, and
project over their union. A ladder lane's cluster, a scaffold band, and the sample band all
contribute — no lane dropped.

## Strategy function `d4_union`
`fn(image, n_lanes, *, expected_pitch, y_start, y_end) -> AdaptiveProjection`:
1. Compute a per-row informativeness score `s[r]` — reuse **D1's row contrast** or **D2's comb
   score** (prefer D2's if D2 landed; import/share the helper). Work within `[y_start:y_end]`.
2. Threshold `s` to a foreground row-mask (robust threshold, e.g. median + k·MAD), and cluster
   contiguous True runs into candidate y-bands (with a small gap-merge so a slightly split cluster
   stays one band).
3. **Greedily keep bands that add coverage:** start from the empty set; for each candidate band by
   descending score, add it if projecting over the running union raises the count of covered grid
   nodes (all-N coverage, as in D3). Stop when coverage plateaus. This keeps the sample cluster AND
   the ladder/scaffold clusters on Cy3.
4. `projection = column_projection over the union of kept row-ranges` (mask-weighted mean, i.e.
   average only the kept rows). Return `AdaptiveProjection(projection, row_weights=mask.astype
   (float), y_band=<list of (lo,hi) kept bands>)` — note `y_band` here may be **multiple** ranges;
   ensure the dataclass/overlay can hold a list (or the union bounding box + weights).
5. Fallback (Phase-0) still applies.

## Why it should help / where
- The only design that explicitly targets `spread_migration`, `ladder_4band`, and **Cy3** without
  relying on the fallback: it should actively *improve* those (crisper projection than full height,
  all lanes covered) rather than merely not-regress.

## Risks / mitigations
- Most knobs (threshold, gap-merge, stop rule) → most overfitting risk. Keep thresholds robust
  (MAD-based, not absolute) and lean on the coverage-plateau stop. Validate across seeds, not one.
- Multi-range `y_band` complicates surfacing → simplest: store the list; overlay shades each range.
  If the dataclass field is scalar, store the bounding union and put the list in a diagnostics dict.

## Acceptance (beyond the shared list)
- On `spread_migration`, `ladder_4band`, and Cy3: composite ≥ baseline AND (ideally) the projection
  is measurably crisper (higher comb-SNR) than full-height while keeping all N covered.
- No condition regresses; `false_ok` stays 0; controls/partial_fill hold.
- Unit test: a synthetic gel with two disjoint y-clusters (some lanes only in cluster A, others
  only in B) → assert both clusters are kept and every lane is covered.

## Files
`core/lanes.py` (strategy fn + register; reuse D1/D2 row score). May need `LaneDetection.y_band`
to accept a list (coordinate with Phase-0). Overlay shades each kept range.
