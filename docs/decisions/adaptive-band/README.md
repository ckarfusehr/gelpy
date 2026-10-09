# Adaptive y-band lane detection — design docs

> **HISTORICAL DESIGN RECORD -- ONE OF THESE FOUR SHIPPED.** D4 (multi-band union) became
> the default projection strategy. D1, D2 and D3 were never built as standalone
> strategies, although D2's comb-periodicity row score shipped inside D4. Baseline numbers
> quoted in these pages are FULL-HEIGHT numbers, which stopped being the default.
> Current behaviour and measured accuracy: `docs/guide/lane-placement.md`.


Implementation plans for making `detect_lane_centers` adaptively emphasize the informative rows
of a gel instead of averaging the full height. Foundation (seeded synthetic suite + evaluation
harness + real fixtures) already lives in `tests/synthetic/` on branch `feat/synth-gel-suite`.

**Order:**
1. `00-context.md` — shared context, the code seam, the Phase-0 scaffold every design reuses,
   the harness/metric, and acceptance criteria. **Read first.**
2. One design per branch (build/measure one at a time; each self-contained after 00-context):
   - `D1-row-weight.md` — simplest; the floor. Row weight = horizontal contrast.
   - `D2-comb-periodicity.md` — recommended core. Row weight = comb-likeness at the pitch.
   - `D3-window-search.md` — explicit `[y_start,y_end]`; coverage-gated search.
   - `D4-multiband-union.md` — union of y-clusters; the Cy3 multi-y case.

Each is scored by `tests/synthetic/harness.py` against the full-height baseline; the winner
becomes the default strategy (default-on with comb-SNR fallback — never worse than full height).
