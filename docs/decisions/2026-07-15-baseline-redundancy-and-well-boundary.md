# Two fit-time problems found in real-data use (baseline redundancy + well boundary)

**Status:** OPEN — problem statement only, not yet designed or implemented. Needs an empirical
decision on the synthetic harness before any code change.

**Provenance:** Surfaced 2026-07-15 while analysing a real agarose gel (`robust_plane` background subtraction, `bayesian` tier). Two operator
observations, recorded here in the operator's framing and then worked into concrete failure
modes + candidate fixes. This is a note to a future rehaul, not a spec of an agreed solution.

---

## Problem 1 — The fit-time 1-D baseline is (probably) redundant and net-harmful after 2-D background subtraction

### Operator observation
> "The baseline is always flat. I don't know if there is any point in fitting any baseline
> after background subtraction. I don't think there is. Fitting the baseline, as I see it now,
> just introduces more errors than anything else."

### What the code does today
Two separate baseline stages exist, by design (`robust-peak-fitting-design.md` §Caching:
"1-D baseline is fit-time … the 2-D background stage is unchanged"):

1. **2-D background removal** (`core/background/`, `robust_plane`) — a low-order IRLS surface
   subtracted from the whole image. The skill now makes this **mandatory before any profile/fit**.
2. **1-D fit-time baseline** (`core/peaks/baseline.py`, `BaselineConfig` default
   `method="poly", poly_order=2, protect_well=True, well_frac=0.08`) — re-estimated per lane,
   *on top of* the already-background-subtracted profile, coupled into the fit.

The peak-fitting spec's problem statement asserts the profile still sits on "a **structured
(non-flat, drifting/curved) baseline** even after 2-D correction"
(`2026-07-14-robust-peak-fitting-design.md:19`). **Real data contradicts this assumption** for
gels where `robust_plane` did its job: the residual per-lane background is essentially flat and
near-zero, so the 1-D baseline has nothing legitimate left to fit.

### The failure mode (evidence from this session)
With the residual background already flat, the polynomial baseline's remaining degrees of
freedom have only one thing left to fit: **real signal**. Observed on this gel:

- On a slice that starts right after a hard exclusion boundary (the well dip, see Problem 2),
  an `poly_order=2` baseline bends up under a genuine post-boundary smear/shoulder and **silently
  absorbs it** — the "baseline too high" the operator saw. It disappears from *both* the signal
  integral and the component areas, so `coverage` still reads ~100% and `confidence` still reads
  `ok`. Nothing flags it. (Empirically, raising `well_frac` 0.08→0.25 made this *worse*, not
  better — more masking = more freedom to bend elsewhere.)
- The interim mitigation this session was a manual `--poly-order 0` (constant offset) on affected
  lanes — added to the CLI this session (`fit run/refit-lane --poly-order`). It removes the DOF
  that eats signal, but it is a **manual per-lane knob, not a principled default**, and it doesn't
  answer the underlying question.

### The underlying question to resolve
After a *good* 2-D background subtraction, what is the correct fit-time baseline?

- **Hypothesis (operator's, and supported by this session):** `none`, or at most a single
  **constant offset** (`poly_order=0`). Any order ≥ 1 buys the fit freedom whose only use, on a
  correctly-background-subtracted lane, is to misfit signal as baseline.
- **Counter-consideration (why the stage was not simply `none`):** a single low-order 2-D surface
  cannot always capture *per-lane* residual structure — smile, local staining gradients, a lane
  that ran hot. In those cases some per-lane baseline freedom is legitimate. So "always `none`"
  may be too aggressive for gels with a poorer 2-D correction; the safe floor is likely
  *offset-only*, not *nothing*.

### Candidate fixes (pick after measuring)
1. **Change the default:** when `recipe.background` is set (2-D subtraction happened), default the
   fit-time `BaselineConfig` to `method="none"` or `poly_order=0`. Keep higher orders opt-in for
   the no-2-D-background path. This makes the pipeline *aware* of whether background was already
   removed (it currently is not — the two stages are independent).
2. **Offset-only baseline as the post-background floor:** a dedicated `method="offset"` (fit a
   single constant = robust min/mode of the lane) that cannot bend by construction.
3. **Keep poly but constrain it against signal:** only accept baseline curvature that is
   *supported by both window edges* (anchor points outside any detected component), so it can
   track true edge-to-edge drift but cannot bow up under an interior band. More complex; probably
   not worth it if (1)/(2) measure well.

### How to decide (do not guess — measure)
The synthetic harness (`tests/synthetic/`) already generates ground-truth profiles with known
areas and structured baselines. Add/repurpose cases that are **2-D-background-subtracted first**
(flat residual), then compare area-fraction error and false-absorption rate across
`baseline ∈ {none, offset, poly_order=0, poly_order=2 (current default)}`. Decision criterion:
lowest area error + zero silent-absorption on the flat-residual cases, without regressing the
genuinely-drifting (no-2-D-background) cases. The current suite's `test_baseline_never_silently_
wrong_and_has_headroom` is the natural home, but note it is **already failing on `main`** for an
unrelated headroom-threshold reason (see caveat at bottom) — fix/relocate rather than trust it
blindly.

---

## Problem 2 — The well/gel-start boundary is gel-specific and is not auto-detected

### Operator observation
> "The position of the well in pixels will vary significantly — it will always be on the left
> side but it varies between gels, so it's not general."

Plus the correct physical reading of the well, established this session:

- The well is a **cavity with less gel material than the surrounding lane**, so it reads as a
  **dip below background**, never a peak (less material → less signal than the extrapolated
  background predicts). This is a fixed optical/geometric artifact, reproducible across every
  lane of a given gel.
- **Genuine well-trapped aggregate** (material too large to enter the gel) appears as a real,
  distinct **peak immediately after that dip**, at the dip's right zero-crossing. Its area is the
  "trapped in the well" quantity.
- **Everything left of that peak** — the dip, and anything before it — **is not part of the
  data-containing gel** and must be excluded from baseline fitting *and* signal integration.

### Why the current mechanisms are not general
Both the classifier and the baseline protection locate the well by a **fixed fraction of the
window**, which does not transfer between gels:

- `core/peaks/classify.py:44` — `well_edge = well_frac * length` (`well_frac=0.08`). A component is
  "well" only if its center is within the first 8% of the window. On this gel the peak right
  after the well dip sat *past* 8%, so it was mislabeled `sample`; on another crop the true well
  could be at 3% or 20%.
- `core/peaks/baseline.py` `well_region()` — same `well_frac` fraction, used to zero-weight the
  well during baseline fitting.

The interim mitigation this session was a manual `--slice 100 550` (eyeballed per gel, after
reading the dip's zero-crossing off one or two lanes). Correct for this crop; **not general** —
exactly the operator's point. It is also exactly the failure mode gelpy already rejects for *lane
placement* (fixed pixel positions never transfer between crops), applied to the *migration axis*.

### The fix direction: detect the boundary from the image, don't assume a fraction
The dip is a robust, physical signature that can be detected generally — analogous to how lane
centers are image-derived, not geometrically guessed:

1. In the first portion of each lane profile (post-2-D-background, so "below background" = "below
   0"), find the **sustained negative excursion** near the origin (the well cavity) and take its
   **right zero-crossing** as the gel-start boundary `t0`.
2. It is the *same physical well* on every lane of one gel — detect on the strong lanes, then take
   a robust consensus `t0` (median) and apply it to the whole gel (same spirit as the even-grid
   lane prior).
3. Use `t0` to (a) set the fit window start automatically (no manual `--slice`), (b) define the
   "well" position bucket by *location relative to `t0`* instead of `well_frac * length`, and
   (c) bound baseline fitting to `t ≥ t0`.

Open sub-questions for the design pass:
- Should `t0` live on the **recipe** (per-gel, like lane centers, editable + hashed) or be derived
  fresh each fit? Recipe is more in keeping with the "durable, verifiable, image-derived intent"
  model and makes it cacheable.
- How to distinguish the well-cavity dip from ordinary baseline undershoot / a lane that simply
  starts high? (Require the excursion to go meaningfully below the noise floor *and* recover, and
  to be consistent across lanes.)
- Interaction with the "inclusive, never remove signal" rule (spec §Confirmed decisions #2): the
  region left of `t0` is **not** signal being dropped — it is *non-gel*, correctly outside the
  measurement domain. Worth stating explicitly so this doesn't read as a violation of that rule.

---

## Cross-cutting

Both problems share one root: **the fit stage assumes structure (a curved baseline; a
fixed-fraction well) that is actually either absent (baseline, after good 2-D subtraction) or
gel-specific and image-derivable (well boundary).** The fix philosophy in both cases is the same
one gelpy already uses for lane placement — *derive it from the image / from whether the prior
stage ran, rather than assuming a fixed shape or fraction* — and *measure the change on the
synthetic harness before shipping a new default.*

## Interim state (so a future reader knows what the current runs did)
- CLI gained `fit run/refit-lane --poly-order N` this session (constant/…/quadratic baseline).
- `_merge_fit` bug fixed this session (nested-config overrides were being flattened to dicts and
  silently dropped on `refit-lane`).
- Real-run provenance: `--slice 100 550 --poly-order 2` (order 0 on the couple of lanes where the
  order-2 baseline visibly ate a shoulder), `--method bayesian`. These are workarounds; they do
  not constitute a fix for either problem above.

## Caveat for whoever picks this up
`tests/synthetic/test_suite.py::test_baseline_never_silently_wrong_and_has_headroom` and four
`tests/core/test_lanes.py` / `test_lane_strategy.py` cases are **already failing on `main`**
independent of this note (verified by stashing all local edits and re-running — the failures
reproduce on a clean tree). Don't attribute them to the baseline/well work, and don't trust the
headroom test as-is when measuring Problem 1 — check why it currently reports `false_low_rate=0.0`
first.

---

## Addendum 2026-09-10 — first measurement of Problem 1, and the headroom caveat resolved

Done while preparing the 1.0 release, in the spirit of "do not guess — measure". This is a
*partial* answer: it quantifies the absorption mechanism but does not yet run the full
harness across baseline settings, so Problem 1 stays OPEN.

**Absorption is monotone in baseline freedom.** A genuine broad smear (sigma = 55 px, flat
near-zero residual background, one sharp band alongside it), measured as the fraction of the
smear's true area that the baseline swallowed:

| baseline | smear absorbed |
|---|---|
| `none` | 0.0% |
| `poly` order 0 | 2.1% |
| `poly` order 1 | 2.2% |
| `poly` order 2 (current default) | 5.5% |
| `poly` order 3 | 13.1% |
| `arpls` lam=1e5 | 34.4% |
| `asls` lam=1e5 | 58.5% |

This supports the operator's hypothesis quantitatively: on a flat residual background the
default eats 5.5% of a smear where an offset-only baseline eats 2.1%, and every extra degree
of freedom costs more signal. It also corrects two claims that were in the source: the
`BaselineConfig` docstring said "poly order 3 leaves [a smear] intact" (it absorbs 13%) and
put arPLS at "~25%" (34%). Both are now stated with these numbers.

What this does *not* settle: whether order 0 or `none` is the right post-background default
depends on the no-2-D-background and per-lane-residual-structure cases, which need the full
synthetic harness comparison described above. Candidate fix (1) — make the fit stage aware of
whether `recipe.background` is set — remains the most promising direction.

**The headroom caveat is resolved.** `false_low_rate` read 0.0 because
`tests/synthetic/harness.baseline_detect` did not pin `projection_strategy="full"`: it
inherited the module default, which became `d4_union`. The "baseline" was therefore
the shipped detector, so the test asserted that the current detector leaves headroom against
*itself*, and `test_d4_harness_composite_at_least_matches_baseline` compared d4_union to
d4_union — a test that could not fail. Fixed in the 1.0 release by pinning the reference
explicitly. Related: `_SMOOTHING_WINDOW_FRACTION` is 0.7, not the 0.35 its own comment and two
schema descriptions claimed; 0.7 measures better (composite 0.98 vs 0.94, 0% vs 2.67% false-ok,
0.019x vs 0.044x pitch error on `folding_screen`), so the code was right and the prose was
stale.
