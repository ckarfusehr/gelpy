---
title: The fit engine
audience: both
summary: How gelpy chooses a lane's peak count, shapes and baseline without per-lane tuning, what area_fraction and coverage actually mean, and how far to trust a fit.
subject:
  - src/gelpy/core/peaks/base.py
  - src/gelpy/core/peaks/baseline.py
  - src/gelpy/core/peaks/detect.py
  - src/gelpy/core/peaks/driver.py
  - src/gelpy/core/peaks/gaussian.py
  - src/gelpy/core/peaks/emg.py
  - src/gelpy/core/peaks/bi_gaussian.py
  - src/gelpy/core/peaks/pseudo_voigt.py
  - src/gelpy/core/peaks/classify.py
  - src/gelpy/core/peaks/quality.py
  - src/gelpy/core/peaks/strategies/_common.py
  - src/gelpy/core/peaks/strategies/grow_select.py
  - src/gelpy/core/peaks/strategies/sparse_seed.py
  - src/gelpy/core/peaks/strategies/bayesian.py
  - src/gelpy/core/peaks/strategies/registry.py
  - src/gelpy/schemas/baseline.py
  - src/gelpy/schemas/fit_config.py
  - src/gelpy/schemas/peaks.py
subject_digest: c91a763b7be3d00f
---

## There is no peak-detection tuning to do

The engine picks the number of components per lane by AICc and each component's shape by a
second AICc comparison. Fitting defaults to the `grow_select` strategy, with two higher tiers
available. <!-- claim: default-fit-strategy -->

The five flags 0.x needed for this — `model`, `maxima-threshold`, `maxima-prominence`,
`peak-width`, `sigma` — are gone, without aliases, and **nothing replaces them**. Measured on `examples/data/example_gel_1.tif` with pure defaults, the 20
lanes come back with 3 to 9 components each — chosen per lane, from one config.

What is left is a different kind of knob. None of these tells the engine where a peak is:
`--method` (which tier proposes components), `--shapes` (the menu each component is selected
from), `--baseline` / `--poly-order` (what counts as background under the peaks), `--seed`
(the RNG for the sampling tier), `--auto-escalate` (climb tiers on a low-confidence lane).
The stopping controls and priors (`max_components` 12, `min_snr` 3.0, `width_prior_strength`
0.5, `tail_shrinkage` 0.3, `max_width_frac` 0.9) live on `PriorConfig` and have **no CLI
flag** — they are calibration, not per-lane workflow. Edit them in `fit_config.json`
([below](#hand-tuning-one-lane)).

```bash
# doc-test: skip
# The doc test sets these two for you; set them yourself if you are following along.
export IMAGE=examples/data/example_gel_1.tif
export RUN=/tmp/gelpy-fit-engine
```

```bash
gelpy init "$RUN" "$IMAGE" --n-lanes 20 --format text
gelpy lanes detect "$RUN" --apply --format text
gelpy background remove "$RUN" --format text
gelpy fit run "$RUN" --lane 3 --format text
```

```console
{"recomputed": ["background", "fit:lane=3", "profile:lane=3"], "reused": ["image"], "lanes":
 {"lane=3": {"status": "ok", "confidence": "ok", "method": "grow_select", "bands": 7,
 "coverage": 1.037, "area_fractions": [0.0185, 0.039, 0.0409, 0.1035, 0.0513, 0.5917, 0.1927],
 "classes": ["well", "smear", "sample", "sample", "smear", "smear", "smear"],
 "reasons": ["note: band 5 overlaps a neighbor (areas correlated)", ...]}}, ...}
```

## One pass, in order

```text
profile
  -> baseline           estimated once and subtracted; never a fit parameter
  -> greedy growth      seed at the residual argmax; keep the new component only if AICc
                        improves by more than 6.0; stop at the first failure
  -> consolidation      adjacent pair whose shared valley fills >= 80% of the smaller peak:
                        retry as ONE component in every allowed shape, accept if AICc
                        worsens by <= 4.0
  -> shape selection    per component, others frozen: swap only if AICc improves by more
                        than 2.0
  -> collapse           maxima < 2.0 px apart are one feature; keep the larger area
  -> prune              drop below-noise height, or a maximum far off the profile
  -> final joint refit  all components, all shapes, one robust-MAP least_squares
```

The three AICc gates are deliberately asymmetric: a new component must *earn* more than 6 AICc
units, a merge is accepted even if it *costs* 4, a shape swap must earn more than 2. The engine
is biased toward **fewer, simpler** components, which is why it does not shatter a tailed band
into a picket fence. Growth is greedy and stops at the first non-improving addition; it is not
an exhaustive search over N.

## The baseline is the biggest lever in the whole engine

The fit-time baseline is a per-lane 1-D estimate, distinct from the 2-D
[background stage](background.md). It defaults to a polynomial of order 2, not arPLS.
<!-- claim: default-baseline-is-poly-order-2 --> The estimator is an arPLS-*weighted*
polynomial: the arPLS logistic reweighting (Baek et al. 2015) pins the baseline to the right
*level* by smoothly down-weighting points above it, while the order-2 basis is too stiff to
bend under a localized smear. A smear is signal, so this asymmetry is the point.

How stiff, in numbers. Take a 600-px trace with a sharp band (sigma 8 px at x=180), a genuine
broad smear (amplitude 0.25, sigma 55 px at x=380), Gaussian noise sd 0.004, and **no
background at all** — anything the baseline picks up is signal it ate. As a fraction of the
smear's area: `none` 0.0 %, `poly` order 0 **2.1 %**, order 1 2.2 %, order 2 (the default)
**5.5 %**, order 3 **13.1 %**, `arpls` **34.4 %**, `asls` **58.5 %**. Absorption is monotone in
baseline freedom, which is exactly why the stiffest option that can still track real drift is
the default. <!-- claim: baseline-absorption-is-monotone-in-order --> `arpls`/`asls` are
Whittaker smoothers: at `lam = 1e5` a second-difference smoother preserves nothing shorter than
about `2*pi*lam**0.25`, roughly 110 px, so it follows anything broader. Reach for them only
when the background is genuinely wiggly and non-polynomial, at that cost.

The cost is not academic. The same real lane, fitted six times with nothing changed but the
baseline (lane 3 of the example gel, area-normalized so the profile integral is exactly 1):

| baseline | A_sig | components | largest `area_fraction` | coverage | confidence |
|---|---|---|---|---|---|
| `none` | 1.000 | 8 | 0.493 | 1.007 | ok |
| `poly` order 0 | 0.966 | 8 | 0.489 | 1.015 | low |
| `poly` order 2 | 0.882 | 7 | 0.592 | 1.037 | ok |
| `poly` order 3 | 0.877 | 7 | 0.597 | 1.040 | ok |
| `arpls` | 0.613 | 6 | 0.810 | 1.034 | ok |
| `asls` | 0.544 | 9 | 0.631 | 1.025 | ok |

Read that before you report a fraction. The baseline sets `A_sig`, the denominator of every
`area_fraction`. Changing it alone moved the dominant band from 0.49 to 0.81, and five of the
six configurations still reported `confidence: "ok"`. Coverage cannot catch this, because
over-subtraction shrinks the components and the denominator together. **Fractions are
comparable only across lanes and runs fitted with the same
baseline.** The default removed 11.8 % of this lane's integral on an image that had already
been background-corrected; whether a fit-time baseline earns its place after a good 2-D
subtraction is an open question, recorded in `core/peaks/baseline.py`.

```bash
gelpy fit refit-lane "$RUN" --lane 3 --method grow_select --baseline arpls --format text
gelpy fit refit-lane "$RUN" --lane 3 --baseline poly --poly-order 2 --format text
```

`protect_well` (default true) zero-weights the first `well_frac` of the window (default 0.08)
during baseline estimation, so a loaded pocket is never absorbed as background; the same field
sets the classifier's well boundary. Zero-weighting also removes the *anchor* that holds a
higher-order basis down at the origin end. Measured on a 400-px trace with a flat 0.05 offset
and one broad shoulder (amplitude 0.25, sigma 25 px at x=60), whose left flank overlaps that
prefix: the default baseline bows from 0.025 up to 0.339 and the recovered signal integral
comes out at **-0.39x** the true value, i.e. negative. With `--poly-order 0` it is 1.000. Use
order 0 when the window starts right after a hard exclusion boundary — a well-cavity dip, a
manual `--slice` — where no anchoring feature is left to hold a curve down.

## Seeding: the residual argmax, once per round

One seeder, running on the *current* residual: smooth by 1.5 px, clip negatives, take the
argmax, estimate the scale from the local half-maximum width (FWHM -> sigma), report
`snr = height / noise_sigma`. Growth stops when `snr < min_snr` (3.0). Working on the residual
is what separates shoulders and overlaps without a prominence knob: fit the tall peak, and its
neighbour becomes the next argmax.

`noise_sigma` is estimated once per lane from the MAD of the second difference
(`sigma = 1.4826 * MAD / sqrt(6)`, since the second difference has variance `6*sigma^2`),
floored at 0.8 % of the profile's 1-99 percentile range. The floor matters: a digitized profile
can have near-zero pixel noise, which would drive the seeding gate to zero and set the fitter
chasing sub-pixel wiggles.

## The four shapes

Every shape has a closed-form area over the whole real line, role-tagged parameters, and a
continuous limit back to the Gaussian, so shape selection cannot jump. The functional forms are
in the four modules under `core/peaks/`; what matters downstream is the area, where the maximum
sits, and the Gaussian limit. `A` is amplitude, `m` the center *parameter*.

```text
gaussian      area = A * s * sqrt(2*pi)                        maximum at m, height A

emg           Gaussian of height A convolved with a unit-area one-sided exponential of
              scale tau.
              area = A * s * sqrt(2*pi)     independent of tau AND of its sign
              tau is SIGNED: tau < 0 is the same curve mirrored about m, i.e. a left
              tail; |tau| <= 1e-6 evaluates the plain Gaussian. The maximum is NOT at
              m -- it sits on the tail side, and the height at m is below A.

bi_gaussian   area = A * sqrt(2*pi)/2 * (s_left + s_right)      maximum at m, height A
              The only shape with two width-role params: two bounds, two prior rows.

pseudo_voigt  Gaussian/Lorentzian blend; Lorentzian HWHM g = s*sqrt(2*ln2), FWHM-matched.
              area = A * ( (1-eta)*s*sqrt(2*pi) + eta*pi*g )    maximum at m, height A
              eta -> 0 is bit-identical to the Gaussian. Sharp core, heavy wings.
```

One trap: `EmgPeakParams`' docstring, and therefore the generated
[contracts reference](../reference/contracts.md), says the maximum sits *left* of `center`. The
code and `tests/core/test_peaks.py` put it on the tail side — for `A=1, m=50, s=5, tau=8` the
argmax is at 54.53 and the height at `m` is 0.507. Trust the code.

The areas are exact by derivation, and `tests/core/test_peaks.py` asserts three of them
(`gaussian`, `emg`, `bi_gaussian`) against Simpson quadrature on `[-500, 500]` at `rtol=2e-3`;
against adaptive quadrature over the whole line they reproduce to 3e-12 or better.
`pseudo_voigt` is *not* in that test, and on that same finite window Simpson under-reads its
closed form by 0.5 % at `eta=0.3` and 1.1 % at `eta=0.9`. That is not a formula error but the
practical consequence of integrating over the whole line: a component whose tail or Lorentzian
wing runs off the window contributes area that was never measured, which inflates `coverage`.

## The joint fit

`driver.fit_components` is the only fitter in the engine. All tiers call it; all mixed shapes
go into **one** `scipy.optimize.least_squares` over the concatenated parameter vector.

- **Residuals are noise-normalized**: `(model - signal) / noise_sigma`, which puts the robust
  scale, the priors and the covariance on a footing independent of the profile's absolute
  scale — necessary, since area-normalized amplitudes here are ~1e-3.
- **Robust loss**, `soft_l1` by default, `f_scale = 3.0`, so a dust spike beyond ~3 noise sigma
  stops dragging the real bands. `linear`, `huber` and `cauchy` are accepted via
  `FitConfig.robust_loss` (no flag; edit `fit_config.json`).
- **Soft priors are appended as extra residual rows** — MAP, not plain least squares. Each
  width-role parameter contributes `width_prior_strength * ln(value / anchor)`, a log-normal
  prior anchored to the *detected* scale so a component cannot balloon to absorb drift; each
  tail-role parameter contributes `tail_shrinkage * value / anchor`, so a peak stays symmetric
  unless the data earns a tail. `anchor = max(seed_scale, 0.5)`.
- **Bounds come from parameter roles**, not shape names: amplitude in `[0, inf)`, center within
  `max(2*seed_scale, 4)` px of its seed and inside the profile, width in
  `[0.5, max_width_frac*L]`, tail symmetric in `[-max_width, max_width]`, `unit` in `[0, 1]`.
  `at_bound` flags **only** a width pinned at its *upper* bound — the runaway "peak" spanning
  the trace to absorb drift. A center at the profile edge, or a width at the lower bound (a
  genuinely narrow dust spike), is legitimate and is not flagged.
- **Covariance** is `inv(J^T J)` at the solution (pseudo-inverse fallback). The prior rows are
  part of `J`, so it is a MAP/Laplace posterior covariance. `area_uncertainty` is a
  delta-method 1-sigma: the finite-difference gradient of the closed-form `area()` contracted
  with that component's own diagonal block.

`area_uncertainty` ignores cross-component covariance, so two degenerate neighbours can each
report a small uncertainty while their *split* is arbitrary — that surfaces as
`low_identifiability` instead. And `0.0` means a zero-variance, degenerate direction, not
certainty; `null` means no covariance was available.

## The three tiers, and when to climb

`grow_select` and `sparse_seed` propose components differently; `bayesian` re-uses the
`grow_select` proposal verbatim (`_greedy_grow` then `refine`) and samples parameters around
that MAP. All three end in the *same* refinement and assembly, so they cannot disagree about
what `area_fraction` means. Cost is given relative to `grow_select`, because the absolute
number is hardware: measured in-process on lane 3 of the example gel (1364 px, 7 components,
profile cached, 5 repetitions), `grow_select` itself took 1.2 s on an Apple M5.

| tier | cost | when to reach for it |
|---|---|---|
| `grow_select` | 1x | the default; greedy residual growth with an AICc stop |
| `sparse_seed` | 2.9x | heavy overlap, where the residual argmax is ambiguous. Seeds by non-negative least-squares deconvolution against a dictionary of shifted, multi-width Gaussian atoms (240 max), clusters the weight map, hands the seeds to the shared refinement |
| `bayesian` | 1.2x | when you need uncertainty rather than a different decomposition. Random-walk Metropolis around the `grow_select` MAP: 6000 steps, 1500 burn-in, thinned by 5, seeded from `FitConfig.seed` so the tier stays deterministic and cacheable |

`bayesian` is barely dearer than the default *because* it is the default plus sampling — the
6000 likelihood evaluations added 0.27 s. Going the other way, restricting the menu to
`--shapes gaussian` costs 0.3-0.4x, since consolidation retries each candidate merge in every
allowed shape.

The tiers do not agree on that lane, and the disagreement is the useful part. All three return
7 components and their coverages span only 0.008 — but `grow_select` and `bayesian` put 0.592
and 0.571 in the dominant band and pick identical shapes, while `sparse_seed` splits that band
into two near-equal components (0.428 + 0.411). Coverage cannot see the difference: it is an
area budget, not a check on the decomposition. If a dominant band might really be a doublet,
`sparse_seed` is the tier that says so.

`--auto-escalate` climbs `grow_select` -> `sparse_seed` -> `bayesian`
<!-- claim: escalation-ladder-order --> and stops at the first tier whose `confidence` comes
back `"ok"`. The tier it stops on is what gets persisted into `fit_config.json`, so the run
folder records what was actually applied (if no tier reaches `"ok"`, that is `bayesian`). Escalation is driven by `confidence`, never `status`.

```bash
gelpy fit refit-lane "$RUN" --lane 3 --method bayesian --format text
```

The sampling tier has a real limitation. Its proposal is the MAP Laplace covariance scaled by
`(2.4/sqrt(d))^2`, with no adaptation, no acceptance-rate tuning and no convergence
diagnostic. On a clean two-Gaussian synthetic (amplitudes 0.6 and 0.3, sigma 12 and 16 px at
x=120 and x=260 in a 400-px window, noise sd 0.005, area-normalized) that is fine: `d = 6`,
24 % acceptance, 658 of 900 retained samples distinct. On this real 7-component lane, `d = 23`
and the Laplace covariance has condition number 5.4e12 — the acceptance rate is **0.9 %** and
the retained chain holds **17 distinct states out of 900**. The resulting "posterior" spread
came out *smaller* than the Laplace uncertainty it was meant to improve on — band by band, by
factors of 1.4 to 6.3. Treat tier-3 `area_uncertainty` on a dense lane as unvalidated; prefer
it on lanes with few, well-separated components.

## The numbers that leave the engine

Let `y` be the profile actually fitted (sliced, normalized), `b` the baseline, `S = y - b`,
`c_k` the k-th fitted component.

```text
signal_area        A_sig = integral(S)      trapezoid, dx = 1 px, over the whole window
area_k             integral of c_k over the real line (closed form), or, for a
                   region-integrated smear, the trapezoid of the positive local excess
area_fraction_k    area_k / A_sig
coverage           sum_k(area_k) / A_sig
residual_fraction  1 - coverage             exactly; bit-identical
```

`area_fraction` is a component's area over the integral of `(profile - baseline)` — the *true
signal* basis, not the fitted total. <!-- claim: area-fraction-is-on-the-true-signal-basis -->
That is the whole point of the design: **fractions do not renormalize to 1.** They sum to
`coverage` identically (measured difference 0.0 on the example lane), so signal the model
failed to explain stays visible as `residual_fraction` instead of being redistributed across
the bands you report. So `coverage` is the number to look at first, and it is not a residual:

- `coverage < 1` — real signal went unmodelled. Below 0.85 the lane is flagged.
- `coverage > 1` — the components over-explain `A_sig`. Three distinct causes: closed-form
  areas extending past the window edge, overlapping components double-counting the same
  signal, overlapping region-integration windows in the smear fallback. Above 1.10 the lane is
  flagged. All 20 lanes of the example gel land between 0.999 and 1.037.
- `residual_fraction` is an area-budget discrepancy, not a fit residual, and it goes
  **negative** whenever coverage exceeds 1. The fit's residual lives in `quality.r2`,
  `quality.reduced_chi2` and `quality.residual_autocorr`.

One band type breaks the parametric story: a component classified `smear` whose local R^2 is
below 0.75 is measured by direct region integration instead. `params` becomes null, `region`
becomes the `(lo, hi)` support, `area_uncertainty` and `shape_r2` become null, and the `shape`
column in `results.csv` reads `region`. On lane 3 with `--poly-order 0` that happens to band 1,
over `region = (37, 603)`. Two such regions can overlap, and then they double-count.

## Labels are semantic, never subtractive

`component_class` is one of `sample`, `well`, `smear`, `dust`, `unknown` — five values, with
`sample` as the contract's default (documentation elsewhere in this repo listing four values is
wrong). Every component is counted, and the label never removes a band from the total.
<!-- claim: component-classes --> `unknown` is an allowed value that nothing in `src/`
currently assigns, so in practice you see the other four.

The rules, first match wins, on a shape-agnostic width — the sigma of a Gaussian with this
component's area and height, so a smear fitted as a pseudo-Voigt still reads as wide:
maximum at or before `well_frac * L` (default 8 % of the window) -> `well` (position beats
width: a narrow spike in the well region is `well`, not `dust`); else width >= 18 px **or**
>= 3x the lane's median width -> `smear` (the OR matters, since a dominant smear drags the
median with it); else width <= 3.5 px -> `dust`; else `sample`.

The 18 px and 3.5 px bars are absolute because median-relative thresholds fail exactly when
the dust or the smear dominates the lane. Two consequences: the well boundary is a fixed
fraction of the window and is *not* image-derived, so a genuine well aggregate whose maximum
lands past 8 % is labelled like anything else; and since the smear test precedes the dust test
with no width floor, a narrow component in a lane of even narrower ones can come back `smear`.

## `status` is not a quality signal

`status: "ok"` means the optimizer converged and a non-empty decomposition exists — nothing
more. `confidence` (`ok` | `low`) plus machine-readable `confidence_reasons` is the quality
signal, and the field to branch on. <!-- claim: status-ok-is-not-a-quality-signal -->
`status` is one of `ok`, `empty` (no components, or `A_sig <= 3 * noise * sqrt(n)`) or
`failed`; a bad lane always comes back as a result, never as an exception. Nothing in `src/`
currently assigns `"failed"` — a lane whose optimizer gives up still assembles as `"ok"` or
`"empty"`, so `confidence` is the only thing standing between you and a bad number.

Reasons that set `confidence: "low"`:

| trigger | reason string |
|---|---|
| `coverage < 0.85` | `coverage {c}: {rf} of the signal is unmodeled` |
| `coverage > 1.10` | `coverage {c}: fitted peaks exceed the signal (overlap/over-fit)` |
| `r2 < 0.90` | `poor fit (R2=...)` |
| a band's width at its upper bound | `band {n} ({class}) width ran to a bound` |
| a band's `area_uncertainty / area > 1.0` | `band {n} area unreliable (+/-{pct}%)` |
| `status: "empty"` | `no significant signal detected` |

Reasons prefixed `note:` are surfaced but do **not** set `low`: an area uncertainty between
50 % and 100 %, and per-band `low_identifiability` (`note: band {n} overlaps a neighbor (areas
correlated)`). Do not key an agent off "reasons is non-empty" — key it off `confidence` and
read the reasons for the `note:` prefix. Dense real lanes always carry identifiability notes;
all 20 lanes of the example gel do.

`reduced_chi2` and `residual_autocorr` (Durbin-Watson) are computed and reported but are
deliberately **not** triggers, for two different reasons. Durbin-Watson collapses on real
digitized profiles: across the example gel's 20 lanes it spans 0.024 to 0.844 while R^2 spans
0.9974 to 0.9999, so every lane has a strongly structured residual and gating on DW would flag
all of them. `reduced_chi2` is not a gate because its scale is set by the noise estimate, not
by the fit. On every one of those 20 lanes the 0.8 % floor from the seeding section is the
binding estimate, and it beats the MAD estimate by 63x to 90x — so `reduced_chi2`, which scales
as `1/sigma^2`, would come out 4000x to 8000x larger without it. As printed it lands between
0.10 and 1.87, which looks reassuring and means nothing. Diagnosis, not gate.

## Hand-tuning one lane

Durable per-lane intent lives in `fit_config.json` (sparse, precious, off the disposable
manifest). Anything on `FitConfig` is editable there, including the fields with no flag:

```json
{"lanes": {"3": {"fit": {"priors": {"max_components": 3}}}}}
```

`fit refit-lane` merges its overrides onto that stored config, so an edited field takes effect
with no overrides at all (with nothing changed, the command is a cache hit). `fit run`, by
contrast, applies the request's config and overwrites the store for every lane in its
selection — so edit and then refit, never edit and then re-run the pass. With exactly that
edit in place, lane 3 goes from 7 components to 3 (fractions 0.020 / 0.103 / 0.876, coverage
0.998, `confidence: "ok"`): a cap does not redistribute honestly, it lets the survivors absorb
their neighbours, and `coverage` stays near 1 while it happens.

```bash
gelpy fit refit-lane "$RUN" --lane 3 --format text
```

Field definitions are generated, not written here: `gelpy schema FitConfig`,
`gelpy schema BaselineConfig`, `gelpy schema LaneFitResult`, or
[reference/contracts.md](../reference/contracts.md). Registry contents:
`gelpy models list --kind fit` and `--kind shape`, or
[reference/registries.md](../reference/registries.md).
