---
title: Background removal
audience: both
summary: Why the 2-D background stage is mandatory before any profile or fit, how the robust_plane IRLS surface finds the background without user input, and where it breaks.
subject:
  - src/gelpy/core/background/base.py
  - src/gelpy/core/background/robust_plane.py
  - src/gelpy/core/background/plane_ransac.py
  - src/gelpy/schemas/background.py
subject_digest: 27351e9e6d05a169
---

## Why this stage is not optional

A gel image is `measured = background + signal`, and the background is additive: camera
offset, uneven illumination, gel autofluorescence, stain haze. On
`examples/data/folding_screen.tif` (1364x4688, canonicalized to [0,1]) the fitted surface
runs 0.00168 to 0.00230 — an offset of ~7 noise sigma with a 2.2-sigma gradient across the
image. Measured over the 20 centres `gelpy lanes detect` finds, in 93-px strips (the width
`init` writes), **14.7 % to 21.0 % of each raw lane integral is background, and the share
differs per lane**. It tracks how much was loaded — Spearman rho = -0.64, p = 0.002, n = 20,
against the corrected lane integral — so the faintest lane carries 21.0 % and the strongest
16.2 %. Skip the stage and that goes straight into every band area you report.

The second failure is worse, because it is invisible. Profiles are area-normalized by
default (`LineProfileConfig.normalization = "area"`): each is divided by its own trapezoid
integral. With an un-removed floor `b` over `N` rows and lane signal `S`, the normalized
floor is

```text
b / (b*N + S)      approximately
```

which depends on `S`. A strong lane's floor is crushed toward 0; a faint lane's inflates
toward `1/N`. Same physical background, different plotted baseline.

Measured on the same 20 lanes, taking each lane's floor as the median of its lowest-fifth
profile values and its strength as its corrected integral: the raw floors span 0.00152 to
0.00201 and are uncorrelated with strength (Spearman rho = +0.05, p = 0.85, n = 20) — as
they should be, since the background does not know what was loaded. Area-normalize the same
profiles and the floors span 9.1e-5 to 1.32e-4 (a factor 1.46) and become **anti-correlated
with strength: rho = -0.56, p = 0.010**. That correlation is manufactured entirely by the
normalization. Subtract the surface first, area-normalize the corrected profiles the same
way, and all 20 floors land between -2.0e-5 and -4.5e-6, i.e. on one common near-zero
baseline. Any cross-lane comparison — purity, ratios, "is this band bigger" — is meaningless
before this stage.

Background removal is **off in a fresh run folder**: `gelpy init` writes
`recipe.json` with `background: null`, which the pipeline treats as passthrough (no cache
entry is created or consulted). You must configure it, and gelpy checks: `profiles extract`,
`fit run` and `fit refit-lane` refuse a run without it (`background_required`) unless you pass
`--allow-raw` to measure the raw image on purpose. The default model is `robust_plane`;
`plane_ransac` is a fallback only. <!-- claim: default-background-model -->

```bash
# doc-test: skip
# The doc test sets these two for you; set them yourself if you are following along.
export IMAGE=examples/data/folding_screen.tif
export RUN=/tmp/gelpy-background
```

```bash
gelpy init "$RUN" "$IMAGE" --n-lanes 20 --format text
gelpy background remove "$RUN" --format text
```

`background remove` computes nothing. It validates the model name against the registry and
the parameters against that model's schema, then stores the **canonical, fully-defaulted**
parameter dict on the recipe. The fit happens lazily, the first time something needs the
corrected image:

```bash
gelpy profiles extract "$RUN" --lane 0 --format text
```

```text
{"recomputed": ["background", "image", "profile:lane=0"], "reused": [], ...}
```

## What `robust_plane` fits, and why it lands on the background

The premise is physical: after canonicalization the image is signal-bright (see
[image-input.md](image-input.md)), so **sample signal only ever adds intensity**. Whatever
the surface fit gets wrong, the background is therefore the *low* side of the residual. That
is the whole trick — no stripes, no lane knowledge, no masks.

The surface is a low-order polynomial in coordinates normalized to [-1, 1] on each axis
(order 0 = offset, 1 = plane, 2 = quadratic; `c0` is the value at the image centre). The
loop is IRLS: alternate a weighted least-squares solve with a re-estimate of the per-pixel
trust.

```text
w <- 1                                      first M-step is plain OLS
repeat, at most max_iter times:
  theta  <- solve (AᵀWA) theta = AᵀW z      k x k normal equations, k in {1,3,6}
  stop if max|Δtheta| / max|theta| < tol
  r      <- z - A theta
  sigma  <- 1.4826 * median(|r[r<0]|)       noise scale from the background side only
  w      <- weight(r, sigma)
```

Three things make this robust where a plain fit is not:

- **The noise scale is re-estimated every iteration from the negative residuals alone.**
  `r < 0` cannot be signal, so it stays a clean noise sample at the signal coverage a real
  gel has (see the envelope below). `1.4826 * median(|r_neg|)` is the MAD-to-sigma constant
  applied about zero, not about the negative side's own median; on 200k samples of N(0, 3) it
  returns 2.99-3.01 (spread over eight RNG seeds). If fewer than 32 negative residuals exist
  (a surface still biased low early on) it falls back to the global MAD about the median.
- **`probabilistic` weighting (the default) is a mixture posterior**, not a threshold:
  background ~ N(0, sigma^2), signal ~ one-sided exponential with mean sigma, so
  `w = P(background | r)`. Below the surface the signal density is zero and `w` is exactly 1
  — the background side is fully trusted. Both densities are written in units of
  `z = r/sigma`, so the Jacobian cancels and **the weight depends only on `r/sigma` and the
  prior**: at `signal_prior = 0.3`, w(z=1) = 0.606, w(z=3) = 0.172, w < 0.01 past z = 4.2,
  and those values are identical to 10 digits for sigma spanning 1e-4 to 1e4. That
  scale-freeness is what lets one default work on a [0,1] image with a 3e-4 noise floor.
- **`asymmetric` weighting** is the cheaper Eilers ALS alternative: a fixed `asymmetry_p`
  above the surface, 1 below. The weight function itself needs no sigma, but `fit_surface`
  calls `robust_sigma` every iteration regardless, so the saving is only the shorter run: on
  this gel 6 iterations instead of 13, for a 78 % larger residual sigma (4.91e-4 vs 2.75e-4).

**Convergence is tested on the coefficients, not the weights**: stop when
`max|Δtheta| / max|theta| < tol` (default 1e-3), capped at `max_iter` (default 15). The soft
probabilistic weights keep jittering after the surface has settled, so a weight-based
criterion would never trip. Caveat: the `tol` field description printed by
`gelpy models list --kind background` still says "max weight change between iterations" —
that string is stale, the code does what is written above. There is no converged flag; the
only signal is `n_iter == max_iter`, and nothing persists `n_iter` (see limitations).

Cost on this gel with the shipped defaults: 13 iterations over 6.4 Mpx, a few seconds of
wall time for the surface fit itself. Absolute seconds are machine-specific, so the escape
hatches below are quoted as ratios. No RNG anywhere, so the result is bit-reproducible —
which is what lets the content-hash cache reuse it.

## Where it holds, and where it stops holding

The estimator is unbiased while observable background remains between the bands. Envelope
reproduced on a synthetic 400x200 gel (gentle true plane, noise sigma 0.5, band amplitude
100, band sd 6 px), sweeping band count:

| signal coverage | surface RMS error | reported sigma (true 0.5) |
|---|---|---|
| 0.35 | 0.06 | 0.45 |
| 0.70 | 0.33 | 0.61 |
| 0.89 | 37 - 158 | 13 - 231 |

Comfortable to ~2/3 coverage; hard failure past ~0.9, where the inter-lane gutters have
closed and there is no background to see. Real gels sit in the safe part of that range: on
`folding_screen.tif` the mean final weight is 0.40, 22.0 % of all corrected pixels are
negative, and among the background-only pixels (corrected < 3 sigma, 55 % of the frame)
40.0 % are negative while their mean sits +0.36 sigma above zero: the surface is inside the
noise, biased low by about a third of a sigma, not eating signal. (That `< 3 sigma` cut lops
off the positive tail, so the real skew is stronger than 40/60.) The tell-tale of the
failure is a reported sigma inflated 26-460x above the true noise — which you cannot read
out of the run folder today, so the weight map in `background show` is the diagnostic you
actually have.

If you hit that regime, raise `signal_prior` toward the true coverage: on the 24-band case
above (coverage 0.89), `signal_prior=0.9` takes the surface RMS from 79.5 back to 0.78. A
crop with a lane-free margin helps more.

## Signed residuals: why `clip_negative` is off by default

After a correct subtraction, background pixels are noise straddling zero. Clamping the
negative half rectifies that noise: the clamped variable has expectation
`E[max(x,0)] = sigma/sqrt(2*pi) = 0.399*sigma` instead of 0 — a positive pedestal, largest
exactly where the corrected image is near zero, i.e. in the regions that are supposed to
contribute nothing.

Measured on this gel over those same background-only pixels, clamping adds +0.37 sigma to
their mean — within 8 % of the 0.399-sigma prediction. It does not average away, because
clipping happens per pixel *before* the lane strip is averaged into a profile, while the
noise it rectifies does. Consequence for the numbers you report: over the same 20 lanes,
clamping inflates each lane's integral by 0.34-0.78 % (median 0.57 %), by a different amount
per lane — so it does not cancel in a ratio, and it destroys the only evidence that the
surface over-subtracted.

So the cache keeps the signed array and only the display path clamps:

```bash
python - <<'PY'
import glob, json, os
import numpy as np
run = os.environ["RUN"]
print(json.load(open(f"{run}/recipe.json"))["background"])
corrected = np.load(glob.glob(f"{run}/cache/background/*.npz")[0])["corrected"]
print(f"{corrected.dtype}  min={corrected.min():+.4f}  negative={100 * (corrected < 0).mean():.1f}%")
PY
```

```text
{'model': 'robust_plane', 'params': {'order': 1, 'weight_mode': 'probabilistic', ...}}
float64  min=-0.0020  negative=22.0%
```

`gelpy image show --kind corrected` clamps at zero for rendering only (clamping, not
shifting by the minimum, so a band at 0.5 still renders at 0.5). `--clip-negative` puts the
clamp into the analysis path itself; use it only if something downstream of gelpy cannot
take negatives, and expect the bias above. `plane_ransac` clips unconditionally, with no
flag.

## Reading `background show`

```bash
gelpy background show "$RUN" --format text
```

For a fitted-surface model you get three image panels — image, fitted surface, per-pixel
weight map (magma, 0 to 1, with a colorbar) — over three column-mean profile panels
(original dashed, background solid grey, corrected green) taken across thirds of the image
width, with a zero reference line. What convinces me the fit is good:

- **The weight map looks like an inverted picture of the sample.** Wells, bands and smears
  dark (the fit ignored them), gutters and margins bright. If a band is bright in the weight
  map the fit is pulling that band into the background.
- **The corrected green trace returns to the zero line between bands** instead of running
  parallel to it. A trace held above zero across the whole lane means under-subtraction; one
  sitting clearly below means the surface ate signal. Small excursions below zero are the
  point of keeping the residual signed — they are the noise you are allowed to see.
- **The surface panel is autoscaled**, so it always looks like a dramatic gradient. Do not
  read intensity from it, read shape: it must be smooth and featureless. Any imprint of the
  bands in it is a failed fit.

Per-figure detail lives in [reading-the-figures.md](reading-the-figures.md). Note that
`background show` re-runs the whole IRLS fit every time before drawing anything (the
diagnostics are not cached), so it costs seconds even on a warm run folder.

## Escape hatches, in the order I reach for them

Parameters and their ranges: `gelpy models list --kind background`. Set any of them with
`--param key=value`; `--order` and `--mode` are shorthands. What the numbers say:

- **`--param subsample=4`** first, on anything large. The fit uses every 4th row and column
  (1/16 of the pixels), the surface is still evaluated on the full grid: ~10x faster here
  (same 13 iterations), the two surfaces differing by an RMS of 9e-7 (0.003 sigma). Not
  bit-identical, so the field description's "lossless" is too strong — but far below the
  noise floor. It also saves memory: the surface is evaluated from two 1-D coordinate vectors,
  so a subsampled fit builds no full-resolution coordinate grids and no design matrix. Peak
  allocation of the fit on this gel (`tracemalloc` around the model's `remove`, a 51 MB image):
  516 MB at `subsample=4` against 900 MB for the default fit.
- **`--order 2`** when the illumination is visibly curved (vignetting). Here it cuts the
  residual sigma 36 % (2.75e-4 -> 1.75e-4) but runs to the 15-iteration cap without
  converging. Opt-in, because a quadratic can also bend into diffuse haze, which is signal.
- **`--mode asymmetric`** for a cheap second opinion. Keep `asymmetry_p` small; as it
  approaches 1 the estimator degenerates to plain OLS (`asymmetry_p=0.99` lands within 0.8 %
  of the unweighted OLS coefficients here, on all three). `signal_prior = 0` is permitted by
  the schema but does *not* do the same thing: the Gaussian density underflows the `1e-12`
  guard in the denominator past z ~ 7, so above-surface pixels are still driven to weight 0
  and band pixels (z ~ 2400 here) are still rejected outright.
- **`contiguity`** blurs the weight field so isolated hot pixels are not trusted as
  background. On a band-dense gel it is a footgun, not a refinement: adding `contiguity=8`
  to a `subsample=4` fit of this gel grows the residual sigma 43 % (2.75e-4 -> 3.93e-4),
  because the blur radius is comparable to a band and leaks trust from the gutters into the
  bands. Use it only if your actual failure is isolated-pixel noise.

## `plane_ransac`, and when it is the right answer

The legacy model samples two horizontal stripes and RANSAC-fits `z = a*x + b*y + c` through
them, in raw pixel coordinates. It is the fallback for exactly one situation: **you know
where the background is and the robust fit disagrees with you** — a gel with no gutters left,
or an artefact the mixture keeps trusting. `--stripe pos,height` is repeatable and the rows
are image-specific; on this gel rows 20-29 and 1340-1359 are above the wells and below the
last band:

```bash
gelpy background remove "$RUN" --model plane_ransac --stripe 20,10 --stripe 1340,20 --format text
gelpy background show "$RUN" --format text
```

Its diagnostic is a different figure: stripes in red over the image plus the three slice
panels — no surface panel, no weight map, and the corrected trace is clipped for display, so
it cannot show over-subtraction.

Why it is not the default: those stripes are the fit's entire evidence. Omitting `--stripe`
falls back to the heuristic `[(20, 10), (height-15, 10)]` — here rows 20-29 and 1349-1358,
1.47 % of the pixels, two thin bands ~1330 rows apart. Whether they contain background
depends on how the image was cropped, and a band, ladder or halo falling in them
contaminates the sample with no complaint. It also cannot fit anything but a plane, and it
clips negatives unconditionally.

Model parameters do not carry over silently. Omitting `--model` below re-selects the default
`robust_plane`, which has no `stripes` field, and every parameter schema is
`extra="forbid"`:

```bash
# doc-test: expect-failure
gelpy background remove "$RUN" --stripe 10,5 --format text
```

```text
{
  "error": "RobustPlaneParams: stripes: Extra inputs are not permitted. `gelpy schema RobustPlaneParams` prints the contract.",
  "code": "validation_error",
  "detail": "[{\"type\": \"extra_forbidden\", \"loc\": [\"stripes\"], \"msg\": \"Extra inputs are not permitted\", \"input\": [[10, 5]]}]",
  "available": null
}
```

## Known limitations

- **No fit diagnostics are persisted.** The model computes the coefficients, `sigma`,
  `n_iter` and a mean weight, and the pipeline keeps only the corrected array: the cache
  `.npz` holds exactly one key, `corrected`. `gelpy status` reports the model name and
  nothing else. So you cannot tell from a run folder whether the fit converged, and a
  reviewer cannot audit the background numbers. `background show` gives you the weight map
  and the corrected slices; for the numbers, call `fit_surface` from
  `core.background.robust_plane` yourself.
- **`inlier_fraction` means two different things.** For `plane_ransac` it is the fraction of
  stripe samples RANSAC kept as inliers; for `robust_plane` it is the mean of the final soft
  weight map (0.40 here), which is not a count of anything. Neither reaches the run folder,
  but both arrive under the same field name.
- **A noiseless image collapses the fit to a zero surface, silently.** On a constant 0.5
  image the sigma fallback returns 1e-12, every weight underflows to 0, and the next solve
  returns all-zero coefficients: the surface is identically 0, `corrected == image` bit for
  bit, no error. Tell-tales are `sigma == 1e-12` and `inlier_fraction == 0`. This bites
  synthetic fixtures, not real gels; `plane_ransac` handles the same input correctly. Same
  root cause as the high-coverage failure: with no negative side left, the sigma estimator
  measures signal spread instead of noise (~1.78 for |N(0,3)| where sigma = 3).
- **The surface is not always a plane.** `BackgroundFitResult.background_plane` is a
  quadratic at order 2 and a constant at order 0; the field name is legacy.
- **Whether a fit-time 1-D baseline is still needed after this stage is an open question.**
  The peak engine re-estimates a per-lane baseline (`poly`, order 2 by default) on the
  already-corrected profile. Where `robust_plane` did its job that baseline has little
  legitimate structure left to fit, and its remaining freedom can only absorb real signal
  (`BaselineConfig`'s own measurement: an order-2 polynomial absorbs 5.5 % of a broad smear
  on a flat residual background, order 3 absorbs 13 %). Recorded and unresolved in the
  historical design note `2026-07-15-baseline-redundancy-and-well-boundary.md` under the
  design records in `docs/`; until it is settled, consider a constant baseline on affected
  lanes. See [fit-engine.md](fit-engine.md) for that stage.
