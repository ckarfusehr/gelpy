---
title: Reading the results
audience: both
summary: What every column of results.csv means, in what units, why area is not mass, and why a lane can vanish from the file after a recipe edit.
subject:
  - src/gelpy/schemas/results.py
  - src/gelpy/runfolder/results.py
subject_digest: e58deb41b4f51f3b
---

## The shape of the file

`results.csv` is one row per fitted band. A lane that produced no bands still gets exactly one
row, with the band columns blank -- lanes are never dropped for being uninteresting -- so the
row count is `sum(bands) + (band-less lanes)`, not the lane count.

`results.csv` and `results.json` are written by `fit run` and `fit refit-lane` only
(`runfolder/commands.py:385`, `:419`), assembled from cached fits with no recomputation. That
is why they can fall behind the recipe; see [Staleness](#staleness-a-lane-that-disappears).

Everything here assumes a run folder and a source image. Set them once; this block is skipped
by the documentation test, which sets the same two variables itself:

```bash
# doc-test: skip
export RUN=/tmp/gelpy-results
export IMAGE=examples/data/folding_screen.tif
```

```bash
gelpy init "$RUN" "$IMAGE" --n-lanes 20 --format text
gelpy lanes detect "$RUN" --n-lanes 20 --apply --format text
gelpy background remove "$RUN" --format text
gelpy fit run "$RUN" --lane 1 --lane 2 --slice 50 600 --format text
```

```bash
head -1 "$RUN/results.csv"
gelpy results show "$RUN" --format table
```

```console
lane_index,label,band_number,component_class,shape,area,area_fraction,area_uncertainty,...
   1  2                 band 0 well     frac=0.158  x=38.8
   1  2                 band 1 well     frac=0.214  x=40.1
   1  2                 band 2 sample   frac=0.046  x=276.2
   ...
```

## The columns

The first twelve columns are fixed and in this order: `lane_index`, `label`, `band_number`,
`component_class`, `shape`, `area`, `area_fraction`, `area_uncertainty`, `maximum_position`,
`residual_fraction`, `confidence`, `status`. <!-- claim: results-csv-columns -->

| column | scope | what it is |
|---|---|---|
| `lane_index` | lane | 0-based index into `recipe.json`'s lane list. |
| `label` | lane | The lane's label from the recipe. `init` defaults to 1-based well numbers, so `label` is usually `lane_index + 1` as a string -- they are not interchangeable. |
| `band_number` | band | 0-based within the lane, assigned *after* sorting by `maximum_position`, so band 0 is always the one nearest the start of the fitted window. |
| `component_class` | band | `sample`, `well`, `smear`, `dust` or `unknown`. Semantic only -- see below. |
| `shape` | band | `gaussian`, `emg`, `bi_gaussian`, `pseudo_voigt`, or `region` for a `smear` component that fitted its own region badly (local R^2 < 0.75) and was integrated numerically instead. |
| `area` | band | The component's closed-form integral, over the whole real line -- not over the fitted window. Units below, and see coverage. |
| `area_fraction` | band | `area` divided by that lane's total baseline-subtracted signal area. |
| `area_uncertainty` | band | 1-sigma on `area`. Blank when unavailable. |
| `maximum_position` | band | Position of the component maximum, in pixels along the fitted profile -- **relative to `slice_start`**, not to the image. |
| `residual_fraction` | lane | `1 - coverage`, repeated on every row of the lane. |
| `confidence` | lane | `ok` or `low`. This, not `status`, is the quality signal. |
| `status` | lane | `ok` or `empty`. |

After those twelve comes `method` (the strategy that produced the lane's fit), then one
`param_<name>` column per fitted shape parameter, then `error`. That tail is **not stable**:

- `param_*` columns are added in first-seen order across the rows in the file
  (`runfolder/results.py:68-72`), so the set and the order follow whichever shapes the fit
  chose. The run above -- 15 rows over two lanes -- gives a 20-column header: the fixed twelve,
  `method`, and seven `param_*` whose order (`param_sigma_left` before `param_sigma`) is an
  artifact of lane 1's first band being a bi-Gaussian.
- `error` appears only once some lane has emitted a band-less row.

So read columns by name; positional indexing past column twelve breaks on the next fit.

## area is not mass

`area` is the closed-form integral of the fitted shape over the whole real line --
`amplitude * sigma * sqrt(2*pi)` for a Gaussian -- in units of *profile intensity x pixels*,
since every width parameter is in pixels. The denominator `signal_area` is a trapezoid sum of
`profile - baseline` over the fitted window only, with `dx = 1` row. The two domains differ,
which is why coverage can exceed 1 (below).

Those units depend on how the profile was prepared. It is a column-mean of the canonical float
image, then sliced, then normalized, in that order (`runfolder/pipeline.py:180-183`). With the
default `--normalization area` the sliced profile is divided by its own integral before
fitting, so it integrates to exactly 1 and `area` is a dimensionless share of the lane's window
integral, baseline included. With `--normalization safe` the same lane's areas come out 2.27x
larger (measured on lane 1 of the example gel: band 0 goes from 0.1275 to 0.2889, and
`signal_area` from 0.807 to 1.828) because the scaling is gone. Either way the number is an
*optical* quantity.

Nothing in gelpy converts that to mass or molarity: there is no ladder handling, no standard
curve, no calibration of any kind in `src/gelpy` -- MW/ladder calibration is a deferred
feature, and the ladder reference images were deleted in 1.0. Mass needs your own dilution
series measured through the same pipeline, and a detector you believe was linear over the
range you used.

`area_fraction = area / signal_area` is the number to compare across lanes -- across lanes
fitted over the *same* slice window. The denominator is per-lane
(`core/peaks/strategies/_common.py:87`, `:138`), which makes the fractions robust to
lane-to-lane loading and illumination differences and invariant to the normalization choice --
numerator and denominator scale together, so lane 1's six fractions are identical to six
decimals under `area` and under `safe`. Absolute `area` is neither. But that denominator is the
integral over *that lane's* window, so fractions from different windows are not on the same
basis. The worked run on this page ends with lanes 1 and 2 fitted over rows 50-600, lane 0 over
900-1300 and lane 19 over 1250-1360, all in one `results.csv`; those fractions do not share a
denominator and comparing them across lanes means nothing.

Summing `area_fraction` over a lane gives that lane's coverage, i.e. `1 - residual_fraction`.
Measured on the two lanes fitted above: 1.032 and 1.024. **Coverage above 1 is real and
`residual_fraction` goes negative** (-0.032 and -0.024 here). Overlap is not the mechanism --
component integrals are additive, so the components always sum to the composite over the window
whatever they do to each other. Numerator and denominator are integrated over *different
domains*: `area` is the closed-form integral over the whole real line, `signal_area` a
trapezoid over the fitted window. Every tail that leaves the window is counted upstairs and not
downstairs.

Lane 1, measured: the six closed-form areas sum to 0.8327; the same six components integrated
over the 550-row window sum to 0.8049, which equals their composite; `signal_area` is 0.8070.
The 0.0278 outside the window more than accounts for the 0.0257 excess -- in-window the fit is
0.0021 *low*. Two components do it: a `gaussian` pinned at the last row (half its area past the
edge) and a `pseudo_voigt` whose Lorentzian wings never terminate. Lane 2 is the same picture
(0.7889 / 0.7713 / 0.7707). Signal dipping below the fitted baseline shrinks the denominator
too, since no component can go negative, but on these lanes that is worth -0.0002.

Coverage far from 1 in either direction is exactly what the `confidence` flag is watching.

## area_uncertainty

1-sigma on the area, propagated by the delta method: the gradient of the shape's closed-form
area with respect to its parameters, numerically differentiated, pushed through that
component's block of the fit covariance (`core/peaks/strategies/_common.py:31-45`). The
`bayesian` tier substitutes the posterior covariance from its sampler, which makes the number
a credible interval rather than a local linearization.

It is blank when there is no covariance to propagate: the least-squares call raised (the driver
then keeps the seed parameters and sets no covariance), both the normal-matrix inverse and its
pseudo-inverse failed, or the band is a `shape: region` component -- region integration has no
parameters to differentiate. It is `0.0` when the propagated variance comes out non-positive.

Treat it as a *within-model* uncertainty. It answers "given this set of components and shapes,
how well is this one's area pinned down", not "is this the right set of components". Band 0 of
lane 1 above has `area = 0.128 +/- 0.0009` -- 0.7% -- while the lane's own
`confidence_reasons` say that band overlaps its neighbour and their areas are correlated. The
covariance is small; the model choice is the risk.

## component_class never removes a band

Classification is a label, not a filter (`schemas/peaks.py:88-90`). A `well` or `smear`
component counts toward coverage and carries its own `area_fraction` exactly like a `sample`
band. `signal_area`, the denominator they all share, is measured off the profile before any
component is assembled (`core/peaks/strategies/_common.py:87` vs `:133-148`), so it does not
know about classes at all. On lane 1 of the example gel the two `well` components carry 37% of
the lane's signal, the two `smear` components 47%, and the two `sample` bands 19% -- summing to
that lane's 103% coverage, not to 100%.
Purity relative to sample signal only is yours to compute from the class column: which
components belong in the denominator is a question about your experiment, not about the fit.

## confidence, not status

`status` says whether the fit ran. `confidence` says whether to trust it. They disagree often,
and the honest column is `confidence`.

```bash
gelpy fit run "$RUN" --lane 0 --slice 900 1300 --format text
gelpy fit run "$RUN" --lane 19 --slice 1250 1360 --format text
head -1 "$RUN/results.csv"
gelpy results show "$RUN" --format table
```

Two deliberately bad windows. Lane 0 over rows 900-1300 returns `status: ok`,
`confidence: low`, coverage 153%. The fit is not what is wrong. Rows 900-1300 hold the lane's
decaying trailing smear -- the windowed profile falls from 0.0098 at the top to a mean of 2e-5
over the last 50 rows -- and the two components reproduce it to local R^2 0.99996 and 0.99821,
their composite integrating to 0.7038 against `signal_area` 0.7032. Neither is flagged
`low_identifiability`. The 153% is window truncation: the broad `bi_gaussian` sits at x=13.9
with `sigma_left` 49.2, so 40% of its closed-form area (0.9103) lies at x<0, above the top of
the window and outside the denominator. That is what a coverage flag looks like when the
*window*, not the fit, is wrong: fit the same lane over rows 600-1300 instead and it returns
coverage 0.999 with `confidence: ok`.

Lane 19 over rows 1250-1360 returns `status: empty`: its signal area
fell below `3 * noise_sigma * sqrt(n_samples)` (`core/peaks/strategies/_common.py:24`,
`:92`), so no components were kept. Its single row carries `residual_fraction = 1.0` and blank
band columns, and its arrival adds the `error` column (20 header columns before, 21 after).

The reasons behind a `low` are not in the CSV. They are in `results.json`, per lane, as
`confidence_reasons`. See [fit-engine.md](fit-engine.md) for what triggers each one.

Two caveats about `status`:

- The contract allows `ok`, `empty` and `failed`, but **nothing in the current code path ever
  writes `failed`** -- no assignment of that value exists outside the tests, so `error` is
  always blank and `gelpy status`'s `n_failed` is always 0. A fit that goes wrong shows up as
  `empty`, or as `ok` with `confidence: low`.
- If `least_squares` raises, the driver keeps the seed parameters, records no covariance and
  carries on (`core/peaks/driver.py:238-241`); if it merely fails to converge, the `success`
  flag it sets is read by nothing. So non-convergence is never reported as such. Its symptoms
  are: a blank `area_uncertainty` for every band in the lane (the raise case), or coverage and
  R^2 off enough to trip `confidence: low`.

## maximum_position is slice-relative

`maximum_position` is in the coordinates of the profile that was fitted, so with
`--slice 900 1300` a maximum at image row 913.9 is reported as 13.9. The per-lane fit PNG uses
the same axis, so figure and CSV agree, but positions are comparable across runs only if the
window matched. Add that lane's `slice_start` (in `fit_config.json`) to get an image row.

## results.csv vs results.json vs report.md

Same fits, three different amounts of truth.

- **`results.csv`** -- the flattened table above. It drops per-band `centroid`, `region`,
  `shape_r2`, `at_bound`, `low_identifiability` and per-lane `baseline`, `signal_area`,
  `coverage`, `quality`, `confidence_reasons`. Written with a plain `open()`, not atomically
  (`runfolder/results.py:74`), so an interrupted `fit run` can truncate it; re-run the fit and
  it is rewritten from cache.
- **`results.json`** -- the serialized `AnalysisResult`: every field of every `LaneFitResult`,
  including those ten, so `quality` (`reduced_chi2`, `r2`, `residual_autocorr`, `noise_sigma`,
  `n_components`, `aicc`, `bic`) is here and nowhere else, and `confidence_reasons` are here in
  full -- the fit command prints them on its own stdout summary as `reasons`, and `report.md`
  shows the first two. Written atomically. Parse this one if you branch on quality.
- **`report.md`** -- an eight-column human table (`lane_index`, `label`, `band`, `class`,
  `area_fraction`, `x`, `confidence`, `status`) plus a summary line per lane with coverage and,
  when confidence is `low`, the first two reasons. Regenerated after each fit command. Cosmetic
  wart: an `empty` lane renders as `fit empty — None`, there being no error string to print.

## Staleness: a lane that disappears

Results are assembled from whatever the manifest's fit records point at, with two filters
(`runfolder/results.py:18-33`): a lane no longer in the recipe is dropped, and a cached fit
whose `fit_hash` no longer equals the hash recomputed from the current recipe plus that lane's
stored `(profile, fit)` config is dropped as stale. A lane with a cached fit but no entry in
`fit_config.json` is kept, best effort, because there is nothing to validate it against.

So a recipe edit can silently empty your results. `gelpy status` is where you see it:

```bash
gelpy lanes set "$RUN" --width 200 --format text
gelpy status "$RUN" --format json | tr ',' '\n' | grep -E '"n_(lanes|fit|stale|failed)"'
gelpy results show "$RUN" --format table
wc -l < "$RUN/results.csv"
```

```console
  "n_lanes": 20
  "n_fit": 0
  "n_stale": 4
  "n_failed": 0
```

Changing the strip width changed every lane's `profile_hash`, hence every `fit_hash`: all four
fits are stale, each lane reports `has_fit: true, fit_current: false`, and
`results show --format table` prints nothing.

Note what the last two commands show together. All three formats of `results show` assemble
from the cache and apply the staleness filter, so they agree with each other -- including
`--format csv`, which renders the live assembly rather than reading the file back.

The **file** `results.csv` is a different thing: it is written by a fit pass, so after an edit
that stales a lane it still holds the rows that pass produced. If `n_stale > 0`, the file on
disk is not describing your current recipe even though `results show --format csv` is. Re-run
a fit command to rewrite it. (Until 1.0 the CSV format *did* read the file back, so one command
could print two disagreeing answers.)

Two honest fixes:

1. **Undo the edit.** Hashes are content-addressed, so putting the width back makes the old
   `fit_hash` current again and the cached fits reappear with no recomputation:

```bash
gelpy lanes set "$RUN" --width 159 --format text
gelpy status "$RUN" --format json | tr ',' '\n' | grep -E '"n_(fit|stale)"'
```

2. **Re-fit under the new recipe.** Prefer `gelpy fit refit-lane "$RUN" --lane N` with no
   further flags: it merges (empty) overrides onto that lane's stored `(profile, fit)` config,
   so the lane is refitted with the settings it had. `gelpy fit run` does *not* merge -- every
   flag you omit falls back to its default, so re-running `fit run --lane 1` after a pass that
   used `--slice 50 600` fits a different window and returns different numbers.

One more way to lose results: `manifest.json` is the ledger of which cache files exist, so
deleting it makes every fit invisible (`n_fit: 0`, `n_stale: 0`) even though the fit JSONs are
still under `cache/fits/`. The next fit command re-records only the lanes it names -- it finds
their caches, so recovery costs nothing, but lanes you do not name stay invisible. Measured on
a run with four cached fits and no manifest: `fit run --lane 1 --slice 50 600` reports
`recomputed: []`, `reused: [background, fit:lane=1, image, profile:lane=1]`, and takes `n_fit`
from 0 to 1. Name every lane you want back.

## Column names changed in 1.0

Every 0.x column name except `label` and `band_number` was renamed;
[migrating-from-0.x.md](migrating-from-0.x.md) has the full mapping. Only one of those changes
is dangerous, because only one is silent. A rename fails loudly: pandas raises `KeyError` on
the first missing column and you go fix it.

`relative_area` -> `area_fraction` also changed *basis*. 0.x divided each component's area by
the sum of the fitted component areas, so `relative_area` summed to exactly 1 per lane by
construction -- a fit that missed half the signal still reported fractions summing to 1.
`area_fraction` divides by the measured baseline-subtracted signal instead, which does not sum
to 1, and whose departure from 1 is the report (see coverage above). Rename the column and your
script runs; every number in it has moved.

## Into Python

`gelpy schema` and the [generated contracts reference](../reference/contracts.md) describe the
objects; the bridge is `AnalysisResult.to_dataframe()` (pandas is imported lazily, so it is not
a hard dependency of the package):

```python
from gelpy.runfolder.commands import get_results
from gelpy.schemas import GetResultsRequest

result = get_results(GetResultsRequest(run_dir="/path/to/run"))   # no recompute
df = result.to_dataframe()

sample = df[df.component_class == "sample"]
print(sample.groupby("label")["area_fraction"].sum())
```

`get_results` applies the same staleness filter as the CSV, so a stale lane is absent here too.
Two things about the frame: its column *order* is the row-construction order rather than the
CSV's fixed order (`status` and `method` come early), and every band-less lane contributes a
row of NaNs in the band columns -- filter on `status` or `band_number.notna()` before
aggregating.
