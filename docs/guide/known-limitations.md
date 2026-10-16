---
title: Known limitations
audience: both
summary: What gelpy is measurably bad at, with the number for each limitation and how that number was obtained.
subject:
  - src/gelpy/core/lanes.py
  - src/gelpy/core/peaks/baseline.py
  - src/gelpy/core/peaks/classify.py
subject_digest: 2d8a9e9e6f2abdb2
---

Read this before publishing a number gelpy produced.

Every figure here was measured, and the command that reproduces it is given. Where a figure
recorded elsewhere in this repo does not reproduce, this page states what I measured instead and
says so. Two in-repo harnesses generate most of them: `tests/synthetic/harness.py` for
**detection** (25 gel conditions x 8 seeds = 200 runs against the generator's own lane centers,
plus two real gels with reference centers in `real.py` — cy3's checked against the well grid,
`example_gel_1`'s an assumed even linspace; see the caveat below), and
`tests/synthetic/fit_harness.py` for **fitting** (16 profile conditions x 3 repeats = 48 runs,
scored by area-fraction error against known component areas).

```bash
# doc-test: skip
export IMAGE=examples/data/example_gel_1.tif
export RUN=./run-limits
```

## Lane-grid pitch is good to ~0.1%, not to a pixel

Detection anchors an even N-lane grid to the column projection. The grid comes back *exactly*
even — the failure mode is a small error in its **pitch**, and a scale error accumulates
linearly across the gel. Phase is sub-pixel. Against the reference centers, from the repo root:

```bash
python -m tests.synthetic.real
```

```console
example_gel_1: shape=(1364, 4688) N=20 pitch=228.2
  baseline: {... 'max_err_px': 4.4386, 'max_err_xpitch': 0.01945, 'median_err_px': 1.8719, ...}
cy3: ... 'max_err_px': 0.2857, 'max_err_xpitch': 0.00162, ... 'confidence': 'ok'
```

The `baseline:` row there is the pinned `full` reference; the shipped default was measured
separately with the same scorer.

- `example_gel_1`: worst lane off by **4.44 px = 0.019x pitch**, median 1.87 px, `confidence:
  ok`. Identical for the shipped default strategy and for `--projection-strategy full`. The
  signed per-lane error is a *pure ramp*: slope +0.270 px/lane, intercept -0.695 px, residual
  off that line **0.0000 px** (max and rms). Detected pitch 228.428 px against 228.158 px in
  the labels, **+0.118%**; over 19 gaps that is 5.13 px — the entire worst-lane error. Phase
  contributes -0.70 px.
- `cy3`: 0.29 px pinned to `full`, 1.29 px (0.0073x pitch) with the default. Same shape —
  residual off the linear term 0.0000 px, intercept +0.29 px (default) and -0.14 px (`full`).
- For the shipped default, the worst error outside `partial_fill` (below) is **0.097x pitch**
  (`dusty_in_band`, seed 7013 — dust confined to the band's y-range); per-condition medians run
  0.009x to 0.049x. Aggregate: median 0.02x pitch, 96% pass, **0% false-ok**. The 4% that fail
  are exactly the eight `partial_fill` seeds, all flagged `low`. On this suite detection never
  returns a wrong grid while claiming it is fine. The pinned `full` reference is worse — 0.449x
  worst outside `partial_fill` (`snr_015`, seed 4030), 12 of 200 failing across four conditions
  — but also 0% false-ok, every failure flagged `low`.

Why it usually does not matter: a detected layout's profile strip is `0.7 * pitch` wide
(`_PROFILE_WIDTH_FRACTION`, `src/gelpy/core/lanes.py:17`) — 159 px here. A 4.44 px displacement
is 2.8% of the strip width, and the strip is a column *mean*, so the effect on the profile is
second order. The suite's pass threshold is 0.25x pitch for the same reason: past a quarter
pitch the strip starts sampling the neighbour.

Caveat on the ground truth itself: `example_gel_1`'s labels are an assumed even grid
(`centers_linspace: [150, 4485]`), so this is two even grids disagreeing about pitch by 0.118%,
and which one is right is undecidable from the fixture — part of the 4.44 px is label error and
the gel's real comb may not be even. An earlier version of that note claimed the grid matched
full-height detection to <1 px; it does not, and the note now says so. Either way: never report
a detected center as a coordinate. Use it to place a strip and verify the overlay —
[lane-placement.md](lane-placement.md).

## A gel filling only part of the crop defeats the width/N prior

Detection's expected pitch is `image_width / N`. If the gel occupies only a fraction of the
crop, that prior is wrong by exactly that fraction and the grid lands nowhere near the lanes.

The `partial_fill` condition (`tests/synthetic/conditions.py`) puts 12 lanes into 55% of the
width. Over 8 seeds, both the default detector and `full` give a median worst-lane error of
**4.644x pitch** (worst seed 5.42x) and a fitted pitch of 111-127 px against a true 64.2 px —
the `width/N` prior is 116.7 px, and that is what the fit returns instead of the gel's pitch.
Total failure, not degradation, and identical in both. The flag is right every time, with a
reason that says what happened: `only 58% of lane positions have a nearby detected peak`, or
`only 3 peaks detected for 12 lanes`. There is no autocrop and no fill-fraction estimator, so
give the endpoints yourself:

```bash
gelpy init "$RUN" "$IMAGE" --n-lanes 20
gelpy lanes set "$RUN" --endpoints 150 4485 --n-lanes 20
```

`--endpoints` takes the outer lane centers in pixels and linspaces between them; `gelpy lanes
pick` is the interactive version. Either way, crop tighter next time.

## The fit-time 1-D baseline eats broad signal, and whether it should exist is open

Two baseline stages run: the 2-D background surface over the image, then a per-lane 1-D baseline
under each profile at fit time. The second defaults to an arPLS-weighted polynomial of order 2
(`gelpy schema BaselineConfig` for the fields). A polynomial with curvature, fitted to a lane
whose residual background is already flat, has only one thing left to fit: signal.

On `example_gel_1` lane 0, changing nothing but the baseline order:

```bash
gelpy background remove "$RUN"
gelpy fit run "$RUN" --lane 0
```

```bash
gelpy fit refit-lane "$RUN" --lane 0 --poly-order 0
```

```console
order 2: "coverage": 1.003, "area_fractions": [0.0014, 0.0061, 0.753, 0.2423], "confidence": "ok"
order 0: "coverage": 1.010, "area_fractions": [0.0012, 0.0064, 0.7106, 0.2915], "confidence": "ok"
```

The dominant component's share moves 0.753 -> 0.711 (-4.2 percentage points), its neighbour's
0.242 -> 0.292 (+5.0 pp). Both report `confidence: ok`; nothing warns you. A 5 pp swing driven
by a baseline knob is larger than most effects people run gels to measure.

How much gets absorbed is governed primarily, at a fixed smear-to-band amplitude ratio, by the
smear's width relative to the fitted window. Measured with `estimate_baseline` on a synthetic
lane — window length L = 400, one broad Gaussian smear of amplitude 0.30 at 0.55L, one sharp
band of amplitude 1.0 and sigma 4 px at 0.30L, flat zero background, no noise, `protect_well`
on — as the fraction of the smear's true area swallowed:

| smear sigma / window length | `poly` 0 | `poly` 2 (default) | `poly` 3 | `arpls` | `asls` |
|---|---|---|---|---|---|
| 0.05 | 0.0% | 0.0% | 0.0% | 0.0% | 9.8% |
| 0.113 | 0.3% | 3.2% | 5.6% | 0.2% | 45.1% |
| 0.150 | 7.7% | 35.4% | 23.4% | 6.2% | 66.9% |
| 0.200 | 108.5% | -5.9% | 52.8% | 90.2% | 85.1% |
| 0.250 | 107.8% | 55.1% | 69.6% | 97.0% | 93.3% |

That is a cliff, not a coefficient — and past the cliff it is not even a monotone one. Below
roughly sigma/L = 0.1 every polynomial baseline is harmless. Above it the fit becomes unstable
rather than steadily greedier: at sigma/L = 0.200 a constant offset is pinned above zero and
over-absorbs by **108.5%** (>100%, the remainder pushed negative elsewhere) while the default
order 2 goes the other way, dipping *below* zero and inflating the smear by **5.9%**. Nor is
absorption ordered by baseline freedom: at 0.150 order 3 absorbs 23.4% against order 2's 35.4%,
and `arpls` (6.2%) sits under both polynomials. No method separates a smear that fills the
window from drift: at that width they are the same function.

The parameterization travels in scale but not in contrast: at fixed sigma/L the numbers are
invariant in L to 0.2 pp (order 2 gives 35.2 / 35.4 / 35.3 / 35.4% at sigma/L = 0.150 for
L = 200 / 400 / 800 / 1364, band width scaled along), but drop the smear amplitude from 0.30 to
0.10 at that same sigma/L and the cell goes from +35.4% to **-122.6%**. Read the table as a
shape, not a lookup. Four consequences:

- The **5.5%** absorption figure recorded for order 2 in `core/peaks/baseline.py`'s docstring
  does not follow from the docstring, which pins neither the window length nor the noise level,
  and both dominate. Noise-free at sigma = 55 px: 0.1% at L = 600, 2.4% at 500, 8.7% at 450,
  29.5% at 400. At L = 600 with white noise: 0.05% at zero noise, 3.3% at sigma_noise = 0.001,
  6.4% at 0.004, 31% at 0.032 (medians over 8 seeds). It reproduces to 5.46% under exactly one
  construction, the one in `tests/docs/test_claims.py` — and not as a width effect: strip the
  noise out of that same construction and absorption falls to 0.05%, while noise with no smear
  gives -0.5%. The figure is a smear-noise interaction through the arPLS weights.
- The ordering the docstring asserts (more freedom absorbs monotonically more) holds only in
  that regime — true at the test's sigma/L = 0.092 (0 / 2.1 / 5.5 / 13.1 / 34.4%), false above
  ~0.12 as the table shows. In the noise-free sigma = 55 px sweep order 1 absorbs *less* than
  order 0 wherever either absorbs anything: 2.9 vs 3.5% at L = 400, 0.7 vs 0.9% at 450,
  0.1 vs 0.2% at 500.
- `asls` at the shipped asymmetry weight absorbed **59.1%** of the smear on a perfectly flat
  zero baseline in the same measurement. It is selectable and it is not calibrated.
- Whether *any* fit-time baseline is right after a good 2-D subtraction is unresolved. Problem
  statement, candidate fixes and measurement protocol:
  [../decisions/2026-07-15-baseline-redundancy-and-well-boundary.md](../decisions/2026-07-15-baseline-redundancy-and-well-boundary.md).
  Until then, `--poly-order 0` on a lane whose residual background is genuinely flat is the
  conservative choice, and which order you used is part of the number you report.

## The well boundary is a fixed fraction of the window, so it does not transfer between gels

A loading well is a cavity with less gel in it, so it reads as a dip below background, and
genuine trapped aggregate is a real peak immediately after that dip. gelpy detects neither. It
assumes the well occupies the first `well_frac` of the profile — default 0.08 — and uses that
one fraction for two jobs: zero-weighting the well during baseline estimation, and labelling a
component `well` when its center falls inside it (`well_edge = well_frac * length`,
`src/gelpy/core/peaks/classify.py:44`).

A fraction of a crop is exactly the geometric guess gelpy refuses to make for lane placement. On
`example_gel_1` lane 0 the `well` component lands at index 89 of a 1364-sample profile — 6.6%,
inside the 8% window by luck of where the crop starts. Crop 200 px higher and it falls out; crop
tighter and a real post-well band falls in and a genuine aggregate peak past the fraction is
silently labelled `sample`.

Mitigation today is per-gel and manual: read the dip's right zero-crossing off one or two lanes
and pass `--slice start end` so the fit window begins at the gel. The general fix — detect the
dip and take a robust consensus boundary across lanes — is designed but not built. Same
decision record as above.

## What the labels and the numbers do not mean

**`component_class` tracks pixel width, not biology.** The thresholds are absolute:
`_DUST_ABS = 3.5` px and `_SMEAR_ABS = 18.0` px of equivalent Gaussian sigma
(`src/gelpy/core/peaks/classify.py:19-20`). On `example_gel_1` lane 0 the two dominant
components have equivalent Gaussian sigma 51 px and 79 px, so both come back `smear` and only
0.6% of the lane's signal is labelled `sample` — the product band sits in the smear bucket. (The
first is an EMG whose own `sigma` param reads 50.4; `classify` thresholds
`area / (height * sqrt(2*pi))` = 51.1, so the `sigma` in `results.json` is not the compared
quantity.) Rescan the same gel at half the resolution and the labels move. Nothing is ever
dropped, whatever the label; treat the class as a width hint and read the fit figure.

**`area` is not mass, and under the default normalization it is not comparable between lanes.**
The default `--normalization area` divides each profile by its own integral
(`src/gelpy/core/line_profiles.py:55-59`), so the `area` column is already a within-lane
quantity. Even with `--normalization safe` the units are canonical image intensity x pixels,
proportional to mass only if staining, exposure and detector response are linear and identical
across the gel. **There is no MW or ladder calibration in gelpy** — no command, no schema, no
code. Area to mass or to a molecular weight is your own standard curve, built elsewhere.

**`area_fraction`'s denominator is the whole lane**: `component area / integral(profile -
baseline)` over the fitted window, with every component counted, dust and smear included
(`src/gelpy/core/peaks/strategies/_common.py`). One dust speck inflates the denominator and
deflates every real band's fraction. On `example_gel_1` lane 0, `coverage = 1.003` means the
fitted model integrates to *more* than the baseline-subtracted data over the window, which is
why `residual_fraction` reads -0.0028 (it is defined as `1 - coverage`). The mechanism is the
model's non-negativity, not overlap: 513 of that lane's 1364 samples sit below the baseline,
carrying -0.0051 of negative mass, and a sum of non-negative peaks cannot follow the data down
there. Overlap is irrelevant — integration is linear, so the components' closed-form areas sum
to the model integral no matter how much they overlap (measured: 0.9897198 against 0.9897168,
a 3e-6 discrepancy against a 0.0028 excess). A negative residual fraction is a baseline
artefact, not an error. Column meanings: [results.md](results.md); normalization choices:
[line-profiles.md](line-profiles.md).

## `confidence: ok` is weaker than it reads

`status: "ok"` does **not** mean the optimizer converged. It means the strategy returned at
least one component *and* the window held more signal than `3 * noise_sigma * sqrt(n)`
(`_EMPTY_SNR = 3.0`, `src/gelpy/core/peaks/strategies/_common.py:24,92-99`) — otherwise
`"empty"`. Convergence lands on `FitResult.success` (`src/gelpy/core/peaks/driver.py:236`) and
is read nowhere in the package, so a `least_squares` that failed still reports `ok`. The
schema's third value, `"failed"`, is never constructed anywhere in `src/gelpy` — only `ok` and
`empty` are — though `gelpy status` counts it. `confidence: "ok"` means no heuristic fired.
Neither says the areas are right.

Fit suite, 48 runs of `grow_select` at default config: mean area-fraction error **0.035**,
**90%** of runs under the 0.10 pass threshold, class accuracy 99%, median 25-31 ms per lane and
worst 0.34-0.49 s across seven repeats on one laptop (timings are machine-dependent; the error
figures are not). Of the 48:

- **2 (4%) are silently wrong** — both `dust_on_band` (6 dust specks landing on one band), area
  errors 0.110 and 0.116, reported `confidence: ok`.
- 2 (4%) are flagged needlessly — `single_faint` at 0.083 and 0.098, reported `low`.
- One condition fails outright: `mixed_five` (well + band + tailed band + smear + dust on a
  curved baseline), mean area error **0.220**.

Three specific ways the flag misleads:

- **`--auto-escalate` searches for a flag, not for a better fit.** It climbs
  `grow_select -> sparse_seed -> bayesian` and stops at the first tier whose confidence is `ok`
  (`src/gelpy/runfolder/commands.py:486-497`). On `mixed_five` seed 7 it does worse than that:
  `grow_select` gives area error 0.211 and `low`, `sparse_seed` gives **0.148** and `low`,
  `bayesian` gives 0.209 and `ok`. The ladder steps *past* the best of the three because that
  tier is flagged, and settles on one 41% worse — but flagged `ok`. Escalate if you like, then
  read the figure, not the flag.
- **`area_uncertainty` does not price the split between overlapping bands.** It is a
  delta-method 1 sigma from that component's own diagonal block of the covariance
  (`src/gelpy/core/peaks/strategies/_common.py:31-45`) — cross-component covariance is dropped.
  Two overlapping bands can each report a small uncertainty while the boundary between them is
  barely determined. The `overlaps a neighbor (areas correlated)` reason and the identifiability
  flag are what tell you that. `area_uncertainty == 0.0` means degenerate, not certain.
- **The `bayesian` tier does not give credible intervals.** It samples around the MAP fit with
  the component count fixed to `grow_select`'s choice, and reports posterior-mean parameters
  with the same delta-method 1 sigma — not a quantile interval, and not on `area_fraction`.

[fit-engine.md](fit-engine.md) explains the reason strings;
[reading-the-figures.md](reading-the-figures.md) is what actually catches these.

## Not built

Designed for, absent from the code — `gelpy --help` is the authority:

- **`replay`.** `log/history.jsonl` is written and appended to; nothing reads it back.
- **MW / ladder calibration**, **`tune` / `refit-all`** auto-tuning, **notebook scaffolding**.

## Sharp edges

- `gelpy init` and `gelpy lanes detect` disagree on strip width. `detect` uses `0.7 * pitch`;
  `init` without `--width` records `image_width / (N * 2.5)` (`guess_profile_width`,
  `src/gelpy/core/lanes.py:744-752`). On `example_gel_1` that is 159 px against 93 px -- 0.70x
  pitch against 0.41x -- i.e. two different measurements of one gel. `lanes set` without
  `--width` keeps the width the recipe holds, whichever of the two wrote it, so pass `--width`
  if you place lanes by hand and want the width to be neither.
- `--format text` prints a pydantic result as single-line JSON, not prose (`_terse` in
  `src/gelpy/cli/_io.py`), and it is not uniform across commands: `results show` ignores `text`
  and prints indented JSON (its own `--format` help lists only `json|csv|table`), while the
  commands that return a plain dict — `background show`, `models list` — fall
  through to `emit`'s `str(value)` branch and print a single-quoted Python repr that is not
  valid JSON at all (`{'names': ['bayesian', 'grow_select'...`). For a table
  use `gelpy results show --format table`. And `gelpy status` reports each lane's `status` but
  not its fit `confidence`, so a run can read 20-for-20 `ok` there with several low-confidence
  lanes. Read `results.json` for flags.
- The persisted `BaselineFit` records `lam` even for a polynomial baseline, never records the
  polynomial order, and never populates `poly_coeffs`. `fit_config.json` is the only durable
  record of which order produced a fit — [concepts.md](concepts.md).
- The verified way to drive gelpy from an agent is Claude Code with a shell on the machine
  holding the images; other agent environments are untested — [driving-from-an-agent.md](driving-from-an-agent.md).
- **A deflate-compressed TIFF can inflate past what its header declares, and nothing checks it.**
  The pixel and page caps count what the header declares, before any pixel is decoded
  ([image-input.md](image-input.md#size-limits)). The inflated size of a compressed strip is in no
  header, and no check for it costs less than decoding the strip. Measured with tifffile 2026.6.1
  (no `imagecodecs`): a 195 KB file that declares one 10 x 10 strip whose deflate stream inflates
  to 200 MB grew the process by about 200 MB while it was read, and came back as a 10 x 10 image.
  Deflate expands up to about 1000 to 1, so one file can cost about 1000 times its size, not what
  the cap allows. For files you do not trust, run gelpy in a process with a memory limit
  (`ulimit -v`, a container) and limit the size of the files you accept. To reproduce, write a
  classic TIFF whose strip is `zlib.compress(bytes(200_000_000))` under a 10 x 10 header and call
  `gelpy.core.image_io.decode_image` on it, watching the peak resident size.

- **Two warnings on stderr that are not about your data.** Neither stops a command, and neither
  changes a result.
  - *Glyph missing from the font.* A lane label with characters that matplotlib's default font
    (DejaVu Sans) lacks, such as Chinese, Japanese or Korean, prints a matplotlib `UserWarning`
    (`Glyph 32454 (\N{CJK UNIFIED IDEOGRAPH-7EC6}) missing from font(s) DejaVu Sans`) from every
    command that draws it, and the figure shows an empty box in its place. The labels in
    `recipe.json`, `results.csv` and the HTML report are intact. gelpy does not pick a fallback
    font; set one for matplotlib (`font.family` in a `matplotlibrc`) if the figures must show
    them.
  - *Accelerate and `matmul`.* numpy 2.0 to 2.2 on arm64 macOS, built against Apple's Accelerate,
    can print `RuntimeWarning: divide by zero encountered in matmul` (also `overflow` and
    `invalid value`) while a fit runs, for products whose result is finite. The cause is in those
    numpy builds, not in the data, and numpy 2.3 and later, which use a newer Accelerate, do not
    show it (not re-measured here). The declared
    floor, `numpy>=2.0`, includes the affected releases.

## Re-measuring after a change

The fit suite is cheap — a few seconds. From the repo root:

```bash
python - <<'PY'
from tests.synthetic.fit_harness import default_fit, evaluate, format_table, per_condition_table
res = evaluate({"grow_select": default_fit})
print(format_table(res))
print(per_condition_table("grow_select", res))
PY
```

The detection suite is `python -m tests.synthetic.harness`, which prints both the pinned `full`
reference and the shipped default. It is 200 detections per detector and took 246 s for both
here — too slow to run as part of this page's tests.

These two harnesses are what this project rates itself against. If you change a default and the
numbers above move, this page is wrong and needs re-measuring — [extending.md](extending.md).
