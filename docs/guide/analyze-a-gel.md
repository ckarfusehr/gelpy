---
title: Analyse a gel
audience: both
summary: The full workflow in the order a human runs it -- init, place lanes, verify the overlay, remove background, fit, inspect, refit, read results.
subject:
  - src/gelpy/cli/app.py
  - src/gelpy/runfolder/commands.py
  - src/gelpy/schemas/requests.py
  - src/gelpy/core/text.py
subject_digest: 0a27e410dee58fbd
---

Every command below was run against `examples/data/folding_screen.tif` (4688 x 1364 px,
uint16, single page, 20 lanes) while writing this page; the numbers quoted are what it
produced. Two ideas make the sequence make sense:

- **Each command records intent, not results.** `lanes detect --apply`, `background remove`
  and `contrast set` write `recipe.json`; pixels are computed lazily, content-hashed and
  cached, so re-running a step costs a hash comparison ([concepts.md](concepts.md)).
- **The run directory is a positional argument on every command that touches one.** There is
  no ambient "current run" (`cli/app.py:1-9`), so your shell history is the analysis.

For exhaustive option lists read [reference/cli.md](../reference/cli.md), or ask the runtime:
`gelpy fit run --help`, `gelpy schema FitRunRequest`, `gelpy models list --kind fit`.

## 0. Set up the run folder

```bash
export IMAGE="${IMAGE:-examples/data/folding_screen.tif}"
export RUN="${RUN:-/tmp/gelpy-folding-screen}"
gelpy init "$RUN" "$IMAGE" --n-lanes 20
gelpy status "$RUN"
```

`--n-lanes` is the true lane count from your plate sheet: ground truth, never inferred. The
default is 10, which is almost certainly not your gel. What `init` writes is a **placeholder**
layout -- `n` centres evenly spaced between 5% and 95% of the image width, strip width
`image_width / (n_lanes * 2.5)`, which was 93 px here (`runfolder/commands.py:195-196`, `core/lanes.py:744-752`). It is
a starting rectangle, not a measurement. Running `init` again on a folder that already holds a
run is refused (`run_exists`) unless you pass `--force`, which replaces the run and removes its
results, reports and renders.

`status` is the orientation command: source image, polarity, background model, per-lane
fit/stale counts and the stale stages, read from disk with nothing recomputed. Read its
`warnings` now. A polarity flagged `low` is the one warning never to ignore -- everything
downstream assumes signal is bright, so getting it backwards measures the gaps between the bands
([image-input.md](image-input.md); the fix is `image set --invert` / `--no-invert`).

## 1. Look at the image

```bash
gelpy image show "$RUN" --kind non_linear --lanes
gelpy image show "$RUN" --kind raw
```

`--kind raw` applies no display contrast at all -- the canonical array as the pipeline sees
it, dark for a fluorescence gel, which is what makes it the honest check on canonicalization
and polarity. `--kind non_linear` applies the recipe's gamma (default 0.1) so faint bands are
visible; `--kind corrected` (the default) shows the background-subtracted array *once a
background model is configured* -- before Step 5 the pipeline passes the decoded image through
(`runfolder/pipeline.py:117-123`), so `corrected` and `non_linear` render pixel-identical
panels that differ only in their title. Only `--lanes` draws the red strips; labels are drawn
as top ticks either way.

## 2. Place the lanes from the image

```bash
gelpy lanes detect "$RUN" --n-lanes 20
```

Detection fits an even N-lane grid to a 1-D projection of the image, so lane placement is
derived from the actual bright columns rather than guessed from the crop
(`runfolder/commands.py:326-394`; the algorithm is [lane-placement.md](lane-placement.md)).
Without `--apply` nothing is persisted -- you get the numbers and the overlay PNG.

Measured on this gel: `n_detected` 20 of 20 expected, `pitch` 228.43 px, `profile_width`
159 px, `confidence` `"ok"`, `projection_source` `"full"`, `y_band` `null`.
`projection_source: "full"` means the multi-band row selection did not beat the full-height
column mean here and was dropped. With the overlay on (the default) detection runs once: the
command hands the renderer the image and the detection it already has
(`cli/_render.py:398-414`), so the overlay costs only its drawing -- about 12 s either way, with
or without `--no-plot`, on this 6.4 Mpx image, single run on one machine.

## 3. Verify the overlay. This is not optional

**Never trust lane positions you did not look at.** Open
`$RUN/render/lane_detect.png` and check, in this order:

1. Every red strip sits on a lane over the **whole** gel height, not only at the wells.
   A grid can be pinned by the well row and still drift off a smiling lane further down.
2. The strip is narrower than the pitch and still covers the band. Here 159 px of strip
   inside a 228 px pitch. The only width check gelpy has fires on `lanes set`, not on
   `lanes detect --apply` (`core/line_profiles.strips_overlap` has one caller,
   `runfolder/commands.py:288`), and only when adjacent strips actually overlap -- a much
   weaker condition than "correct".
3. The label above each lane is the sample you think it is.

`confidence: "ok"` is a statement about geometry, not about biology: peak count between N/2
and N+2, at least 60% of grid nodes with a peak within half a pitch, matched-peak residuals
under 0.3 pitch, pitch not pinned at the search-band edge
(`core/lanes._detection_confidence`). For N=20 that accepts anything from 10 to 22 detected
peaks, and nothing in the list separates a lane from any other set of evenly spaced bright
columns. The overlay does.

Nor can you skip it by reusing numbers from an earlier analysis: the recipe stores
**absolute** pixel centres (`LaneDefinition.x_center_px`), so re-cropping or re-scanning the
same physical gel moves every one of them. Lane positions never transfer between images --
which is why detection is image-derived.

If the overlay is wrong, in increasing order of desperation: restrict the projection band
(`--y-start` / `--y-end`; the well region often separates faint lanes best), raise
`--prominence` to drop artefact peaks, or go manual with
`gelpy lanes set "$RUN" --endpoints <left> <right> --n-lanes 20`, reading the two outer
centres off the overlay. `gelpy lanes pick` does that interactively but needs a GUI, so it is
useless headless or from an agent. [troubleshooting.md](troubleshooting.md) has the decision
tree.

Once the overlay is right, persist it:

```bash
gelpy lanes detect "$RUN" --n-lanes 20 --apply --no-plot
```

`--apply` also overwrites the strip width with the detected one (159 px here, replacing the
93 px guess) -- `runfolder/commands.py:368`. If you had hand-set a width, re-apply it with
`gelpy lanes set "$RUN" --width <px>` afterwards.

## 4. Label the lanes

```bash
seq -f 'cond_%02g' 1 20 > "$RUN/labels.txt"
gelpy lanes set "$RUN" --labels-file "$RUN/labels.txt"
```

One label per line; blank lines are dropped, so the count must match the lane count exactly
or the command fails with `code: "invalid_value"` and the message
`labels (2) must match positions (20)`; its `detail` says to pass one label per lane position.
The file is read as UTF-8 (a byte order mark at its start, which Notepad and Excel write, is
ignored): a file in another encoding fails with the same code, and the message names the file,
the encoding and the line (a line ends at LF, CR LF or a lone CR). Labels are cosmetic to the cache: the profile hash excludes them
(`runfolder/stages.py:84-95`), so renaming a lane after fitting invalidates nothing.
Verified: `status` still reported `n_fit: 20`, `n_stale: 0` after a relabel. `results show` carries
the new label at once, without a refit.

## 5. Remove the background before extracting anything

```bash
gelpy background remove "$RUN" --plot
```

The default model is `robust_plane`. <!-- claim: default-background-model --> The command only
writes the recipe; the surface is fit lazily the first time an image is needed
(`runfolder/commands.py:432-450`). `--plot` forces it now and writes
`render/background_diagnostic.png` -- surface, weight map, and before/after column slices.

Why before the profiles, not after. The background is a 2-D property of the illumination and
the gel, and a per-lane 1-D baseline cannot represent its cross-lane component. On this gel
the background carries **16 to 22%** of each raw lane profile's integral (corrected/raw
integral ratio 0.780 to 0.841 over the 20 lanes). Normalization then divides each lane by its
own integral (`core/line_profiles.normalize_profile`, trapezoid), so the same physical floor
lands at levels spread by a factor of 1.56 (median of the area-normalized raw profile over
rows 1150-1350: 0.97e-4 to 1.51e-4 across the 20 lanes): a bright lane's floor is crushed by
its own signal integral, a faint lane's is inflated, and the per-lane baselines are no longer
on a common scale.

What that costs, measured on lanes 0 and 9 fitted with and without `background remove` (the raw
side now needs `--allow-raw`): every component's absolute `area` scaled by exactly that lane's
integral ratio (x0.808 and x0.780), while `area_fraction`, `maximum_position`, band count,
component class and `residual_fraction` came out identical to 4+ digits. That is a span
argument, not luck: `robust_plane` fits an order-1 surface by default, and averaging one strip's
columns turns `a + bx + cy` into a function affine in y, which lies inside the order-2 polynomial
baseline's span -- so the per-lane baseline absorbs it. Only `area`, the number meant to be
comparable across lanes, moved, by a lane-dependent 16-22%. Ask for `--order 2`, or let the
background vary faster in y than an order-2 polynomial can follow, and the decomposition moves
too.

gelpy enforces the ordering. `profiles extract`, `fit run` and `fit refit-lane` refuse a run
with no background model configured, with `code: "background_required"`, before they compute
anything. `--allow-raw` is the explicit opt-out for measuring the raw image on purpose: the
pipeline then passes the decoded image straight through (`runfolder/pipeline.py:117-123`) and
says nothing. Configuring the background *after* such a raw fit is safe -- `background_hash`
feeds `profile_hash` feeds `fit_hash`, so every fit goes stale and `status` reports it
(`n_stale`, and the stage keys under `stale_stages`). [background.md](background.md) covers model
choice and the diagnostic figure.

## 6. Fit

```bash
gelpy fit run "$RUN" --plot
```

This needs Step 5: without a background model `fit run` stops with `background_required`, unless
you pass `--allow-raw`. No lane selection means every lane. The default strategy is
`grow_select`, which picks the number of components by AICc and each component's shape itself --
there is no peak-detection threshold to tune. What the parameters mean, and which tier to reach
for, is [fit-engine.md](fit-engine.md).

The 20-lane pass with per-lane PNGs took 37 s (same machine). Re-issuing the same command
recomputes nothing -- 2.4 s, 0 recomputed, 42 stages reused -- and the `recomputed` /
`reused` lists in the output name each stage. `bayesian` is not the expensive tier: measured
per 20-lane pass, no plots, profiles already cached, one run each, `grow_select` 28 s,
`bayesian` 36 s (1.3x, it is a Metropolis walk seeded from the `grow_select` MAP),
`sparse_seed` 140 s (5x, an NNLS deconvolution per lane). Ask for `bayesian` per gel if you
want credible intervals; ask for `sparse_seed` per lane. `fit run` writes `results.csv`,
`results.json` and `report.md` on every invocation, plus `render/fits/lane_NNN.png` per lane
with `--plot`.

## 7. Inspect every lane

```bash
gelpy results show "$RUN" --format table | head -8
ls "$RUN"/render/fits/lane_*.png | head -3
```

Look at all of them. `status: "ok"` means the optimiser converged; `confidence` (`ok` /
`low`) with `confidence_reasons` is the quality signal, and neither replaces the figure. All
20 lanes here read `confidence: "ok"`, `coverage` 0.9995 to 1.0375 -- and `cond_01` still
split one front into a 0.730 + 0.265 pair of `smear` components, which is converged, plausible
on the numbers, and worth a second look on the PNG.
[reading-the-figures.md](reading-the-figures.md) is how to judge one, [results.md](results.md)
what each column means.

## Selecting lanes

`--lane` is repeatable, and each term is an integer index (0-based), `re:<pattern>` for a
regex over labels, `label:<text>` to force a literal match on a numeric-looking label, or a
bare literal label; `all`, or omitting `--lane` entirely, means every lane. <!-- claim: lane-selector-grammar -->

```bash
gelpy profiles extract "$RUN" --lane 're:^cond_0'
gelpy profiles extract "$RUN" --lane label:cond_07 --lane 3
```

The first matched 9 lanes (`cond_01` .. `cond_09`), the second exactly two. A selector that
matches nothing raises `lane_selection` and lists the available labels instead of falling back
to all lanes. And a literal term returns **every** lane carrying that label
(`core/lanes._resolve_term`): give a replicate group one label and `--lane ctrl` takes the
whole group, which is convenient for `fit run` and fatal for `refit-lane`, which insists on
exactly one lane:

```bash
# doc-test: expect-failure
gelpy fit refit-lane "$RUN" --lane 're:^cond_0' --poly-order 0
```

```text
{
  "error": "refit-lane needs exactly one lane; 're:^cond_0' matched 9.",
  "code": "lane_selection",
  "detail": "Pass exactly one lane: a label or an index.",
  "available": [
    "cond_01",
    "cond_02",
    ...
    "cond_20"
  ]
}
```

## 8. Refit the stragglers

`refit-lane` merges the options you pass onto that lane's last-used settings and stores them
in `fit_config.json`, which is per-lane, precious, and hand-editable:

```bash
gelpy fit refit-lane "$RUN" --lane cond_01 --slice 120 1364 --poly-order 0 --plot
```

The profile has one value per image row, so slice bounds are row indices (0 to 1364 here;
`status` reports `image_height`). This window starts past the well cavity, and `--poly-order 0`
pins the baseline to a constant: an order >= 1 polynomial in a window with no anchoring
feature left can bend up and absorb a real smear.

Measured effect on `cond_01`: 4 components became 3 (the `well` component was sliced away),
the largest moved from `area_fraction` 0.730 to 0.697 and the second from 0.265 to 0.301,
`confidence` read `ok` both times. A 4-14% relative shift from a defensible change of window,
with no quality flag either way -- that is what "`status: ok` is not proof" means in numbers.

Three consequences:

- `maximum_position` indexes the **fitted** window, not the image: `cond_01`'s front is
  reported at 907.47 unsliced against 787.36 sliced, i.e. roughly the same band minus
  `slice_start` (907.47 - 120 = 787.47; the refit also moved the centre by 0.11 px)
  (`schemas/peaks.py:96-102`). Add `slice_start` back before comparing.
- This lane now uses a different window from the other 19, so its `area_fraction` is not
  strictly comparable to theirs. Accept that, or re-run the pass with the same options
  (`gelpy fit run "$RUN" --slice 120 1364 --poly-order 0`).
- `--baseline` and `--poly-order` replace the *entire* baseline config, resetting any
  hand-edited `lam` / `p` / `protect_well` / `well_frac` to defaults
  (`runfolder/commands.py:820-827` merges field-level on `FitConfig`).

`--auto-escalate` climbs `grow_select` -> `sparse_seed` -> `bayesian`, stops at the first tier
reporting `confidence: "ok"`, and persists whichever tier it ended on
(`runfolder/commands.py:476-497`).

## 9. Close the loop

```bash
gelpy status "$RUN"
gelpy results show "$RUN" --format csv | head -3
```

`status` should read `n_stale: 0` and `n_failed: 0`, with `n_fit` equal to your lane count.
`n_stale > 0` means a recipe edit, or a new gelpy version, invalidated a fit nobody re-ran, or a
fit's cache file is gone; `stale_stages` lists the recorded stages the next command would
recompute, upstream ones included.
Stale fits are excluded from the assembled result (`runfolder/results._fit_is_current`), and
all three `results show` formats render that same assembly -- csv included
(`cli/app.py:617-624`) -- so they cannot disagree. The **file** on disk can: `results.csv` is
rewritten only by `fit run` / `refit-lane`, so it keeps the rows of the last pass. Measured: after
`gelpy lanes set "$RUN" --width 150`, `status` read `n_fit: 0, n_stale: 20`, `--format table`
printed 0 rows and `--format csv` printed the header alone -- while `$RUN/results.csv` still
held all 129 lines of the post-refit pass (header + 128 band rows). Trust `status` over the
file, and re-run the fit.

The deliverables are `results.csv`, `results.json`, `report.md`, `render/*.png`, and
`log/history.jsonl`, the append-only record of every command that shaped the folder.
`gelpy report html` writes the same run as one static page: [html-report.md](html-report.md).

## Known limitations of this workflow

- The `well` component class is a fixed position bucket (the first 8% of the fitted window),
  not a detection, so it moves when you change `--slice`. See
  [known-limitations.md](known-limitations.md).
- Lane detection has no notion of what a lane is, only of even spacing. Step 3 is the check
  that catches this, and there is no automatic substitute for it.
- `results.csv` can mix lanes fitted with different windows and baselines; `fit_config.json`
  is the only record of what each lane actually used.
