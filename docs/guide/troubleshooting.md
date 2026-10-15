---
title: Troubleshooting
audience: both
summary: Symptom to cause to the exact command that fixes it, for the failures that actually happen -- bad overlays, low-confidence detections and fits, stale results, a changed source image.
subject:
  - src/gelpy/cli/_io.py
subject_digest: fe38f218542cae9d
---

Every symptom below was reproduced against `examples/data/folding_screen.tif` (4688 x 1364 px,
uint16, 20 lanes); the numbers are what that run produced. Ordered by how often it bites.

## The three moves, in this order

1. **`gelpy status "$RUN"`** reads the recipe, manifest and fit-config store and recomputes
   nothing (`runfolder/commands.py:590-602`): free, and it works even when the source image is gone.
2. **Look at the render**: `render/lane_detect.png` for placement, `render/<kind>_lanes.png` for
   the strips on the gel, `render/fits/lane_NNN.png` for one fit.
3. **Ask the runtime what the command takes** -- `gelpy fit run --help`,
   `gelpy schema RefitLaneRequest`, `gelpy models list --kind shape`. Never guess a flag.

Failures are structured. `handle_errors` in `cli/_io.py` and the root command group in
`cli/app.py` turn every failure -- a domain error, a bad value, an unreadable file, a malformed
command line -- into one `ErrorResponse` printed as JSON **on stderr**, and exit 1 (2 for a
malformed command line). Branch on `code`, not on the message; `detail` names the next step and
`available` usually lists the valid choices. Catalogue:
[reference/errors.md](../reference/errors.md).

```bash
export IMAGE="${IMAGE:-examples/data/folding_screen.tif}"
export RUN="${RUN:-/tmp/gelpy-troubleshooting}"
gelpy init "$RUN" "$IMAGE" --n-lanes 20
gelpy lanes set "$RUN" --endpoints 149 4489 --n-lanes 20 --width 159
gelpy background remove "$RUN" --model robust_plane
gelpy status "$RUN"
```

Those endpoints and that width are what `gelpy lanes detect "$RUN" --n-lanes 20` measured here
(`pitch` 228.43 px, `profile_width` 159, `confidence` `"ok"`); `lanes set` replays it in about 2 s
instead of the 33-37 s the detection cost.

## Symptom table

| symptom | most likely cause | run this |
|---|---|---|
| Red strips are not on the lanes | the placeholder layout from `init` was never replaced | `gelpy lanes detect "$RUN" --n-lanes N --apply`, then look at `render/lane_detect.png` |
| `lanes detect` returns `confidence: "low"` | wrong `--n-lanes`, or no wells in the projection band | read `confidence_reason`, retry with `--y-start`/`--y-end`, fall back to `lanes set --endpoints` |
| `"code": "lane_selection"` | bare digits are **indices**, not labels | `available` in the error lists every label; force a label with `label:<text>` |
| `"code": "source_image_changed"` | the file on disk no longer matches `recipe.json` | restore the file, or `gelpy init` a fresh run folder |
| `"code": "not_found"`, the message says the source image recorded for this run is missing | the image was moved or deleted after `init` | put it back where `detail` says, or set `source_image.path` in `recipe.json` to where it is now |
| `"code": "background_required"` | `profiles extract`, `fit run` or `fit refit-lane` on a run with no background removal | `gelpy background remove "$RUN"`; pass `--allow-raw` only to measure the raw image on purpose |
| `"code": "registry"` | misspelled model, strategy or shape | `available` is the whole list; also `gelpy models list --kind fit\|shape\|background` |
| `"code": "validation_error"` | a value outside a `Literal` | parse `detail` (JSON inside a string); `gelpy schema <Model>` for the allowed set |
| `"code": "invalid_value"` on `report html`, the message says the folder is not empty and no earlier `report html` wrote it | `--out`, or a `report/` from before 1.1, holds files that are not a gelpy report (it has no `.gelpy-report.json`) | pass an empty or new folder, or one an earlier `report html` wrote; [the HTML report](html-report.md#where-it-writes) says what it replaces |
| `"code": "invalid_value"` on `report html`, the message says the folder is the run folder, inside it, around it, holds a `recipe.json`, or `report` is a link | a report is never written into a run folder or around it, and the default `report/` must be a real folder | pass a new or empty folder outside the run folder; for a link, delete it (not what it points to) and repeat the command |
| `status` says `report.current: false` | the recipe, a fit or the gelpy version changed since the page was built | `gelpy report html "$RUN"` |
| A lane is `confidence: "low"` | coverage, R2, at-bound or area uncertainty -- see below | `gelpy fit refit-lane "$RUN" --lane N ...` with the knob that reason names |
| A band you can see is missing from the fit | the fit-time baseline absorbed it | `--poly-order 0` or `--baseline none`; check `render/fits/lane_NNN.png` |
| `adjacent lane profile strips overlap` | `--width` exceeds the lane pitch | `gelpy lanes set "$RUN" --width <below the pitch>` |
| The file `results.csv` shows rows `results show --format json` does not | the file is a snapshot; only a fit command rewrites it | `gelpy fit run "$RUN"`, or read results through `gelpy results show` (every format is live) |
| `n_stale > 0` | a recipe edit, or a new gelpy version, moved the expected `fit_hash`, or a fit's cache file is missing or damaged | `gelpy fit run "$RUN"` -- read the warning in [stale results](#stale-results-and-a-disagreeing-csv) first; `stale_stages` in `gelpy status` lists what it would recompute |
| Bands look dark, numbers look inverted | polarity detection was flagged `low` and defaulted to bright-on-dark | `gelpy image set "$RUN" --invert`, then `gelpy image show "$RUN" --kind raw` |
| Band positions jumped after a refit | `maximum_position` is relative to `slice_start` | add `slice_start` back, or refit without `--slice` |

## Overlay lines are not on the lanes

Recoveries in cost order; stop at the first that works.

```bash
gelpy image show "$RUN" --kind non_linear --lanes
```

1. **Look, for free.** `--lanes` writes `render/non_linear_lanes.png` with the red strips exactly
   where the profiles will be taken; without `--lanes`, `image show` renders no overlay.
   (`lanes detect` writes its own overlay, `render/lane_detect.png`, unless you pass `--no-plot`.)
2. **Re-detect** -- `gelpy lanes detect "$RUN" --n-lanes 20`, 33-37 s here. The grid is fitted to a
   1-D projection of the image, so it cannot be off by a constant the way a guessed crop fraction
   can. `--n-lanes` is ground truth and is never inferred.
3. **Restrict the projection band** with `--y-start`/`--y-end` (the wells are usually the most
   periodic part of the image), or raise `--prominence` (default 0.05) to drop noise peaks. Do not
   reach for `--min-width`: it applies to the smoothed projection -- a moving average of 0.7 x the
   expected pitch, `core/lanes.py:31` -- not to the trace the overlay plots, so widths read off
   the figure do not transfer.
4. **Set the geometry by hand**: `lanes set --endpoints <left> <right> --n-lanes 20` spreads N
   centres evenly between the outer two -- instant and exact; `--x-center` (repeatable) places
   each lane individually.
5. **Click them**: `gelpy lanes pick "$RUN"` needs a real display, the one command that cannot run
   headless (without one it answers `display_required`; a window closed before the clicks are done
   answers `pick_cancelled`).

Why a grid rather than per-lane peak-picking: [lane-placement.md](lane-placement.md).

## `lanes detect` says `confidence: "low"`

A flag to go and look, not a failure: nothing is persisted without `--apply`.

```bash
gelpy lanes detect "$RUN" --n-lanes 40 --no-plot --projection-strategy full
```

That deliberately lies about the lane count (40 for a 20-lane gel) to produce a real one:
`n_detected` 20, `n_expected` 40, `confidence` `"low"`, `confidence_reason`
`"only 50% of lane positions have a nearby detected peak"`. Those two flags cut it to 3 s from 13:
`--no-plot` saves only the overlay's drawing (under a second: the detection runs once with or
without it), and `full` takes one full-height mean instead of the default `d4_union`, which
builds several band projections and keeps the best -- that is where the other 10 s go.

Every reason string (`core/lanes.py:692-741`):

| `confidence_reason` | trigger | action |
|---|---|---|
| `only N peaks detected for M lanes` | fewer than `max(2, N//2)` peaks found | project over the wells (`--y-start`/`--y-end`); lower `--prominence` |
| `N peaks detected, expected M (+/- a few) ...` | more than N + 2 peaks | raise `--prominence`; look for dust streaks |
| `only X% of lane positions have a nearby detected peak` | under 60% of grid nodes have a peak near them | check `--n-lanes` is the true count, then the projection band |
| `matched-filter pitch pinned at the search-band edge ...` | the width/N pitch prior is fighting the data, e.g. the gel does not fill the crop | check the crop, then set endpoints |
| `detected peaks don't sit close to the fitted grid`, `a lane center fell outside the image`, `degenerate pitch (<=0)`, `no peak detected` | uneven spacing, a collapsed pitch, or nothing bright in the band at all | set the geometry by hand (`lanes set --endpoints`/`--x-center`); check polarity |

## A lane selector matched nothing, or the wrong lanes

```bash
# doc-test: expect-failure
gelpy fit run "$RUN" --lane frac9 --no-plot
```

```console
{
  "error": "Lane selector ['frac9'] matched no lanes.",
  "code": "lane_selection",
  "detail": "Select lanes by a label from `available`, a 0-based index, or `re:<pattern>`; `gelpy status RUN` lists the lanes.",
  "available": [
    "1",
    "2",
    ...
    "20"
  ]
}
```

The error carries every label, so you never open `recipe.json` to find out what you should have
typed. The grammar (`cli/_io.py:38-63`, resolved in `core/lanes.py:787-809`) has four cases; the
first surprises people:

```bash
gelpy profiles extract "$RUN" --lane label:1 --no-plot
gelpy profiles extract "$RUN" --lane 're:^1' --no-plot
```

- **Bare digits are an index.** Labels here are `"1"`..`"20"`, so `--lane 1` selects index 1,
  label `"2"`, while `--lane label:1` selects index 0 -- the first command above reports
  `recomputed: [..., "profile:lane=0"]`.
- **`re:<pattern>` is a regex over labels and returns every match** -- `re:^1` hit 11 lanes here
  (`"1"` plus `"10"`..`"19"`). Anchor both ends if you mean one lane.
- **Anything else is a literal label match**, also returning every lane with that label.
- **No negative indexing.** `--lane -1` parses as -1: `Lane index -1 out of range (0..19)`.

`--lane` is repeatable on `fit run` and `profiles extract`; omitting it means all lanes, and so
does `--lane all`. `refit-lane` takes one term and refuses a selector matching several lanes.

## A lane came back `confidence: "low"`

`status: "ok"` means the fit routine returned components. It is not evidence the fit is right, and
not even evidence the optimizer converged: convergence is recorded in `FitResult.success`
(`core/peaks/driver.py:236`), nothing in the package ever reads it, and `status` is set to `"ok"`
unconditionally once there are components (`core/peaks/strategies/_common.py:158`). `confidence` is
the quality signal.

```bash
gelpy fit run "$RUN" --lane 0 --no-plot
gelpy fit refit-lane "$RUN" --lane 7 --slice 50 600 --no-plot
```

Lane 0: `confidence: "ok"`, 4 bands, `coverage` 1.003, two reasons of the form
`note: band N overlaps a neighbor (areas correlated)`. Entries prefixed `note:` are surfaced but
never flip a lane to `low` (`core/peaks/quality.py:85-91`). Lane 7 over the window 50-600:
`confidence: "low"`, 1 band, `coverage` 0.775, reason
`coverage 77%: 23% of the signal is unmodeled`. Thresholds (`core/peaks/quality.py:18-22`):
coverage outside 0.85-1.10, R2 below 0.90, or a band whose area uncertainty exceeds 100% of its
area (50% is a `note:`).

The `--slice` did not open that gap, which is the first thing everyone assumes. Coverage divides
fitted areas by the integral of the **sliced** profile (`core/peaks/strategies/_common.py:87,154`),
so signal a window removes leaves the denominator with it -- lane 7's bright bands sit at rows
600-1364 and are absent from both sides of the ratio. The deficit is inside the window. Integrating
that fit's residual over the window integral: +16% across rows 70-120, the well plateau one narrow
pseudo-Voigt cannot cover; +11% in a low broad shoulder around rows 300-600 that no component was
seeded on; about -5% where the baseline rides above the trace. Nothing reached the shoulder because
every residual bump the grower found sat on the well, and AICc stopped growth after two components
(which consolidate to the one reported band). Over the full profile the same lane returns 5 bands
at `coverage` 0.999.

| reason | meaning | action |
|---|---|---|
| `coverage X%: Y% of the signal is unmodeled` | components sum to less than the signal | widen the window (loosen or drop `--slice`); allow tails with `--shapes emg,bi_gaussian`; `--method sparse_seed`; `--auto-escalate` |
| `coverage X%: fitted peaks exceed the signal (overlap/over-fit)` | overlapping components double-count | narrow `--shapes`; check the baseline is not pushed low (`--poly-order 0`) |
| `poor fit (R2=...)` | the model does not describe the trace | open `render/fits/lane_NNN.png` before changing anything |
| `band N (class) width ran to a bound` | one component grew to absorb drift | not a tier problem: narrow the window or lower `--poly-order` |
| `band N area unreliable (+/-X%)` | the covariance says that area is not identifiable | treat that area as unusable; `--method bayesian` gives credible intervals instead |
| `no significant signal detected` (`status: "empty"`) | the integrated signal is below 3 x noise x sqrt(n) (`strategies/_common.py:24,92`) | check lane placement and `--slice`; an empty lane is often a misplaced strip |

`--auto-escalate` climbs `grow_select` -> `sparse_seed` -> `bayesian` and stops at the **first**
tier that returns `ok` (`runfolder/commands.py:476-497`). On lane 7 over the same 50-600 window it
stopped at `sparse_seed` with `coverage` 0.881 and `confidence: "ok"`, while forcing
`--method bayesian` gave `coverage` 0.740 and stayed `low`. A higher tier is not a better fit, and
clearing the flag is not being right: 0.881 still leaves 12% of the signal unexplained, it merely
clears the 0.85 threshold. Mechanism: [fit-engine.md](fit-engine.md); cases where nothing helps:
[known-limitations.md](known-limitations.md).

Two knobs that gate adding a component are not on the CLI: `priors.min_snr` (default 3.0) and
`priors.max_components` (default 12). They live in `fit_config.json` -- hand-editable, not
disposable -- and `fit refit-lane` merges onto the stored config, so an edit there moves the fit
hash and the lane recomputes. Lowering the gate mostly buys noise: on lane 0, `min_snr` 1.0 with
`max_components` 20 still chose 4 components (the next residual bump has SNR 0.95, so the prior was
still what stopped growth), and `min_snr` 0.5 added a fifth band at `area_fraction` 0.0034 with
`coverage` unchanged at 1.003.

## The baseline ate a smear

Silent failure: `coverage` still reads about 1.0 and `confidence` can still read `ok` while a
broad component has been absorbed into the baseline and its area is gone. The tell is the fit
figure -- the baseline bending up under a shoulder instead of running flat.

```bash
gelpy fit refit-lane "$RUN" --lane 3 --slice 100 550 --poly-order 2 --no-plot
gelpy fit refit-lane "$RUN" --lane 3 --slice 100 550 --poly-order 0 --no-plot
```

Same lane, same window, one number different:

| | bands | confidence | largest `area_fraction` |
|---|---|---|---|
| `--poly-order 2` (default) | 8 | `low` (two bands at +/-142% and +/-146%) | 0.534 |
| `--poly-order 0` | 7 | `ok` | 0.447, plus a `smear` component at 0.037 |

Nine percentage points of reported product fraction from the baseline order alone. Order 2 has
enough freedom to bend into the smear once `--slice 100 550` removed the well dip that anchored the
left end of the window; order 0 (a constant) cannot. Measured absorption of a genuine broad smear
-- a sigma = 55 px hump on a flat residual background, as a fraction of the smear's true area
(`core/peaks/baseline.py:1-32`): `none` 0.0%, `poly` order 0 2.1%, order 1 2.2%, order 2 5.5%,
order 3 13.1%, `arpls` 34.4%, `asls` 58.5% (both at `lam` 1e5) -- monotonic in baseline freedom.

Use `--poly-order 0` whenever the window starts right after a hard exclusion boundary.
`--baseline none` is defensible after a good 2-D background subtraction; whether any fit-time
baseline is the right default there is an open question, recorded with this measurement in the
repo's design records.

## Stale results, and a disagreeing CSV

```bash
gelpy lanes set "$RUN" --width 300
gelpy results show "$RUN" --format json
gelpy results show "$RUN" --format csv | head -3
gelpy lanes set "$RUN" --width 159
gelpy results show "$RUN" --format table
```

That one `lanes set` does three things.

- **It warns about overlap.** At `--width 300` against a 228 px pitch:
  `overlapping_strips: [[0,1],[1,2],...]`, message `adjacent lane profile strips overlap at this
  width`. Non-fatal -- gelpy will average neighbouring lanes into each other if you ignore it.
- **It makes every fit stale.** Lane width feeds `fit_hash`, so `results show --format json`
  returns an empty `lanes` list at once: `runfolder/results.py:26-46` recomputes each lane's
  expected hash and drops any cached fit that no longer matches. `status` agrees -- `n_fit: 0`,
  `n_stale: 3`, and `fit_current: false` on all three lanes fitted above (0, 3 and 7).
- **The file `results.csv` does not agree.** `results show --format csv` does: it renders the same
  live assembly as the other formats (`cli/app.py:617-624`) and prints the header row alone here.
  The file is different: only `fit run` and `refit-lane` ever rewrite it
  (`runfolder/commands.py:523,563`), so it keeps rows `results show` has already dropped. Treat
  `results.csv` as a snapshot dated to the last fit command, and read results through
  `results show`.

Width 159 restores everything from cache with no recompute -- the hash returns to a value the cache
already holds ([concepts.md](concepts.md)).

The obvious fix for `n_stale > 0` has a cost: `gelpy fit run` rebuilds each selected lane's fit
config **from the command line** and writes it into `fit_config.json`, discarding per-lane
`--slice`/`--poly-order`/`--shapes` set with `refit-lane`. Verified: `fit run --lane 3` reset lane
3's stored `slice_start` to 0 and `poly_order` to 2. To refresh a lane without losing its intent,
run `fit refit-lane` with no overrides -- that merges onto the stored config.

## The source image changed

```bash
# doc-test: expect-failure
WORK="$(dirname "$RUN")"
cp "$IMAGE" "$WORK/gel-copy.tif"
gelpy init "$WORK/copy-run" "$WORK/gel-copy.tif" --n-lanes 4 > /dev/null
printf '\0' >> "$WORK/gel-copy.tif"
gelpy profiles extract "$WORK/copy-run" --lane 0 --no-plot --allow-raw
```

```console
{
  "error": "Source image '.../gel-copy.tif' (recorded as '../gel-copy.tif') has changed on disk (recipe sha256 6031cfb91ba5…, current 18ecc93ea75f…). Re-run `gelpy init` against a fresh run folder, or point the recipe at the intended file.",
  "code": "source_image_changed",
  "detail": "Restore the original file, or analyse the changed image in a new run folder (`gelpy init NEW_RUN IMAGE`).",
  "available": null
}
```

The message names where gelpy looked (the directory is elided above) and how the recipe records
the image. The recipe records the image's sha256; the pipeline checks a cheap (size, mtime)
fingerprint of the file first and re-hashes only when that moved (`runfolder/pipeline.py:97-103`).
A byte change is refused rather than cached under a stale hash: the alternative is a run folder
whose pixels and recipe describe different images. Restore the file or re-initialise -- no flag
overrides the check. `gelpy status` still works here; it never touches pixels.

If the image is gone, the command fails with `code: "not_found"` and the message says that the
image recorded for this run is missing; `detail` names where to put it back, or tells you to point
`source_image.path` in `recipe.json` at where it is now.

`--allow-raw` is in the block only because `copy-run` has no background stage. Without it,
`profiles extract` stops at `background_required` before it reads a pixel and the check above is
never reached.

## Polarity

Everything below `image_io` assumes signal is bright, so a mis-detected polarity measures the gaps
between the bands. Two independent cues (histogram skew, tail asymmetry) must agree and be
non-trivial to earn `ok`; otherwise the image is left un-inverted and flagged
(`core/image_io.py:120-146`), and `status` surfaces `polarity`, `polarity_confidence` and a
`warnings` entry naming the fix. Check by eye with `gelpy image show "$RUN" --kind raw` -- bands
must be bright -- and force it with `gelpy image set "$RUN" --invert` / `--no-invert`, or re-detect
with `--auto`. Confirming a decision that already holds is free: `image_hash` folds in `inverted`,
`channel_strategy`, `channel_index` and `page`, not the confidence (`runfolder/stages.py:42-57`),
so `image set --no-invert` on an already-bright-on-dark run recomputed nothing on the next fit.
Flipping it invalidates the image and, by chaining, every background, profile and fit:
[image-input.md](image-input.md).

## Not errors, though they look like it

- **A closed pipe exits 1 without a word.** If the reader of gelpy's stdout leaves before gelpy has
  written everything (`gelpy schema FitConfig | true`), gelpy stops with exit 1 and prints nothing:
  no error object, because a reader that leaves early is not a failure of the command, and no
  traceback. `set -o pipefail` or
  `${PIPESTATUS[0]}` shows the 1; without them the pipeline reports the reader's status.
- **`n_failed` is always 0.** No code path sets a fit's `status` to `"failed"`; the only outcomes
  constructed are `"ok"` and `"empty"` (`core/peaks/strategies/_common.py:94,157`), and the field
  survives only because a foreign cache file could carry it. Never read `n_failed: 0` as "no fit
  went wrong" -- read `confidence`.
- **`profiles extract --normalization` and `--slice` are plot-only.** Both travel as
  `ExtractProfilesRequest.profile` and only the plot reads them (`cli/tools.py:111-113`); the
  cached profile is always the raw full-length one. They are validated all the same:
  `--normalization bogus --no-plot` is a `validation_error` whose `available` lists `area`,
  `min_max` and `safe`. Slicing and normalization that affect a *fit* are set per lane by
  `fit run`/`fit refit-lane`.
- **`maximum_position` is relative to `slice_start`.** With `--slice 50 600` lane 7's component
  sits at `39.7`, not image row ~90. The per-lane figure uses the same axis, so those agree, but
  positions compare across lanes only when the window matches ([results.md](results.md)).
- **`--format text` is not prose.** `cli/_io.py:95-109` dumps the model as single-line JSON, for
  `jq`. `results show --format table` is the human view.
- **A deleted `manifest.json` empties `status` and `results show`.** Both read the ledger, so they
  report `n_fit: 0` and an empty `lanes` list while every cache file is still on disk. A fit
  re-records the ledger and reuses those caches: after a wipe, `fit run --lane 0` reported
  `recomputed: []`. The manifest is disposable by design; `recipe.json` and `fit_config.json`
  are not.

## When none of the above applies

```bash
gelpy schema RefitLaneRequest | head -20
gelpy models list --kind shape --format text
```

`gelpy schema <Model>` prints the JSON Schema of any contract exported from `gelpy.schemas`,
descriptions and defaults included -- that is where a default you are unsure about is written down.
A misspelled name lists every available model, so `gelpy schema x` is a usable index.

If behaviour contradicts this page, the page is the bug; report it as one. [reference/](../reference/)
is generated from the code and cannot drift, so diff your expectation against that first.
