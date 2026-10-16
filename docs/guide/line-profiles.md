---
title: Line profiles
audience: both
summary: How a lane strip becomes a 1-D profile, what the strip width and the three normalization modes do numerically, and why slicing and re-normalizing are cheap.
subject:
  - src/gelpy/core/line_profiles.py
  - src/gelpy/schemas/profiles.py
subject_digest: 1ce9e5c3905147fb
---

## What a profile is

A lane's profile is the row-by-row mean of a vertical strip of `profile_width_px` columns
centred on that lane's `x_center_px`:

```text
profile[y] = mean( I[y, x] for x in [x_center - w/2, x_center + w/2) )
```

One sample per image row. The example gel is 1364 rows x 4688 columns, so every profile is
1364 samples long; `gelpy status` reports that number as `image_height`. Each profile is
cached as one JSON array per lane under `cache/profiles/`. The strip is averaged, not summed,
so the profile stays in the intensity units of the canonical image no matter how wide the
strip is.

The array being reduced is the background-corrected image. gelpy refuses to extract profiles
from a run that has no background stage (`background_required`); only with `--allow-raw` is it
the decoded canonical image instead, and then no background cache entry is written at all. By
that point signal is always bright, so a band is always a maximum.

The axis is the image row index, i.e. distance along the migration direction in pixels. gelpy
never rotates or transposes the source — there is no image rotation or transpose anywhere in
`src/gelpy` (the only `rotation` in the codebase is the cosmetic lane-label angle in plots) — so
lanes must run top-to-bottom in the image; rotate the file before `gelpy init` if they do not.

Profile extraction itself does not care which end of the axis is the well. The fit engine does:
`core/peaks/classify.py` labels every component centred in the first `well_frac` of the window
`well`, and the baseline's well protection down-weights that same region
(`BaselineConfig.well_frac` defaults to 0.08, counted "from the origin"). So the well must sit at
the low-index end — the top of the image — or both the `well` label and the baseline protection
land at the wrong end of the lane.

Everything below assumes a run folder and a source image. Set them once. This block is skipped
by the documentation test, which sets the same two variables itself:

```bash
# doc-test: skip
export RUN=/tmp/gelpy-profiles
export IMAGE=examples/data/example_gel_1.tif
```

```bash
gelpy init "$RUN" "$IMAGE" --n-lanes 20 --format text
gelpy lanes detect "$RUN" --n-lanes 20 --apply --format text
gelpy background remove "$RUN" --format text
gelpy profiles extract "$RUN" --plot --format text
```

`profiles extract` is what fills `cache/profiles/`, and it needs the `background remove` line
above: on a run without one it stops with `background_required`. The plot is optional; the fit
engine reads the same cached arrays.

## Strip width

Three ways the width gets set:

- `gelpy init` guesses `image_width / (n_lanes * 2.5)` — 93 px on the example gel. Purely
  geometric, and the 2.5 divisor is empirical and comb-dependent.
- `gelpy lanes detect --apply` uses 0.7 x the fitted lane pitch — 159 px at the fitted pitch
  of 228.4 px on the example gel. Image-derived, so prefer it.
- `gelpy lanes set --width N` overrides either.

A wider strip averages more columns, so independent pixel noise falls as 1/sqrt(N_columns) —
up to the point where the strip crosses into a neighbouring lane and averages in someone
else's band. `lanes set` checks for that: it flags adjacent lanes whose centre distance is
less than the full width (a gap of exactly
`width` is not flagged) and reports the index pairs in `lanes.warning.overlapping_strips`. It
is a warning, not a refusal — the recipe is written either way.

Width is part of `profile_hash`, together with the background hash and `x_center_px`, so
changing it recomputes profiles. Caching is content-addressed rather than overwritten, so
reverting to a width you already used is a cache hit again:

```bash
gelpy lanes set "$RUN" --width 300 --format text
gelpy profiles extract "$RUN" --lane 0 --format text
gelpy lanes set "$RUN" --width 159 --format text
gelpy profiles extract "$RUN" --lane 0 --format text
```

```console
{... "overlapping_strips": [[0, 1], [1, 2], ... [18, 19]], "message": "adjacent lane profile strips overlap at this width"}
{"recomputed": ["profile:lane=0"], "reused": ["background"], ...}
{... "files": {"recipe": "recipe.json"} ...}
{"recomputed": [], "reused": ["profile:lane=0"], ...}
```

Two sub-pixel caveats, both real and neither surfaced anywhere. Near an image edge the
strip is silently clamped to the image, so it is narrower than requested (no warning). And the
column bounds are computed with `int()`, which truncates: the extracted strip's centroid sits
between 0.5 and 1.5 px *left* of `x_center`, never on it. Both bounds truncate the same
fractional part, so away from an edge the strip is exactly `width` columns wide and the offset is
exactly `-(frac(x_center - w/2) + 0.5)` — e.g. `x_center=4.0`, `width=3` takes columns 2, 3, 4, a
1.0 px shift. Against a 159 px strip neither matters — but do not go hunting for a 1 px
inconsistency elsewhere.

## Normalization

Three modes. Measured on lane 0 of the example gel, whose raw profile has min -8.97e-4,
max 6.71e-2 and trapezoid integral 10.397 (unit sample spacing):

| mode | what it computes | lane 0 result |
|---|---|---|
| `area` (default) | `p / trapezoid(p)` | integral becomes exactly 1, max 6.45e-3 |
| `min_max` | `(p - min) / (max - min)` | range exactly [0, 1] |
| `safe` | returns the array unchanged | raw column-mean units |

`area` is the default on both `profiles extract` and `fit run`. It multiplies by a constant —
positive whenever the window's trapezoid integral is positive, which is not guaranteed; see the
caveat below — so it changes scale, not shape. Lanes loaded with different amounts of material
overlay comparably, and any quantity that is a ratio of two integrals over the same window is
unaffected by it.

`min_max` also subtracts the minimum, which is an affine map rather than a scaling. It moves
the baseline and discards absolute scale. Useful to make a plot legible; not what you want
under a number you intend to report.

`safe` is the no-op escape hatch, and what it guards against is the other two modes dividing
by a data-dependent quantity. Each has a degenerate case:

- `min_max` on a perfectly flat profile returns all zeros (span is 0).
- `area` divides by a *signed* trapezoid integral. Background-plane subtraction does leave
  negative samples — lane 0's minimum is -8.97e-4 — so a narrow window can integrate to
  something near zero or negative, and a negative denominator flips the profile's sign.
  `normalize_profile([-1, -2, -3, -4], "area")` returns `[0.133, 0.267, 0.400, 0.533]`:
  positive. The only guard is the exact `area == 0` case, which returns the profile unchanged.

So: use `safe` when you need raw intensity units, or when the window you are fitting could
integrate to near zero. An unknown mode name never reaches this code: the schema types it as
`Literal["area", "min_max", "safe"]`, so it is rejected as a `validation_error` the moment the
request is built, whether or not `--plot` is given: `profiles extract` carries the option as
`ExtractProfilesRequest.profile`. The error's `available` lists the three names.

## Slicing

`--slice START END` is `profile[START:END]`: START inclusive, END exclusive, plain Python
slice semantics. `--slice 200 900` therefore yields 700 samples. The schema enforces
`slice_start >= 0` and `slice_end > slice_start`; with no slice given, the window is the whole
profile.

Two consequences.

**Slice and normalization are not independent.** Slicing happens first, normalization runs over
the surviving window. On lane 0, `--slice 200 900` cuts the `area` denominator from 10.397 to
3.720 (a factor 0.358) and scales every plotted value up accordingly. Cropping out a bright
well is usually exactly what you want, but it means normalized values — and area fractions —
are only comparable between lanes sliced the same way. The plot's x-axis is offset by
`slice_start`, so pixel coordinates still line up with rows in the image.

**Changing them is cheap.** What gets cached is the full, unsliced profile. Slice and
normalization are array operations applied after that array is loaded, and are deliberately
excluded from `profile_hash` — they feed `fit_hash` instead. Re-slicing re-runs the fit, never
the strip average:

```bash
gelpy profiles extract "$RUN" --lane 0 --lane 1 --lane 2 --lane 3 --slice 200 900 --normalization min_max --plot --format text
```

```console
{"recomputed": [], "reused": ["profile:lane=0", "profile:lane=1", "profile:lane=2", "profile:lane=3"], ...}
```

On `profiles extract` these two flags affect the PNG only — `ExtractProfilesRequest` carries a
run folder, a lane selector and `allow_raw`, nothing else, so nothing about them is persisted.
The per-lane profile config that *is* persisted, in `fit_config.json`, is written by `fit run` and
`fit refit-lane`.

## Plots without fitting

`profiles extract --plot` is the way to look at lanes before committing to a fit — comparing
a sample against its fractions, or checking whether a smear is real before choosing a slice.

The PNG name follows one rule, and it branches on the **selector**, not on how many lanes that
selector matches: `profiles_overview.png` when `--lane` is omitted entirely (or `--lane all` is
the only term), otherwise `profiles_<label>.png` where `<label>` is the label of the lowest-index
selected lane (not the first `--lane` you typed) with every non-alphanumeric character replaced by
`_`. Listing all 20 lanes explicitly is therefore *not* the overview: it writes `profiles_A.png`.
One call per group gives one file per group:

```bash
printf '%s\n' A '#1' '#2' '#3' B '#1' '#2' '#3' C '#1' '#2' '#3' D '#1' '#2' '#3' E '#1' '#2' '#3' > "$RUN/labels.txt"
gelpy lanes set "$RUN" --labels-file "$RUN/labels.txt" --format text
gelpy profiles extract "$RUN" --lane 0 --lane 1 --lane 2 --lane 3 --plot --format text
gelpy profiles extract "$RUN" --lane 4 --lane 5 --lane 6 --lane 7 --plot --format text
ls "$RUN"/render
```

```console
fits
lane_detect.png
profiles_1.png
profiles_A.png
profiles_B.png
profiles_overview.png
```

`profiles_1.png` is debris from the slicing block above, which ran while lane 0 still carried the
`init` default label `1`. The filename records the label at render time and nothing cleans it up,
so a run folder accumulates PNGs named after labels that no longer exist.

Relabelling lanes does not invalidate anything: the label is excluded from `profile_hash`, so
the extractions above are cache hits (`"reused": ["profile:lane=0", ...]`) even though the
lanes were just renamed.

The name encodes neither the normalization nor the slice, so re-running the same group with
different settings overwrites that group's PNG in place. That is convenient when you are
iterating and a trap when you wanted to compare two settings side by side — copy the file out
between calls.

**Select grouped plots by index, not by label.** Fraction labels repeat across groups by
construction, and a literal label term matches all of them:

```bash
gelpy profiles extract "$RUN" --lane '#1' --plot --format text
```

```console
{"recomputed": [], "reused": ["profile:lane=1", "profile:lane=13", "profile:lane=17", "profile:lane=5", "profile:lane=9"], "files": {"profiles_overview": ".../render/profiles__1.png"}, ...}
```

Five lanes from five different groups, in one plot named `profiles__1.png`. Nothing is wrong
from gelpy's point of view — the selector did what it says. Use explicit indices (or make
labels unique) whenever a plot is meant to be one group.

## Known limitations

- Edge clamping is silent: a lane whose strip runs off the image gets a narrower strip, with
  no warning. Only a strip entirely off-image raises.
- With `--normalization safe` the plot's y-axis label still reads `normalized to safe`. The
  values are unnormalized column means.
- The result key for the rendered PNG is `profiles_overview` even when the selection is a
  subset and the filename is not. Read the value, not the key.
- Lanes must be vertical in the image. There is no image rotation or transpose option.
- The fit engine assumes the well is at the low-index end of the window (first `well_frac`,
  default 0.08). A gel imaged wells-at-the-bottom gets its well mis-labelled and its baseline
  protected at the wrong end; extraction itself is unaffected.

Flags and field lists: `gelpy profiles extract --help`,
`gelpy schema LineProfileConfig`, and [reference/cli.md](../reference/cli.md).
