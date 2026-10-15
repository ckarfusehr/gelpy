---
title: Migrating from gelpy 0.x
audience: both
summary: What each Gel-object method became, why the peak-tuning arguments have no successor, and the results.csv column renames that will break downstream pandas code.
subject:
  - src/gelpy/cli/app.py
  - src/gelpy/schemas/results.py
subject_digest: a505cc9b7bb84777
---

## Read this first

The `Gel` object is gone. Not deprecated, not aliased, not importable:

```console
$ python -c "from gelpy import Gel"
ImportError: cannot import name 'Gel' from 'gelpy' (.../gelpy/__init__.py)
```

`gelpy/__init__.py` binds nothing but `__version__`, and the modules the object lived in
(`package_api`, `image_handling`, `line_profile_handling`, `profile_fit_handling`,
`profile_fitting_models`) no longer exist. So **`.pkl` gel objects written by `gel.save()`
cannot be loaded** — unpickling them looks for a module that isn't there:

```console
ModuleNotFoundError: No module named 'gelpy.package_api'
```

There is no converter and there will not be one: a pickled `Gel` held a mutated image array
and fit parameters, not the intent that produced them, so the only faithful port is to re-run
the analysis. That is cheaper than it sounds — one command sequence, below.

If you need a 0.x number reproduced exactly, install 0.x in a separate environment from
`git`. Do not expect 1.0 to reproduce it; see [The numbers will move](#the-numbers-will-move).

## Your notebook, as commands

A 0.x notebook cell looked like this:

```python
# gelpy 0.x -- for reference only, none of this exists in 1.0
gel = Gel("./data/folding_screen.tif")
gel.setup_gel(labels=labels, x_label_pos=(150, 4485, 20), remove_bg=True)
gel.show_line_profiles(select_lanes="all", slice_line_profile_length=(50, 600), fit="emg")
```

The 1.0 equivalent. Set the two variables once — the documentation test sets the same two
itself, which is why this block is skipped:

```bash
# doc-test: skip
export RUN=/tmp/gelpy-migration
export IMAGE=examples/data/folding_screen.tif
```

```bash
gelpy init "$RUN" "$IMAGE" --n-lanes 20 --format text
gelpy lanes detect "$RUN" --n-lanes 20 --apply --format text
gelpy background remove "$RUN" --format text
gelpy image show "$RUN" --kind corrected --lanes --format text
```

Two files to look at before fitting anything, and they show different things.
`$RUN/render/lane_detect.png`, from `lanes detect`, is the detection overlay: fitted centers on
the gel, plus the column projection with the detected peaks and the grid marked underneath, so
"what was found" vs "where lanes were placed" is one glance. `$RUN/render/corrected_lanes.png`,
from `image show --lanes`, draws the profile-strip width markers on the background-corrected
image — that is the check that each strip covers its band and not its neighbor. 0.x did draw
the strips on the adjusted image, so the second one is not new; what is new is that the
positions were *measured* from the image, so the overlay checks a measurement rather than your
own guess, and `lanes detect` reports a `confidence` flag alongside it.

```bash
gelpy profiles extract "$RUN" --lane 1 --lane 2 --slice 50 600 --plot --format text
gelpy fit run "$RUN" --lane 1 --lane 2 --slice 50 600 --plot --format text
gelpy results show "$RUN" --format table
```

Two lanes here to keep the documentation test quick. Drop `--lane` entirely to fit all 20.

## Method-to-command map

| gelpy 0.x | gelpy 1.0 |
|---|---|
| `Gel("img.tif")` | `gelpy init "$RUN" img.tif --n-lanes N` |
| `setup_gel(x_label_pos=(left, right, n))` | `gelpy lanes detect "$RUN" --n-lanes N --apply` |
| `setup_gel(labels=[...])` | `gelpy init`/`lanes set`/`lanes detect --labels-file FILE` |
| `setup_gel(line_profile_width=W)` | `gelpy lanes set "$RUN" --width W` (or `init --width`) |
| `setup_gel(label_rotation=45)` | `gelpy lanes set "$RUN" --label-rotation 45` (default is now 90) |
| `setup_gel(gamma=, gain=, intensity_range=)` | `gelpy contrast set "$RUN" --gamma --gain --intensity-low --intensity-high` |
| `setup_gel(remove_bg=True)` | `gelpy background remove "$RUN"`. Optional in 0.x; from 1.1 `profiles extract`, `fit run` and `fit refit-lane` refuse a run without it (`background_required`), so a ported script that skipped it needs this line, or `--allow-raw` to measure the raw image on purpose. |
| `setup_gel(bg_model="2d_plane_fit")` | `gelpy background remove "$RUN" --model plane_ransac` — that model's docstring says it ports the legacy `PlaneFit2d`. The 1.0 default is `robust_plane`. |
| `show_adjusted_images()` / `show_raw_gel()` | `gelpy image show "$RUN" --kind raw\|linear\|non_linear\|corrected` |
| `show_line_profiles(select_lanes=[...])` | `gelpy profiles extract "$RUN" --lane ... --plot` |
| `show_line_profiles(slice_line_profile_length=(a, b))` | `--slice a b` on `profiles extract` and `fit run` |
| `show_line_profiles(normalization_type="area")` | `--normalization area` (still the default) |
| `show_line_profiles(fit="emg")` | `gelpy fit run "$RUN"` — see [below](#the-tuning-arguments-have-no-successor) |
| `show_line_profiles(fit=..., maxima_prominence=, peak_width=, sigma=)` | nothing. The engine no longer needs them. |
| `gel.LineFits.fit_model.fit_df` | `$RUN/results.csv`, or `AnalysisResult.to_dataframe()` |
| `show_line_profiles(save_df=True)` → `fitted_selected_line_profiles.csv` | `results.csv` is written by every `fit run` / `fit refit-lane`; no flag needed |
| `save_overview=`, `save_fits=`, `plot_fits=`, `show_df=`, `show_overview=` | `--plot` on the command that produces the figure; everything lands in `$RUN/render/` |
| `img_height_factor=`, `title_label_spacing=` | no equivalent. Figure geometry is not configurable. |
| `show_ladder=` | no equivalent. The bundled ladder images were deleted; MW calibration is unbuilt. |
| `gel.save(name)` / `Gel.load(path)` | share `$RUN/` — see [Sharing](#sharing-a-run-folder-instead-of-a-pickle) |

Every option above is in [reference/cli.md](../reference/cli.md), or live from
`gelpy <command> --help`. Two of the commands with no 0.x counterpart are worth knowing:
`gelpy status` (what state is this run folder in — pure assembly over the recipe, manifest
and fit-config store, nothing recomputed) and `gelpy fit refit-lane` (redo one lane, reusing
every upstream cache; in 0.x the equivalent was calling `show_line_profiles` again, which
re-extracted and re-fitted from scratch).

### Hand-typed lane positions are what detection replaces

`x_label_pos=(left, right, n)` was passed straight to `np.linspace`, so it asserted that
lanes were evenly spaced between two pixel columns you read off the image by eye. `lanes set
--linspace left right n` still does exactly that, and it is still the right tool when
detection fails — but it should not be your first move.

On `folding_screen.tif`, the hand-typed `(150, 4485, 20)` from the 0.x notebook against what
`lanes detect` measures from the image: the error is not random, it is a pitch error of
0.27 px per lane, growing monotonically from -0.7 px at lane 0 to +4.4 px at lane 19.

The strip width is guessed too, and that guess is not a 0.x relic: `image_width /
(n * 2.5)` is still what `gelpy init` uses when you give it no `--width`, and it returns 93 px
on this gel. `lanes detect` replaces it with 159 px, 0.7 of the measured 228.4 px pitch. The
0.x notebook for this gel sidestepped the guess by hand-typing `line_profile_width=180`.
Neither the pitch nor the width is knowable by eye, and nothing in 0.x told you a hand-typed
number had drifted.

If detection returns `confidence: "low"`, or the overlay looks wrong, the fallbacks are
`lanes set --endpoints`, `lanes set --x-center` (repeatable), and the interactive
`lanes pick`. Details in [lane-placement.md](lane-placement.md).

## The tuning arguments have no successor

`maxima_threshold`, `maxima_prominence`, `peak_width` and `sigma` were removed in 1.0 with no
replacement and no alias, and so was `fit="gaussian"|"emg"` as a way to pick one shape for a
whole gel. A ported 0.x script that turns those kwargs into flags stops at argument parsing
with exit code 2 — the first 1.0 CLI did carry them, and they were deleted along with the
tuning arguments themselves:

```bash
# doc-test: expect-failure
# --sigma was removed in 1.0; this is what a ported 0.x script does. Exit code 2.
gelpy fit run "$RUN" --lane 1 --sigma 10
```

```console
{
  "error": "No such option: --sigma",
  "code": "usage_error",
  "detail": "Run `gelpy fit run --help` to see its arguments.",
  "available": [
    "--allow-raw",
    "--auto-escalate",
    "--baseline",
    "--format",
    "--lane",
    "--method",
    "--no-plot",
    "--normalization",
    "--plot",
    "--poly-order",
    "--seed",
    "--shapes",
    "--slice"
  ]
}
```

This is deliberate. In 0.x those four arguments went straight into peak *detection*:
`sigma` smoothed the profile with `gaussian_filter1d`, then `maxima_threshold`,
`maxima_prominence` and `peak_width` became the `height`, `prominence` and `width`
arguments of `scipy.signal.find_peaks`, and whatever came back was fitted. You chose the
peak count implicitly, per lane, by tuning until the maxima count looked right. In 1.0 the
peak count is a model-selection outcome: components are added greedily
against the residual and the growth stops on AICc, each component's shape is chosen
per-component from the registered set, and near-duplicates are collapsed. Thresholds still
decide the component count — an AICc margin per added component, a collapse distance, and a
seeding floor: a residual bump must exceed `min_snr * noise_sigma` to seed a new component,
`min_snr` defaulting to 3.0. But those are the engine's own margins, and `min_snr` lives in
the editable `fit_config.json`, not on a `fit run` flag. What is gone is the per-lane knob.
See [fit-engine.md](fit-engine.md).

The knobs that do exist control different things, and reaching for them by reflex is a
mistake: `--method` picks a strategy tier, `--shapes` restricts which shapes are allowed,
`--baseline`/`--poly-order` control the 1-D baseline under the peaks, `--auto-escalate`
climbs tiers on low-confidence lanes. The 0.x ritual of a loose pass followed by a tight pass
has no meaning here; the 1.0 loop is *fit everything, read the per-lane `confidence`, refit
the stragglers*.

## results.csv columns

Every 0.x column name except `label` and `band_number` changed. The 0.x frame was
`["selected_lane_index", "label", "band_number", "relative_area", "maxima_position"]` plus
one column per shape parameter. Today:

| 0.x column | 1.0 column | also changed |
|---|---|---|
| `selected_lane_index` | `lane_index` | 0.x numbered lanes *within the selection*; `lane_index` is the lane's own zero-based index in the layout, so it is stable whichever lanes you fit |
| `relative_area` | `area_fraction` | different denominator — see below |
| `maxima_position` | `maximum_position` | still measured in the fitted profile, so still relative to `--slice` |
| `Amplitude`, `Mean`, `Standard Deviation`, `Lambda` | `param_amplitude`, `param_center`, `param_sigma`, `param_tau` | lower-cased and prefixed, and which ones appear depends on which shapes were fitted. For EMG this is not only a rename: 1.0's EMG is area-preserving (`area = amplitude * sigma * sqrt(2*pi)` for every `tau`) with `tau` a *signed* tail time-constant, while 0.x's `Lambda` was an inverse rate dividing the amplitude, so the numbers are not comparable. Only `Mean` → `param_center` ports as-is. |

`grep` for the old names in the current source returns nothing, so there is no compatibility
shim to find. In your downstream code, rename those three, re-map the per-shape parameter
columns, and expect new columns you did not have: `component_class`, `shape`, `area`
(absolute, not a fraction), `area_uncertainty`, `residual_fraction`, `confidence`, `status`,
`method`. Full definitions in [results.md](results.md).

`AnalysisResult.to_dataframe()` returns the same column *set* as `results.csv` whenever some
lane produced bands, but the CSV writer imposes a fixed leading column order and the dataframe
follows insertion order — sort or select by name, not by position. (The CSV header always
carries the full fixed set, so a run where every lane failed writes the seven band columns
empty; the dataframe omits them.) [`examples/downstream_analysis.ipynb`](../../examples/downstream_analysis.ipynb)
is a worked `results.csv` → pandas → figure example.

## The numbers will move

Do not expect a 1.0 `area_fraction` to equal a 0.x `relative_area`. Three reasons, all
structural:

- **Different denominator.** 0.x divided each component's area by the sum of the fitted
  component areas, so `relative_area` summed to exactly 1 per lane by construction — a fit
  that missed half the signal still reported fractions summing to 1. 1.0 divides by the
  integrated baseline-subtracted signal, `∫(y − baseline)`, which is measured from the
  profile and not from the fit. It does not sum to 1: on lanes 1 and 2 of the example gel it
  sums to 1.032 and 1.024. The two are not on the same support — `area` is each shape's
  closed-form integral over the whole real line, the denominator is a trapezoid over the
  slice — so a component pinned at the slice edge contributes its whole analytic area while
  only half of it lies in the window. That is exactly what these two lanes do: both carry a
  Gaussian centered on x = 549, the window's last index (0.0338 analytic vs 0.0169 inside on
  lane 1). Re-integrating every component inside the 550-px window instead gives 0.997 and
  1.001, so here the excess is edge overhang, not double-counted overlap. Overlap does the
  same thing when it happens, which is why the engine reads a coverage above 1.10 as
  "overlap/over-fit" and flags the lane. `residual_fraction` is `1 − Σ area_fraction`, so it
  goes negative on both these lanes.
- **A baseline is subtracted before fitting**, and 0.x fitted the profile as it stood.
- **The lanes are in different places.** Different centers and a different strip width
  integrate different pixels — 180 px hand-typed in the 0.x notebook for this gel against the
  159 px `lanes detect` measures.

Treat a 0.x-to-1.0 comparison as two measurements, not as a regression test. If they disagree
by more than a few percent, the per-lane fit figures under `$RUN/render/fits/` will show you
which one to distrust.

## Sharing: a run folder instead of a pickle

`gel.save()` served two purposes — resuming your own work, and sending the analysis to
someone else. The run folder covers both, and a collaborator does not need Python to open it:
`recipe.json` is the editable intent, `results.csv`/`results.json` the numbers, `report.md`
and `render/*.png` the figures. Re-running any command against the folder reproduces the
analysis, and re-running the unchanged parts is a few sha256-over-small-JSON ops plus a
cached-array read: under the same gelpy version it never re-decodes the image or re-runs the
background fit. Under another version the first run recomputes every stage, because the version
is part of every hash.

The source image is referenced by path and sha256, not copied in, so send the image alongside
the folder — or check both into the project. [concepts.md](concepts.md) explains the caching
and what invalidates what.
