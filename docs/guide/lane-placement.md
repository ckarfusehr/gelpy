---
title: Lane placement
audience: both
summary: How lane centers are fitted to the image rather than guessed, and the recovery ladder when the overlay looks wrong.
subject:
  - src/gelpy/core/lanes.py
  - src/gelpy/schemas/lanes.py
  - src/gelpy/cli/_pick.py
subject_digest: a3ae25a08c3b7f15
---

Every downstream number depends on this step. A lane profile is the column mean over a strip
centered on `x_center_px`; if the strip sits between two lanes, the fit is of a mixture and the
areas are meaningless. Nothing later in the pipeline can detect that.

## Why the centers have to come from the image

Two things are known before you look at the gel: the lane count `N` (from the plate sheet), and
that the wells came from a comb, so the true centers lie on a uniform-pitch grid. Nothing else is.

- **Pixel positions do not transfer between images.** Crops differ every time. A center list
  copied from last week's run is a coincidence, not a measurement.
- **Spreading `N` lanes evenly across the frame is not a placement.** It is right only when the
  crop happens to be tight to the loaded lanes. `gelpy init` does exactly that as a placeholder
  (`compute_lane_positions`, 5% to 95% of the image width), and on the 20-lane
  `folding_screen.tif` example it lands **84.4 px = 0.370 x pitch** off the true left-edge
  center -- past the quarter-pitch point where the strip starts eating the neighbouring lane.
  Fitted detection on the same image lands 4.44 px = 0.019 x pitch off.

So `init` gives you a run folder, not a layout. `lanes detect` gives you the layout.

Set the two variables the blocks below use once. This block is skipped by the test that executes
the rest, because that test sets both itself:

```bash
# doc-test: skip
export IMAGE=examples/data/folding_screen.tif
export RUN=./run-lanes
```

```bash
gelpy init "$RUN" "$IMAGE" --n-lanes 20
gelpy lanes detect "$RUN" --apply
```

```text
{ "centers": [149.30526315789496, ..., 4489.438596491229], "pitch": 228.42807017543862,
  "profile_width": 159, "n_detected": 20, "n_expected": 20, "confidence": "ok",
  "confidence_reason": "", "projection_source": "full", "y_band": null, "applied": true,
  "overlay": "render/lane_detect.png" }
```

## What the fit actually does

1. **Collapse rows to a 1-D signal.** `column_projection` takes the plain unweighted mean down
   the rows, so a loaded lane becomes a bright bump along x. A thresholded / height-weighted
   column score was tried and reverted: on real gels it either saturated identically for a lane
   and the trough beside it, or let one bright lane suppress the dim ones after normalization.
2. **Register an `N`-tooth comb against it.** Not against the raw column mean: the projection
   is min-max normalized and low-passed with the pitch-scaled moving average described below
   before the comb ever sees it, so that window sets the fitted pitch as well as the peaks.
   `_matched_filter_pitch` then scores every `(offset, pitch)` on a coarse grid by summed tooth
   intensity times a Gaussian prior on pitch centered at `width / N`: search band 0.6 to
   1.3 x `width / N`, step 0.25 px, prior sigma 0.08 x `width / N`.
3. **Keep the pitch, throw away the offset.** The offset landscape is nearly degenerate and a
   few empty interior wells can flip it by half a pitch. The origin is anchored to the detected
   peaks instead (the discrete left-shift that best aligns the grid), then refined by a
   least-squares fit gated to peaks within 0.25 x pitch of a node -- and that refinement is
   discarded if it moves the origin by more than half a pitch or the pitch by more than
   0.2 x pitch.

The pitch prior is insurance for the non-uniform gels, not a general necessity. Recomputing the
prior-free (raw comb) score: on 22 of the 25 synthetic conditions, and on `folding_screen`
(prior-free argmax 228.14 px against a labeled pitch of 228.16), it already peaks at the true
pitch, so the prior costs nothing there. It collapses toward the bottom of the search band only
where brightness is strongly non-uniform -- `faint_right_cluster` (72.5 px against a true 116.7)
and `gradient_bg` (70.0 px, the band floor, and the only condition whose raw score is monotone in
pitch). The mechanism is not a monotone bias: the tooth count is fixed at `N`, so shrinking the
pitch raises the sum only when the bright sub-region is compact enough to take all `N` teeth.
This is the failure the old median-peak-gap estimator had on a gel whose right third is faint.
(`partial_fill` lands on the band floor too, 70.0 px against a true 64.2 -- there the prior-free
answer is the *better* one; see limitations.)

**Detected peaks are votes on phase, not lane positions.** The grid comes from `N` plus one
pitch; the peaks only pin its origin, so losing a quarter of them barely moves it:

```bash
gelpy lanes detect "$RUN" --prominence 0.15 --projection-strategy full --no-plot
```

reports `n_detected: 15` for `n_expected: 20` and still places the grid within 6.4 px
(0.028 x pitch) of the hand-labeled centers. An empty interior well lands on an unmatched grid
node.

## Projection strategies, and why the default cannot be worse

How rows are collapsed to that 1-D signal is a registry with two entries, `full` and `d4_union`
(both in [reference/registries.md](../reference/registries.md#lane-projection-strategies) -- this
is the one registry `gelpy models list` does *not* cover; that command handles fit strategies,
peak shapes and background models). Detection uses `d4_union` by default and falls back to the
full-height mean.
<!-- claim: default-projection-strategy-is-d4-union -->

`d4_union` is for gels where no single y-window holds a band from every lane -- a ladder or
scaffold lane migrating to a different depth than the samples. It scores each row by
comb-periodicity (an in-phase `N`-tooth comb minus an anti-phase one, so a lone dust spike hits
at most one tooth and a flat row scores ~0), thresholds at `median + 3 x 1.4826 x MAD`, clusters
the surviving rows into candidate bands, and greedily unions the bands that raise all-`N` node
coverage. The coverage gate is what keeps it honest: a band lighting up only the bright lanes
does not raise coverage, so it is skipped even when its own score is the highest.

Adoption is decided per image by comb-SNR (mean normalized height at the `N` grid nodes minus the
mean at the anti-phase midpoints). The candidate must beat full height by more than 0.02 comb
units **and** must not turn a full-height `"ok"` into a `"low"`. A tie loses. That is why the
default is never worse than `full` -- and why on `folding_screen` it is not adopted at all:

```bash
gelpy lanes detect "$RUN" --projection-strategy full --no-plot
```

returns centers identical to the default run above, and `"projection_source": "full"` in both.
Measured cost of the difference on this 4688 x 1364 image, warm cache and `--no-plot` both
times: 4 s pinned to `full` against 16 s with the default, i.e. about 4x (the absolute times are
machine- and load-dependent, the ratio is not). The default pays one matched-filter sweep for
its pass-1 pitch plus a full detection per candidate band -- 5 sweeps on this gel, 11.4 s of the
strategy's 11.5 s. The overlay adds under a second to either: `lanes detect` draws it from the
detection it already ran and does not run it again. Pinning `full` on a gel you already know
is single-band is free.

Comb-SNR being computed on the min-max-normalized projection means it rewards a *clean*
projection, not a high-contrast one: averaging fewer rows leaves more per-column noise, which
inflates the normalization span. Full height wins whenever one band already spans every lane.

## The smoothing window, and why `--min-width` is not the answer to dust

Before peak detection the projection is low-passed with a moving average 0.7 x the expected pitch
(`width / n_lanes`) wide.
<!-- claim: smoothing-window-fraction -->

That is far wider than any denoising blur, deliberately. A loaded well is a **plateau**, not a
peak, and real wells are often brighter on one edge -- comb-tooth shadow, meniscus, a bubble --
with the bright edge varying well to well. That is per-lane jitter, not a crop-wide offset the
grid fit could absorb, and `find_peaks` on a lightly-blurred plateau locks onto the brighter edge
instead of the center. A window scaled to the known pitch collapses the plateau to one bump
first.

The useful side effect: the same low-pass destroys narrow dust before peak detection runs.
Measured on a synthetic 20-lane, 4688-px projection (default window = `round(0.7 x 234.4)` =
164 px): a unit-height Gaussian band of sigma 61 px (0.26 x pitch) keeps 0.766 of its height
under the average, while a sigma = 2.5 px fleck as bright as a band keeps 0.038
(= sqrt(2 pi) x 2.5 / 164). Standing alone in a flat field that is already under the 0.05
default prominence floor; dropped between two bands it is not even a local maximum any more, and
detection returns 20 peaks. Under a 5 px blur the same fleck has prominence 0.63 and detection
returns 21 -- a phantom lane. The 20-against-21 split held for every band width I tried,
sigma 40 px through a 126 px plateau.

`--min-width` is therefore **off by default (1 px), and it is not the first move against dust.**
It only bites on a peak that already survived a 164 px average, i.e. something genuinely wide.
There is also no safe scaled default for it: on a real gel a faint band can be exactly as narrow
as a dust fleck, so a width floor chosen without looking at *this* image's overlay silently
rejects real lanes. Raise `--prominence` first.

## Verify the overlay. Every time.

`lanes detect` writes `render/lane_detect.png` unless you pass `--no-plot`: the gel at a fixed
1-99.5 percentile stretch with a red line per fitted center, above the projection with detected
peaks marked `x` and the fitted grid dashed. A chosen `y_band`, or a manual
`--y-start`/`--y-end` window, is shaded. Check three things:

1. Do the red lines sit on lanes at the left **and** right edge? Phase error shows at the ends
   first.
2. Do the lines line up with the labels you expect? `n_detected` below `n_expected` is fine --
   an empty well yields no peak but still gets a grid node. `n_detected` *above* `n_expected`
   means the projection has structure that is not a lane.
3. `confidence`. `"low"` carries a `confidence_reason` naming the check that fired: too few peaks
   for `N`, more than `N + 2` peaks (dust), a center off-image, low node coverage, or the
   matched-filter pitch pinned at the edge of its search band -- that last one usually means the
   gel does not fill the crop.

`"ok"` means every check passed, not that the answer is right. On the synthetic suite the flag
agrees with the truth on 100% of runs and never once says `"ok"` while being wrong (table below),
but it is a heuristic and the overlay is the evidence.

## Recovery ladder, cheapest first

**1. Read the two outer lane centers off the overlay and set them.** The documented fallback, and
almost always enough: a wrong fit is usually only wrong in phase, and the comb constraint does
the rest.

```bash
gelpy lanes set "$RUN" --endpoints 150 4485 --n-lanes 20
```

**2. Focus the projection on the informative rows** with `--y-start`/`--y-end`, if faint lanes
only appear in one migration region. Then verify it helped -- narrower is not automatically
better. On `folding_screen`, restricting to the bright lower product band makes it *worse*,
because that band smiles:

```bash
gelpy lanes detect "$RUN" --y-start 750 --y-end 960 --projection-strategy full --no-plot
```

Measured 12.0 px = 0.052 x pitch, against 4.44 px = 0.019 x pitch for full height. Still `"ok"`,
still well inside the strip, but three times the error from a change that sounds like an
improvement. (Pinned to `full` so the row band is the only variable.)

**3. Raise `--prominence`** if the projection shows peaks that are not lanes. The default is 0.05
on the min-max-normalized projection. Losing real peaks to a high floor costs little (above);
letting dust peaks in can drag the origin.

**4. `lanes pick`, if a human with a display is available.** It shows the image, collects two
clicks (or `--all` for every center) and writes them through the same path as `lanes set`. It
displays the image at the run's *configured gamma contrast*, not the overlay's fixed percentile
stretch, so a very faint gel may want `gelpy contrast set` first. It needs a real matplotlib
backend (MacOSX, QtAgg or TkAgg) and a display. Without one it answers `display_required`, whose
`detail` says to set the lanes with `lanes set --endpoints` or `--x-center`, or to run it on a
machine with a display; a window that is closed (or Enter pressed) before the clicks are done
answers `pick_cancelled`, whose `detail` says to run it again, and nothing is changed. So it is not
for agents or headless runs, which is why this block is not executed:

```bash
# doc-test: skip
gelpy lanes pick "$RUN" --n-lanes 20
```

**5. Explicit centers, one per lane.** The last resort, and the only option when the spacing is
genuinely not uniform (a bent comb, two combs, a montage):

```bash
gelpy lanes set "$RUN" --x-center 149 --x-center 378 --x-center 606 --x-center 835
gelpy lanes set "$RUN" --endpoints 150 4485 --n-lanes 20
```

The first command sets a 4-lane layout for illustration -- pass all `N` -- and the second puts
the run back. `lanes set` warns, without failing, when adjacent strips overlap at the current
`--width`.

## Measured accuracy

Reproduce with `python -m tests.synthetic.harness`, which prints both rows (400 detections, a few
minutes). 25 conditions x 8 seeds; a run **passes** when the max per-lane center error is within
0.25 x pitch. `medErr` is the median over the 8 seeds of that max error, in pitch units, and the
table row below is the median of the 25 per-condition medians; the four rate columns are means
across conditions.

| detector | medErr | pass | flagOK | falseOK | falseLow | composite |
|---|---|---|---|---|---|---|
| shipped default (`d4_union` + fallback) | 0.02p | 96% | 100% | 0% | 0% | 0.98 |
| `--projection-strategy full` reference | 0.04p | 94% | 82% | 0% | 18% | 0.88 |

`falseOK` = 0% for both is the number that matters: neither detector ever reported `"ok"` on a run
that missed. What `d4_union` bought was the 18% -> 0% collapse in needless manual fallbacks
(`falseLow`), across nine conditions: `snr_015`, `snr_020`, `narrowband_mid`, `narrowband_hard`,
`empty_edge`, `empty_edge_and_interior`, `smile`, `gradient_bg`, `n8_wide`. (The snapshot in
`tests/synthetic/README.md` lists eight of them; `narrowband_mid`, at 25% falseLow for full
height, is missing from that list.)

Worst *single run* of the 200 for the shipped default, excluding `partial_fill`:
**0.097 x pitch**, on `dusty_in_band` seed 7013 -- 30 dust flecks inside the band's own y-range,
brighter than the bands. Under half the quarter-pitch threshold. `d4_union` was adopted on 175 of
the 200 runs; the rest fell back to full height.

Real gel, `examples/data/folding_screen.tif` against hand-labeled centers
(`tests/synthetic/real_labels.json`): **4.44 px = 0.019 x pitch**, `"ok"`.

Sub-pixel phase is not achieved and is not needed. The profile strip is 0.7 x pitch wide (159 px
on this gel) and is a hard integer column window, so a 4.44 px error swaps four columns in at one
edge and four out at the other: 155 of the 159 averaged columns are the same ones, and the shift
is 2.8% of the strip width. The quarter-pitch pass threshold sits inside the geometric
crossover -- a strip misplaced by 0.30 x pitch first touches the neighbouring lane's own
0.7 x pitch extent, and
reaches that lane's center at 0.65 x pitch -- which is where the error becomes real.

## Known limitations

- **A gel that fills only part of the crop breaks the `width / N` prior.** The deliberate trade.
  The synthetic `partial_fill` condition (lanes on ~55% of the frame) is wrong by **4.64 x pitch**
  in *both* detectors, and both correctly flag it `"low"` on 100% of runs. There is no knob for
  it: crop tighter, or set `--endpoints`.
- **The confidence reason `"detected peaks don't sit close to the fitted grid"` cannot fire.** Its
  residual is gated to peaks within 0.25 x pitch of a node and then compared to a 0.3 x pitch
  threshold, so the branch is unreachable. Harmless -- node coverage catches the same failures --
  but do not expect that string.
- **`d4_union`'s band selection ignores your peak-detection overrides.** Its internal trial
  detections run with default `--prominence`, `--distance`, `--min-width` and
  `--smoothing-window`; your values apply only to the final detection on the chosen projection.
  When tuning those, pin `--projection-strategy full` so the knob you turn is the knob that acts.
- The adoption score and per-row weights `d4_union` computes are not carried on the result;
  `projection_source` and `y_band` are all you see of that decision.

## Labels

Lane labels are drawn vertically, 90 degrees, by default.
<!-- claim: default-label-rotation-is-90 -->
Sample names are routinely longer than the lane pitch at any usable `N`, and 45 degrees collides
as soon as labels get long or the lane count grows. Override per run with `--label-rotation`.

Choosing *which* lanes a later command acts on (`re:`, `label:`, indices) is a different
mechanism, documented in [reference/contracts.md](../reference/contracts.md).
