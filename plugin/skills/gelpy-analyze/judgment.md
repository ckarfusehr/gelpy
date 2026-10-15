# Judging a gel analysis

The judgment part of the gelpy-analyze skill: what to check and how to decide. It does not depend
on how gelpy is driven -- its command line or its MCP tools -- so it applies unchanged to either.
Parameters are named by their fields in gelpy's data contracts (the JSON Schemas of its requests);
SKILL.md maps every step to a command and a tool.

## Ground truth comes from the user

- **The lane count and the labels come from the user or a sample sheet, never from the image.**
  Detection anchors that many lanes to the image; it does not count them. If neither the user nor a
  sheet gives them, ask.
- **Labels are used verbatim**, left to right: never renamed, expanded, merged or suffixed. If one
  looks wrong, ask. An empty well still needs a placeholder label such as `(empty)`, or every label
  after it shifts by one lane.

## Check how the image was read

Creating a run reports how the image was interpreted: channel, polarity, page. If it warns that the
polarity is low-confidence, look at the raw view before anything else: bands must be bright. If
they are dark on light, set the polarity explicitly (`invert`) and look again. For a multi-page
source, check that the page in use is the one you want.

## The one rule: verify lane positions against the image

Gels are cropped differently every time, so pixel positions never transfer between images, and
spreading N lanes evenly over the whole frame is wrong whenever the gel does not fill the crop.
Place lanes by detection, then **look at the detection overlay before any measurement**: every
line must sit on a lane. This step decides more than any other whether the numbers mean anything.

How detection works, so that its failures make sense: the image is collapsed along y into a
projection (loaded lanes are bright columns, so peaks), and an even N-lane grid is fitted to it by
a matched filter with a soft prior toward `width / N`. It uses what is known -- N, and that a comb
spaces lanes evenly -- instead of trusting the brightest lanes. It reports `confidence` (`ok` or
`low`) and, when low, a `confidence_reason`. Read the reason first; it usually names the problem
("25 peaks detected, expected 20", "only 45% of lane positions have a nearby detected peak").
Detection runs on the raw image by default: dark-background fluorescence gels detect cleanly that
way, and background subtraction can add noise to the projection.

Known limitation, by design: the `width / N` prior assumes the lanes span roughly the whole crop.
A gel that fills only part of the width comes back `low` although nothing is wrong with the image.
Use a fallback below; do not change the lane count or repeat the same detection.

If the overlay is wrong or confidence is `low`, resolve it, cheapest first:

- **Read the outer lane centers off the overlay yourself** and set them as `endpoints` together
  with the lane count (`n_lanes`); the lanes in between are spaced evenly. This is the normal
  recovery for an agent working alone. Afterwards look at an overlay of the lanes as set, not of a
  fresh detection: a new detection ignores what you set.
- **Faint lanes not detected:** restrict the projection band to the well region (`y_end`, roughly
  the top 40% of the height) and/or lower the prominence floor (`min_prominence`, for example 0.03).
- **Narrow spikes detected as lanes** (dust, artifacts: narrow peaks in the projection panel, where
  a lane is a wide plateau): compare their widths with the genuine peaks' in the same overlay and
  set `min_width` between the two. There is no safe default; read the widths off this image.
- **Uneven illumination drowning the lanes:** detect on the background-corrected image
  (`use_corrected`) after configuring background removal.
- **Irregular spacing:** set every center explicitly (`x_centers`).
- **A person wants to place the lanes by eye:** gelpy has an interactive picker; it needs a display.

Repeat until every line sits on a lane. Record where manual positions came from (the overlay), so
the run can be reproduced.

## Clean views and QC views are different images

The sample view (linear or gamma contrast, labels on top) never carries the lane-strip overlay.
The overlay is a QC view of where gelpy measures: for your own check, or for someone who asks how
the lanes were sliced. Never hand the overlay over as "the gel image".

Labels are drawn vertically (90 degrees). Sample names are usually longer than the lane pitch, and
any smaller angle makes them collide; keep 90 unless the user asks for another angle.

## Background removal comes before any profile or fit

Normalizing raw profiles by area crushes a strong lane's background floor toward zero and inflates
a faint lane's, so lanes get different, meaningless baselines although the background is the same.
Subtracting the background first is what makes lanes comparable, and gelpy refuses profiles and
fits without it unless raw measurement is requested on purpose.

`robust_plane`, the default, fits a low-order surface by iteratively reweighted least squares; it
needs no hand-placed stripes and stays unbiased when bands and smears cover much of the image.
Check it in the background diagnostic (surface, weights, before and after slices). `plane_ransac`
is a fallback only, for when `robust_plane` visibly fails on an unusual image.

## Judging a per-lane fit

A lane's fit plot has two panels. Left: the profile, the estimated baseline (dashed) and the summed
model, which should overlap. Right: the baseline-subtracted signal split into components colored
by class, each with its area fraction; the coverage should read close to 100%.

`status: "ok"` only means the fit converged. **`confidence` and its reasons (`confidence_reasons`;
`reasons` in the per-lane summary a fit returns) are the quality signal.** Map a `low` lane to an
action:

- "coverage X%: Y% of the signal is unmodeled": real signal was missed; widen the slice window or
  escalate.
- "fitted peaks exceed the signal (overlap/over-fit)": escalate to `method = bayesian`.
- "poor fit (R2=...)": the shapes do not capture the profile; escalate.
- "band N (class) width ran to a bound": one component grew to its widest allowed width, which
  means a broad "peak" is absorbing baseline drift or a slope across the window; that band's area
  and every area fraction of the lane are not trustworthy. A higher tier does not help. Narrow the
  slice window to the bands of interest and/or lower the baseline order (`poly_order`; 0 on a short
  window), refit the lane, and check that the dashed baseline now sits below every bump.
- a band's "area unreliable" or "overlaps a neighbor": that split is uncertain; `bayesian` gives
  credible intervals, or accept the band as lower-confidence.

The strategy ladder is `grow_select` (the default), then `sparse_seed` (heavy overlap), then
`bayesian` (uncertainty). Escalate only the low-confidence lanes; the cache refits only those.

A lane can report `confidence: "ok"` while its baseline silently absorbed a real feature. Always
check the left panel: the dashed baseline must sit below every visible bump, never on top of one.
Closely spaced real bands (a shoulder) are split into separate components; a smear stays one broad
component, whose area fraction is "how much sample is in the smear".

## A well-cavity dip is not signal

The well is a cut-out with less gel material than the lane around it, so it reads as a dip below
the baseline, never as a peak. Its minimum and its return above the baseline sit within a few
pixels of each other on every lane of a gel: the same wells give the same artifact.

Material too large to enter the gel shows up as a distinct peak right after that dip, where the
profile crosses back above the baseline; that peak's area is the "trapped in the well" quantity.
Everything before it -- the dip and anything to its left -- is not part of the gel's data.

- **Recognize it:** at the start of a lane's profile, a dip that bottoms out and crosses back above
  the baseline within the first 10-15% of the window, before any band. A gradual slope is baseline
  drift, not this. Find it on one or two lanes and use the same boundary for the whole gel.
- **Exclude it** with the profile slice window: `slice_start` just past the dip's crossing, applied
  to every lane. Do not try to fix it with the baseline's well settings (`BaselineConfig` fields
  `well_frac` and `protect_well`): masking more of the window gives the baseline more freedom to
  bend up elsewhere, and it then swallows more signal.
- **Watch the side effect:** once the dip is cut, an order-2 polynomial baseline on a short window
  can bend up under a real smear or shoulder right after the cut and absorb it silently, often with
  `confidence: "ok"`. The tell: the dashed baseline sits on top of a visible bump. The fix is a
  constant baseline (`poly_order = 0`) on that slice. It cannot follow genuine drift, so use it only
  on a window that starts right after a hard boundary like this one, and compare order 2 and order
  0 on a couple of lanes before choosing for the whole gel.
- **It is not universal.** Some crops leave the well out, or show no dip. Look for the pattern
  before slicing.

## Several images

Detect lanes on every image separately, even for rows of the same experiment: crops differ, so
another image's pixel positions do not transfer.
