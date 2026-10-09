---
name: gelpy-analyze
description: >
  Analyze a gel-electrophoresis image end-to-end with the gelpy CLI: place lanes correctly
  from the actual image (never guessed/hardcoded positions), remove background, fit per-lane
  band intensities, and produce a browsable run folder (results.csv/json, report.md, per-lane
  fit PNGs). Encodes the verify-as-you-go judgment loop — detect lanes → confirm the overlay →
  fit → inspect each lane → refit stragglers. Use when asked to analyze / quantify / measure
  bands or purity on an agarose or protein gel image (.tif/.png), or to drive gelpy on a new gel.
  Invoked with only a folder (or no explicit ask), it does the default scope ONLY: label the
  lanes and render linear, non-linear and lane-detection images — no profiles, no fitting.
  Also builds the dense summary PDF report when asked for a "summary PDF" / "report".
---

# gelpy — analyze a gel image

gelpy turns a gel image into per-lane band intensities through a content-hash-cached pipeline:
`image → background-corrected image → per-lane line profiles → peak fits → run folder`. Every
command is `gelpy <group> <verb> <RUN_DIR> ...`, takes/returns JSON, and only recomputes what a
change actually invalidates. Run `pip install -e .` (env with the sci stack) if `gelpy` isn't on
PATH; `gelpy --help` lists commands, `gelpy schema <Model>` prints any contract.

## Default scope — what to do when only a folder is given

If the skill is invoked with just a folder (or with no request beyond "analyze this gel"), do
**exactly this and stop**:

1. Find the gel image(s) in the folder (one run per image / gel row; skip byte-identical copies —
   check with `md5`).
2. Get lane count `N` and labels from the user's message or a sample sheet in the folder
   (xlsx/csv/txt). If neither exists, ask — `N` and labels are never inferred.
3. `gelpy init` → `gelpy lanes detect --apply` → **verify the overlay** (workflow step 3) →
   `gelpy lanes set --labels-file`.
4. Render `gelpy image show --kind linear` and `--kind non_linear` (clean, no `--lanes`), and
   the lane-detection QC (`render/lane_detect.png`). If lanes were changed after detection,
   re-render the QC so it shows the applied grid.
5. Copy the three PNGs to `agent_analysis/figures/<row>_{linear,non_linear,lane_detect}.png`.
6. Report the file paths, lane pitch/confidence, and anything odd visible in the images
   (saturation, dust, smiles, unexpected bands) in a few bullets.

**Do not** remove background, extract profiles, fit, or write a report unless asked. The
linear/non-linear renders use the raw image under display contrast and do not need background
removal. Everything beyond this list (profiles, fits, PDF) is opt-in per request.

## Run-folder layout (keep it next to the data)

```
<gel folder>/agent_analysis/
  row1/ row2/ ...          # gelpy run folders (one per image)
  labels_<row>.txt         # labels as given, one per line (empty well -> "(empty)")
  figures/                 # all deliverable PNGs, prefixed with the row name
  results/                 # csv/json of fits or metrics
  scripts/                 # any custom fit / report script used (reproducibility)
  report.pdf               # only when a report was asked for
```
If a row's image lives in a different experiment folder, its `agent_analysis/` goes there.
`lanes set --labels-file` drops blank lines, so empty wells need a placeholder label.

## The one rule that matters most

**Never trust lane positions you didn't verify against the image.** Gels are cropped differently
every time, so absolute pixel positions never transfer between images, and spreading N lanes
evenly across the whole frame is wrong whenever the gel doesn't fill the crop. Always place lanes
with `lanes detect` and **look at the overlay** before fitting. This is the step that most affects
whether the numbers mean anything.

## Workflow

Steps 1-5 are the default scope (above). Steps 6-8 (background, fitting, close-out) run only
when the user asks for profiles, fits or quantification.

Pick a run folder path (keep it next to the data). `N` = the true lane count — get it from the
plate's Excel/CSV or ask the user; it is ground truth, never inferred.

1. **Init + look at the raw image.**
   ```
   gelpy init RUN IMAGE --n-lanes N
   gelpy image show RUN --kind raw        # opens with the current (placeholder) lane overlay
   ```
   Open the PNG. Note roughly where the leftmost and rightmost lanes sit — you'll sanity-check
   detection against this.

2. **Detect lanes from the image.**
   ```
   gelpy lanes detect RUN --n-lanes N --apply
   ```
   `lanes detect` projects the image down the y-axis (loaded lanes are bright columns → peaks),
   then fits an even N-lane grid to that projection using a **matched filter with a soft prior
   toward `width/N`** — not a raw peak-gap estimate. This is what makes it robust to a faint or
   unevenly-loaded gel: it uses the two things you already know (N is ground truth, spacing is
   uniform — it's a comb) instead of trusting whichever lanes happen to be brightest. It writes
   `render/lane_detect.png` and reports `confidence` (`"ok"`/`"low"`) plus a `confidence_reason`
   string when low — read the reason before opening the PNG, it usually tells you exactly what's
   wrong (e.g. `"25 peaks detected, expected 20 (+/- a few) — check for noise/dust"` or `"only 45%
   of lane positions have a nearby detected peak"`). It runs on the **raw** image by default —
   dark-background fluorescence gels detect cleanly that way, and background-subtraction can
   actually add projection noise. Add `--corrected` only for gels with strong uneven illumination
   where the raw background swamps the lanes.

   **Known limitation, by design:** the `width/N` prior assumes the lanes span roughly the full
   crop (the usual case) and tolerates a handful of empty/faint interior wells. A gel that fills
   only a *fraction* of the crop width (large empty margins) fights this prior and will come back
   `confidence: "low"` even though nothing is "wrong" with the image — that's the deliberate
   trade-off for fixing the more common, worse failure (a faint gel's pitch collapsing onto its
   brighter half instead of spanning the true full width). Use one of the fallbacks below for
   that case, not a smaller/larger `--n-lanes` or repeated retries.

3. **VERIFY the overlay — this is not optional.** Open `render/lane_detect.png`. Every red line
   must sit on a lane. If it does and `confidence: "ok"`, continue. Otherwise, resolve it with
   whichever of these fits the situation — cheapest first:
   - **You (the agent) can see the image.** Look at `render/lane_detect.png` (or re-render at
     `--y-end <well region>` for a tighter view) yourself: read the x-position of the leftmost and
     rightmost real lane center off the plot, then set them directly — no human needed:
     ```
     gelpy lanes set RUN --endpoints LEFT RIGHT --n-lanes N
     ```
     This is the normal recovery path for an agent working alone. Re-render (`lanes detect
     --no-apply` or `image show`) and re-check the overlay afterward.
   - **A human is available and wants to place lanes precisely by eye** → the interactive picker
     opens the high-contrast image and takes clicks (needs a GUI/display, not for headless runs):
     ```
     gelpy lanes pick RUN --n-lanes N            # click the two outer lane centers
     gelpy lanes pick RUN --n-lanes N --all      # or click every lane (irregular spacing)
     ```
   - Faint lanes not detected at all → focus on the well region and/or lower prominence:
     ```
     gelpy lanes detect RUN --n-lanes N --y-end <~top 40% of height> --prominence 0.03 --apply
     ```
   - Extra peaks that look like narrow spikes rather than lane-width plateaus in the overlay's
     bottom panel (dust/artifacts) → compare their measured width to the genuine peaks' width in
     the same overlay, then set `--min-width` between the two:
     ```
     gelpy lanes detect RUN --n-lanes N --min-width <between-spurious-and-genuine-width> --apply
     ```
     There's no safe default for this (real-data testing found a genuine faint band can be just
     as narrow as a dust spike on some gels), so it's off by default — always read the widths off
     *this* image's overlay before setting it.
   - Uneven illumination drowning the lanes → add `--corrected` (after a `background remove`).
   - Fully irregular spacing → `gelpy lanes set RUN --x-center C1 --x-center C2 ...` (one per lane)
     or `gelpy lanes pick RUN --n-lanes N --all`.
   Re-render and re-check until the grid is right.

4. **Label the lanes** from the sheet (order = left→right). Labels are **always vertical, i.e.
   90°** — this is the schema default (`LaneLayout.label_rotation` / `InitRequest.label_rotation`
   both default to `90.0`), so plain `lanes set --labels-file labels.txt` already gets it right;
   pass `--label-rotation 90` explicitly only if you're re-affirming it after something else
   changed it:
   ```
   gelpy lanes set RUN --labels-file labels.txt
   ```
   Sample names are usually longer than the lane pitch, at any lane count. At `0` (horizontal)
   they overlap into an unreadable smear; any small in-between angle (e.g. `45`) does too, just
   less obviously — it still collides once labels get long or lanes get numerous. `90` prints
   each label in its own column reading bottom-to-top, so it stays legible regardless of label
   length or `N`. Do not change this default unless the user explicitly asks for a different
   angle — it is not a per-gel judgment call.

5. **Render clean sample views, separately from the QC/lane-placement view.** `image show`
   only draws the semi-transparent red lane-width/center overlay when asked (`--lanes`) —
   without it you get an unobstructed view of the sample, with labels still on top for
   identification:
   ```
   gelpy image show RUN --kind linear                  # clean, linear contrast — labels, no overlay
   gelpy image show RUN --kind non_linear               # clean, gamma contrast — labels, no overlay
   ```
   **Never pass `--lanes` on these.** The labeled contrast-view deliverable (whatever `--kind`)
   must NEVER carry the red lane-center/profile-width overlay in the same image — those are two
   different images serving two different purposes (one is the sample readout, the other is a
   QC check on where gelpy sliced the lanes) and must not be merged. If a `--lanes` QC render is
   wanted at all, produce it as its own separate call/file:
   ```
   gelpy image show RUN --kind non_linear --lanes       # separate QC-only render; never hand this over as *the* gel image
   ```
   Produce the clean (no-`--lanes`) renders as the "here's the sample" deliverable. The
   `--lanes` variant — like `render/lane_detect.png` from `lanes detect` — is for your own
   verification only, or for when someone explicitly asks to see how gelpy sliced the lanes.

6. **Remove background, then a broad fit pass; inspect every lane.**

   **Background removal is mandatory before any profile plot or fit, always, no exceptions** —
   skipping it and normalizing raw profiles by area is a trap: a big-signal lane's background
   floor gets crushed toward 0 while a faint lane's floor gets inflated (division by a much
   smaller total integral), so lanes end up with visibly different, physically meaningless
   baselines even though the underlying background is the same. Subtracting the background
   first is what makes lanes comparable at all.

   ```
   gelpy background remove RUN --model robust_plane
   gelpy background show RUN               # trust the correction: surface + weights + before/after slices
   ```
   `robust_plane` (the default — you can omit `--model`) is an iterative-reweighted-least-squares
   fit of a low-order polynomial surface: it needs no manually placed stripes, self-scales its
   noise estimate each iteration, and stays unbiased even when bands/smears cover a large
   fraction of the image. **Always use it.** `plane_ransac` (two hard-coded background stripes)
   is a fallback ONLY — reach for it if `robust_plane` visibly fails to converge on a clean
   surface (check `background show`'s corrected-slice panels) or the image is unusual enough
   that IRLS keeps latching onto signal; don't reach for it by default or "for simplicity".
   ```
   gelpy fit run RUN --slice <y0> <y1> --plot
   ```
   The fitter needs **no peak-detection tuning**. It estimates a conservative baseline, then
   grows-and-selects components (peak count chosen by AICc, not a prominence knob), picks each
   component's shape (gaussian / EMG-tailed / bi-gaussian / pseudo-voigt), classifies them
   (`sample` / `well` / `smear` / `dust`), and reports each area as a fraction of the **true**
   integrated signal plus a `residual_fraction` and a per-lane `confidence` (`ok`/`low`) with
   `reasons`. `--slice` bounds the window — check **"A well-cavity dip is not signal"** below
   before assuming the well needs no slicing; on some gels it does. Optional: `--method`,
   `--shapes`, `--baseline`, `--poly-order`, `--auto-escalate`. Then open each
   `render/fits/lane_NNN.png` and judge it (see below).

7. **Escalate only the low-confidence lanes** — the cache makes this cheap (only that lane
   recomputes). Read each lane's `confidence` + `reasons`; for a `low` lane, climb the strategy
   ladder (`sparse_seed` for heavy overlap, `bayesian` for uncertainty/robustness) or adjust
   the window:
   ```
   gelpy fit refit-lane RUN --lane <index-or-label> --method bayesian --plot
   ```
   Or let `fit run --auto-escalate` climb the ladder automatically on any low-confidence lane.

8. **Close out.**
   ```
   gelpy status RUN            # expect n_fit == N, n_stale == 0
   gelpy results show RUN --format csv
   ```
   Deliverables live in the run folder: `results.csv` / `results.json`, `report.md`, `render/`.

## Grouped line-profile plots without fitting

Sometimes the ask is just "show me the profiles" (e.g. a purification gel: one named sample lane
followed by several generic fraction lanes) — no peak fitting wanted yet. Use
`gelpy profiles extract RUN --lane <idx> --lane <idx> ... --plot` per group; skip `fit run`
entirely.

**Still run `gelpy background remove RUN --model robust_plane` first, before any `profiles
extract`.** This is easy to skip when there's no fitting on the agenda, but `profiles extract`
reads from the background-corrected image whenever `recipe.background` is set (passthrough to
the raw image otherwise) — skip this step and every plotted profile carries its own raw
background floor into the "normalized to area" plot, which then distorts baselines differently
per lane (see the mandatory-background-removal note in step 6 above for why). Do this once per
run before extracting any profiles; the profile cache picks up the corrected image automatically
on the next `profiles extract`.

- **Select by index, not by label.** Fraction labels like `#2`/`#3` repeat identically across every
  sample group, so `--lane "#3"` matches every group's `#3` lane, not just the one you want. Work
  out each group's lane indices (e.g. groups of 5 → `0-4`, `5-9`, `10-14`, ...) and pass `--lane`
  once per index.
- **The output PNG is named after the first selected lane's label** (sanitized), not a fixed
  filename — so one `profiles extract --plot` call per group produces one distinct PNG per group
  automatically (`render/profiles_<first-lane-label>.png`), and re-running the *same* group with a
  different lane subset (e.g. dropping specific fractions) overwrites that file in place instead of
  piling up stale plots. This makes "replot this group with just lanes X/Y/Z" a matter of re-issuing
  the command with a different set of `--lane` indices — no separate cleanup step.
  Consequence: two requested groups that start with the same lane (e.g. "lanes 0-2" and
  "lanes 0-5") write the same file. Copy each plot to `figures/` under a descriptive name right
  after its call; the path is in the JSON output at `files.profiles_overview`.
- Indices are 0-based; users usually number wells from 1 — convert and say so.

## Constrained component fit (user specifies the allowed peaks)

When the user prescribes the peak set ("one well peak, one smear, one leading band, ..."), the
free-form `fit run` does not honour it. Write a per-gel script on the gelpy profile cache
(`<run>/cache/profiles/lane_NNN__*.json`, full-length background-corrected profile) instead.
Worked example: `examples/constrained_component_fit.py` (well EMG, generalised-Gaussian smear,
optional middle Gaussian, slow-side-tailed EMG band, bi-Gaussian fastest peak, bounded offset;
windows and bounds are parameters; set them from your own profile plots).
Lessons from it:
- Bound every component to its physical region and width, or components trade places (a free
  middle component can widen and take over the smear).
- Use 2 starts for the smear shape (long-tailed vs flat-topped) and keep the lower cost; one
  start can land in a different local minimum on some lanes.
- Drop switched-off components from the parameter vector; freezing them with near-zero-width
  bounds made `least_squares` ~1000x slower. Use `x_scale="jac"`.
- Mask dust only for narrow runs (<= ~7 px) and never over a band top; mask wider specks by hand
  after confirming them in the image, and record the masked ranges.
- Look at every lane's fit plot; R2 > 0.99 can still hide a swapped component.

## Summary PDF report ("summary PDF", "report", "make it a PDF")

A dense, factual record of what was done and what the data shows. Build it with reportlab
(`pip install reportlab` if missing) from a script saved in `agent_analysis/scripts/`; adapt
`examples/report_pdf.py` (images, profiles, and optionally component fits). One PDF per gel row /
experiment, saved as `agent_analysis/report.pdf`.

Content, in order:
1. Title (experiment, gel date, row); one paragraph: image file, size/dtype, saturation check
   (count pixels at the dtype ceiling and where they are), lane/label map or design summary,
   where the run folder / scripts / numbers live.
2. **Processing** bullets: lane detection (pitch, confidence, profile width), background model,
   profile definition and normalisation, fit window and model with all bounds, masking, how
   fractions / positions / migration are computed. Enough to reproduce, nothing else.
3. Every figure that was produced: gel images (linear, non-linear), lane-detect QC, background
   QC, profile plots, fit overviews, summary plots, then all per-lane fit plots at the end.
   Each with a one-line caption: what is plotted + the processing behind it.
4. Results table with every number behind the summary plots (one row per lane).
5. **Observations**: numbers only — positions, areas, fractions, ranges, outliers relative to
   neighbours, artefacts. No interpretation (no "aggregation", "loading error",
   "incomplete folding"); keep interpretations for the chat reply if at all.

Style: no narrative, no motivation, no conclusions section, no emojis. Captions and bullets
are short declarative fragments. Every number in the observations must come from the results
table or a computed metric — recompute, don't quote from memory. Render the pages to PNG
(`pdftoppm -r 45`) and check the layout before handing it over.

## Judging a per-lane fit (`render/fits/lane_NNN.png`)

Left panel: the raw profile, the estimated baseline, and the summed model (they should overlap).
Right panel: the baseline-subtracted signal decomposed into components colored by class, each
labeled with its area fraction; coverage should read ~100%.

**Trust the `confidence` flag and its `reasons`** instead of hand-tuning detection knobs — the
engine already selects the peak count and shapes. Map a `low` lane to an action:
- *"coverage <X>%: signal unmodeled"* → real signal missed; widen `--slice` or escalate.
- *"fitted peaks exceed the signal (overlap/over-fit)"* → `--method bayesian` (or eyeball).
- *"poor fit (R2=...)"* → the shapes don't capture the profile; escalate.
- per-band *"area unreliable"* / *"overlaps a neighbor"* note → that band's split is uncertain;
  `--method bayesian` gives credible intervals, or accept it as lower-confidence.

A `low` lane isn't necessarily wrong — it's flagged for a look. `status: "ok"` only means the
fit converged; the `confidence` flag (not `status`) is the real quality signal. Closely-spaced
real bands (a shoulder) ARE split into separate components; a smear is kept as one broad
component (or region-integrated) so its area fraction is the "how much sample is in the smear"
number.

**A lane can report `confidence: "ok"` while still silently hiding a real feature in the
baseline** — see the next section. A clean coverage number is not proof the baseline didn't
quietly absorb something; always eyeball the left panel too (does the dashed baseline sit
*below* every visible bump, or does it sit right on top of one?).

## A well-cavity dip is not signal — and what real well-trapped aggregate looks like

The well is a physical cutout in the gel with *less gel material in it than the surrounding
lane*, so it always reads as a **dip below baseline**, never a peak — less material means less
signal than the smoothly-extrapolated background would predict. This is a fixed optical/geometric
artifact of the well geometry, not biology: on real gels, the dip's minimum and its zero-crossing
back above baseline sit within a few px of each other in every lane, regardless of sample — same
wells, same imaging setup, same artifact, every time.

If material aggregated and got stuck at the well (too large to migrate into the gel at all), it
shows up as a genuine, distinct **peak immediately after that dip** — right where the profile
crosses back above baseline. That peak's area is the real "trapped in the well" quantity.
**Everything to the left of that peak — the dip itself, and anything before it — is not part of
the data-containing gel.** Don't fit it, don't baseline through it as if it were signal, don't
count it toward anything.

- **Recognize it**: open a lane's raw profile (`render/fits/lane_NNN.png` left panel, or
  `profiles extract --plot`) and look at the very start (near the origin). A dip that bottoms out
  and crosses back above baseline within the first ~10-15% of the window, *before* any real band,
  is this artifact — not a soft, gradually-declining slope (that's just baseline drift, fine to
  leave in). It's the same physical well on every lane of the same gel, so the dip sits at
  essentially the same x-position across all of them — find it on one or two lanes, then reuse
  that boundary for the whole gel.
- **Exclude it**: `--slice <past-the-dip's-zero-crossing> <end>`, applied uniformly across the
  gel via `fit run --lane all`. Don't try to fix this by tuning `well_frac` / `--protect-well`
  instead — masking *more* of the window from baseline-fitting only gives the baseline *more*
  freedom to bend up elsewhere, which makes things worse, not better (verified empirically:
  raising `well_frac` from 0.08 to 0.25 dropped a shoulder's residual from a healthy +0.0009 mean
  to essentially zero — i.e. it made the baseline swallow *more* signal). A slice that starts
  right past the dip is the actual fix.
- **Watch for the side effect**: an order-2 polynomial baseline, once the dip is gone, can have
  just enough freedom left on a short window to bend up and quietly absorb a real smear/shoulder
  sitting right after the cut — silently, with no warning and often with `confidence: "ok"`,
  because the "signal" it ate is now missing from both numerator and denominator. Tell: the dashed
  baseline in the left panel sits right on top of a visible bump instead of below it. Fix:
  `--poly-order 0` (a flat/constant baseline) on that slice — it has no freedom to bend up, so it
  can't eat the bump, but it also can't track genuine gradual drift, so only reach for it on a
  window that starts right past a hard exclusion boundary like this one (a general-purpose default
  of `0` is wrong; it can instead blow a single component's width out to the whole window when the
  window's actual shape needs *some* slope — check both `--poly-order 2` (default) and `0` on a
  couple of lanes and compare before committing to one for the whole gel).
- **This isn't universal.** Some gels/crops don't include the well in the profiled region at all,
  or don't show this dip. Check for the pattern before assuming a gel needs this treatment —
  don't reflexively slice past a fixed offset "just in case."

## Conventions

- **Labels**: use exactly what the sample sheet says, verbatim — never rename, expand, merge, or
  add suffixes (e.g. don't turn a sheet's `#2` into `<sample>_frac2`). If a label looks ambiguous
  or you think it's a mistake, ask the user; do not silently "fix" or reinterpret it. This applies
  everywhere labels get set (`lanes set --labels-file`, manual `--labels`) — the label column is
  the user's data, not something gelpy or the agent gets to redesign.
- **Reproducibility**: prefer detected/derived positions over hand-typed pixels. If you must set
  endpoints manually, record where you read them from (the overlay) so the run is reproducible.
- **Multiple rows/gels**: re-run detection per image — do NOT reuse another gel's pixel endpoints
  even for the "same" experiment; crops differ. (Same-shape crops may coincide, but verify anyway.)

## Machine-readable surface (for agents / scripting)

Every command emits JSON; `gelpy schema <Model>` documents the contracts (e.g. `DetectLanesRequest`,
`LaneDetectionResult`, `FitRunRequest`, `RunStatus`). Errors carry a stable `code`
(`lane_selection`, `registry`, `source_image_changed`, `validation_error`, ...) so you can branch on
failure kind without string-matching messages.
