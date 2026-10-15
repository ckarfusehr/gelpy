---
name: gelpy-analyze
description: >
  Analyze a gel-electrophoresis image end-to-end with gelpy, through its CLI or its tools
  (gelpy tool call, gelpy mcp): place lanes correctly from the actual image (never
  guessed/hardcoded positions), remove background, fit per-lane band intensities, and produce a
  browsable run folder (results.csv/json, report.md, per-lane fit PNGs). Encodes the
  verify-as-you-go judgment loop — detect lanes → confirm the overlay → fit → inspect each lane →
  refit stragglers. Use when asked to analyze / quantify / measure bands or purity on an agarose
  or protein gel image (.tif/.png), or to drive gelpy on a new gel.
  Invoked with only a folder (or no explicit ask), it does the default scope ONLY: label the
  lanes and render linear, non-linear and lane-detection images — no profiles, no fitting.
  Also builds the dense summary PDF report when asked for a "summary PDF" / "report".
---

# gelpy — analyze a gel image

gelpy turns a gel image into per-lane band intensities through a content-hash-cached pipeline:
`image → background-corrected image → per-lane line profiles → peak fits → run folder`. Every
step is a CLI command (`gelpy <group> <verb> RUN ...`) and the same step as a tool
(`gelpy tool call NAME`, or over MCP with `gelpy mcp`); both take and return JSON and recompute
only what a change invalidated. Every failure is one JSON error whose `detail` names the next
step. `gelpy --help`, `gelpy tool list --json` and `gelpy schema <Model>` describe everything.

**Read [judgment.md](judgment.md) before you place lanes or judge a fit.** It holds every check
and decision rule of this skill, independent of how gelpy is driven; this file says which
command or tool does each step.

## Steps, commands and tools

| step | CLI | tool |
|---|---|---|
| create the run | `gelpy init RUN IMAGE --n-lanes N --labels-file F` | `init` |
| read its state | `gelpy status RUN` | `status` |
| set polarity, channel or page | `gelpy image set RUN --invert` | `image_set` |
| detect lanes, render the overlay | `gelpy lanes detect RUN --n-lanes N --apply` | `lanes_detect` |
| set lanes, labels or width | `gelpy lanes set RUN --endpoints LEFT RIGHT --n-lanes N` | `lanes_set` |
| set the display contrast | `gelpy contrast set RUN --gamma 0.2` | `contrast_set` |
| render a gel view | `gelpy image show RUN --kind linear` | `image_render` |
| configure background removal | `gelpy background remove RUN --plot` | `background_remove` |
| line profiles | `gelpy profiles extract RUN --lane 0 --lane 1 --plot` | `profiles_extract` |
| fit lanes | `gelpy fit run RUN --plot` | `fit_run` |
| refit one lane | `gelpy fit refit-lane RUN --lane 3 --method bayesian --plot` | `fit_refit_lane` |
| read the results | `gelpy results show RUN` | `results_get` |
| HTML report | `gelpy report html RUN` | `report_html` |

Tools take the commands' fields by their contract names (`gelpy schema InitRunRequest` and so
on): `RUN` is `run_dir`, a labels file becomes `labels`, `--endpoints LEFT RIGHT` becomes
`endpoints`. In `fit_run` the fit fields nest as in the contracts: `slice_start` and `slice_end`
in `profile`; `method`, `shapes` and `baseline` (with `poly_order`) in `fit`. `fit_refit_lane`
takes the same as `profile_overrides` and `fit_overrides`.
`gelpy lanes pick` (click lane centers on screen) has no tool: it needs a display.
`gelpy background show` has none either: `background_remove` with `plot` true renders the same
diagnostic.

## Default scope — what to do when only a folder is given

If the skill is invoked with just a folder (or with no request beyond "analyze this gel"), do
**exactly this and stop**:

1. Find the gel image(s) in the folder (one run per image / gel row; skip byte-identical copies —
   check with `md5`).
2. Get lane count `N` and labels from the user's message or a sample sheet in the folder
   (xlsx/csv/txt). If neither exists, ask — `N` and labels are never inferred.
3. `gelpy init` → `gelpy lanes detect --apply` → **verify the overlay** (judgment.md, "The one
   rule") → `gelpy lanes set --labels-file` (tools `init`, `lanes_detect`, `lanes_set`).
4. Render `gelpy image show --kind linear` and `--kind non_linear` (clean, no `--lanes`; tool
   `image_render`), and the lane-detection QC (`render/lane_detect.png`). If you set the lanes by hand after
   detection, that file still shows the detection: draw the lanes as set with
   `gelpy image show --kind non_linear --lanes` (tool `image_render` with `kind` and `lanes`) and
   use that image as the QC.
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

## Workflow

Steps 1-5 are the default scope. Steps 6-8 run only when the user asks for profiles, fits or
quantification.

1. **Create the run and look at the raw image** (tools `init`, `image_render`).
   ```
   gelpy init RUN IMAGE --n-lanes N
   gelpy image show RUN --kind raw --lanes     # the raw image with the current (placeholder) lane strips
   ```
   Check the polarity and page warnings (judgment.md, "Check how the image was read").
2. **Detect lanes** (tool `lanes_detect`): `gelpy lanes detect RUN --n-lanes N --apply` writes
   `render/lane_detect.png`.
3. **Verify the overlay** (judgment.md, "The one rule"). Its recoveries, as commands (tools
   `lanes_set` and `lanes_detect`; where a flag and its field differ, the comment names the field):
   ```
   gelpy lanes set RUN --endpoints LEFT RIGHT --n-lanes N
   gelpy lanes detect RUN --n-lanes N --y-end Y --prominence 0.03 --apply   # y_end, min_prominence
   gelpy lanes detect RUN --n-lanes N --min-width W --apply                 # min_width
   gelpy lanes detect RUN --n-lanes N --corrected --apply     # use_corrected; after gelpy background remove RUN
   gelpy lanes set RUN --x-center C1 --x-center C2 ...        # x_centers, one per lane
   gelpy lanes pick RUN --n-lanes N            # two outer clicks; --all clicks every lane
   ```
   To look again without changing the lanes: `gelpy image show RUN --kind non_linear --lanes`
   draws the lanes as the run holds them, which is how to check lanes set by hand or by clicking;
   `gelpy lanes detect RUN --n-lanes N` without `--apply` draws a fresh detection instead, and
   ignores the lanes as set.
4. **Label the lanes** (tool `lanes_set`) from the sheet:
   `gelpy lanes set RUN --labels-file labels.txt`. Labels are vertical by default
   (`InitRunRequest.label_rotation` and `LaneLayout.label_rotation` are 90).
5. **Render the clean views** (tool `image_render`): `gelpy image show RUN --kind linear` and
   `gelpy image show RUN --kind non_linear`, never with `--lanes` (judgment.md, "Clean views
   and QC views").
6. **Background, then a broad fit** (tools `background_remove`, `fit_run`):
   `gelpy background remove RUN` (default `robust_plane`), check it with
   `gelpy background show RUN`, then `gelpy fit run RUN --slice Y0 Y1 --plot`. Options:
   `--method`, `--shapes`, `--baseline`, `--poly-order`, `--auto-escalate`. Start the slice past
   the well dip when the gel has one (judgment.md, "A well-cavity dip is not signal"). Open each
   `render/fits/lane_NNN.png` and judge it (judgment.md, "Judging a per-lane fit").
7. **Escalate only the low-confidence lanes** (tool `fit_refit_lane`):
   `gelpy fit refit-lane RUN --lane I --method bayesian --plot`, or
   `gelpy fit run RUN --auto-escalate`.
8. **Close out** (tools `status`, `results_get`, `report_html`): `gelpy status RUN` (expect
   `n_fit == N`, `n_stale == 0` and no `stale_stages`), `gelpy results show RUN --format csv`, and
   `gelpy report html RUN` for a page to read (see "HTML report" below). Deliverables:
   `results.csv` / `results.json`, `report.md`, `report/index.html`, `render/`.

## Grouped line-profile plots without fitting

Sometimes the ask is just "show me the profiles" (e.g. a purification gel: one named sample lane
followed by several generic fraction lanes) — no peak fitting wanted yet. Use
`gelpy profiles extract RUN --lane <idx> --lane <idx> ... --plot` (tool `profiles_extract`) per
group; skip `fit run` entirely.

Background removal comes first here too (judgment.md): gelpy refuses `profiles extract` on a run
without it unless `--allow-raw`. Run `gelpy background remove RUN` once per run before extracting
profiles.

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
free-form `fit run` (tool `fit_run`) does not honor it. Write a per-gel script on the gelpy profile cache
(`<run>/cache/profiles/lane_NNN__*.json`, full-length background-corrected profile) instead.
Worked example: `examples/constrained_component_fit.py` (well EMG, generalized-Gaussian smear,
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
   profile definition and normalization, fit window and model with all bounds, masking, how
   fractions / positions / migration are computed. Enough to reproduce, nothing else.
3. Every figure that was produced: gel images (linear, non-linear), the lane QC (the lanes as set,
   which `examples/report_pdf.py` draws itself), background
   QC, profile plots, fit overviews, summary plots, then all per-lane fit plots at the end.
   Each with a one-line caption: what is plotted + the processing behind it.
4. Results table with every number behind the summary plots (one row per lane).
5. **Observations**: numbers only — positions, areas, fractions, ranges, outliers relative to
   neighbors, artifacts. No interpretation (no "aggregation", "loading error",
   "incomplete folding"); keep interpretations for the chat reply if at all.

Style: no narrative, no motivation, no conclusions section, no emojis. Captions and bullets
are short declarative fragments. Every number in the observations must come from the results
table or a computed metric — recompute, don't quote from memory. Render the pages to PNG
(`pdftoppm -r 45`) and check the layout before handing it over.

## HTML report (`gelpy report html`, tool `report_html`)

A static page of the run: lane overlay, clean views, labels, results, profiles, fits and
provenance. Open `report/index.html`. What an agent must know is below; the guide page
`html-report.md` has the rest.

- **The report folder is gelpy's.** A build writes `.gelpy-report.json` into it, and a rebuild
  replaces only the files that marker lists. `--out` (`out_dir` in the tool) must name a new or
  empty folder, or one an earlier report build wrote; any other non-empty folder is refused with
  `invalid_value`.
- **Refused whatever the folder holds:** the run folder, a folder inside it other than its own
  `report/`, a folder that holds the run folder, a folder that holds a `recipe.json` or a
  `manifest.json`, and a default `report/` that is a link (delete the link and repeat).
- **The page is a snapshot.** `status` reports `report` as `{path, current}`; `current: false`
  means a relabel, a refit or another gelpy version changed the run since the build: build again.

## Machine-readable surface (for agents / scripting)

Every command emits JSON; `gelpy schema <Model>` documents the contracts (e.g.
`DetectLanesRequest`, `LaneDetectionResult`, `FitRunRequest`, `RunStatus`), and
`gelpy tool list --json` the tools. Errors carry a stable `code` (`lane_selection`,
`registry`, `background_required`, `run_locked`, `validation_error`, ...) and a `detail` that
names the next step, so you can branch on the failure kind without matching messages.
