---
title: Architecture
audience: both
summary: The five layers, the one-directional import rule, and what each boundary buys — including the request-to-result command seam the tool layer wraps (and the MCP server through it), and where it leaks.
subject:
  - src/gelpy/schemas/requests.py
  - src/gelpy/render/report.py
  - src/gelpy/render/html_report.py
  - src/gelpy/runfolder/commands.py
  - src/gelpy/runfolder/lock.py
subject_digest: 5f6f86988cd3e68c
---

`src/gelpy/` holds five layers and one rule. The rule is that imports only ever point
one way, from contracts towards the user interface. Everything a first-time reader needs to
predict where a piece of code lives follows from it.

## The layers

Stacked; each layer imports only from the layers above it, never sideways and never down.

```text
   schemas/     pydantic contracts. stdlib + pydantic only.
                16 files, ~1640 lines.
   ------------------------------------------------------------------
   core/        pure algorithms on ndarrays. numpy / scipy / skimage / sklearn.
                image_io, contrast, lanes, line_profiles, background/, peaks/.
                28 files, ~3760 lines.
   ------------------------------------------------------------------
   render/                       runfolder/
   data -> PNG/markdown/HTML     run-folder format, the hash-chain cache,
   matplotlib lives here         the run lock, and f(request) -> result commands
   9 files, ~950 lines           13 files, ~2320 lines
   ------------------------------------------------------------------
   cli/         typer adapter: argv -> request, result -> JSON; the tool
                layer runs each step with its rendering; mcp_server.py is
                the stdio MCP server over that layer. 7 files, ~2000 lines.
```

Measured, not asserted: an AST sweep over every `.py` file in `src/gelpy/` resolving each
relative import to its layer gives `core -> {schemas}`, `render -> {core, schemas}`,
`runfolder -> {core, schemas}`, `cli -> {core, render, runfolder, schemas}`. No edge points
upward, and `render/` and `runfolder/` do not know about each other. A sixth directory,
`utils/`, holds two small modules and sits outside the story: `cm_to_inch` (figure sizing, used by
`render/`) and `atomic_write`, the one atomic file writer. Both layers write through it, which is
how the figures and the cache share one implementation without either layer importing the other.

The data flows down the same graph, one cache directory per stage:

```text
recipe.json + source image
  |  core.image_io.canonicalize_image      -> cache/image/      (image_hash)
  v  float64 [0,1], signal-bright, 2-D
  |  core.background                       -> cache/background/ (background_hash)
  v  corrected array
  |  core.line_profiles                    -> cache/profiles/   (profile_hash)
  v  one 1-D profile per lane
  |  core.peaks                            -> cache/fits/       (fit_hash)
  v  one LaneFitResult per lane
  +-> runfolder.results  -> results.csv, results.json
  +-> render.report      -> report.md                  } triggered from cli/,
  +-> render.fits        -> render/fits/lane_NNN.png   } never from runfolder/
  +-> render.html_report -> report/index.html          }
```

The four hashes and what invalidates each are the subject of
[concepts.md](concepts.md) — read that for the caching story. This page is about the
boundaries.

## What each boundary buys

**`schemas/` imports standalone.** No numpy, no scipy, no matplotlib — the only third-party
import is pydantic, and the one pandas use (`AnalysisResult.to_dataframe`,
`schemas/results.py:49`) is a deliberate lazy import inside the method. So a thin client can
validate a request and discover the whole contract surface without installing the scientific
stack. That is not a hypothetical: `gelpy schema <Model>` is schema discovery over exactly
these models, and `gelpy tool list --json` builds each tool's input schema from them.

**`core/` is algorithms, not plumbing.** It takes arrays and small config objects and returns
arrays and results. It never reads a run folder, never writes a cache file, never decides what
to recompute. One exception, and it is the input funnel: `core.image_io.decode_image` reads the
file header (`probe_image`), then calls `skimage.io.imread` (`image_io.py:414`) to turn a path
into pixels. Those two reads are the only filesystem access in the layer.

**`runfolder/commands.py` is request-in / result-out.** Eleven command functions
(`init_run`, `set_image`, `set_lanes`, `detect_lanes`, `set_contrast`, `remove_background`,
`extract_profiles`, `fit_run`, `refit_lane`, `get_results`, `status`), each taking one
pydantic request from [`schemas/requests.py`](../../src/gelpy/schemas/requests.py) and
returning one pydantic result. `detect_lanes` has one sibling, `run_lane_detection`: the same
command, returning a `LaneDetectionOutcome` (that result plus the image, the detection's arrays
and the row bounds it used) to a front-end that also draws the overlay, so the overlay comes
from the detection that ran and not from a second one; `detect_lanes` is its `.result`. No
argv, no `print`, no `sys.exit` — it writes plenty of files (`recipe.json`, `manifest.json`,
`fit_config.json`, `results.csv`, the caches, `log/history.jsonl`), it just never touches the
terminal. Every function that writes holds an advisory lock on the run folder
([`runfolder/lock.py`](../../src/gelpy/runfolder/lock.py); the renderers in `cli/_render.py` take
it too), so a second writer fails at once with `run_locked`, and a run whose own folders are links
is refused (`invalid_value`) before anything is written,
while `get_results` and `status` never wait for it. Domain failures are raised as
`GelpyError` subclasses, which `core.errors.to_error_response` turns into an `ErrorResponse`
with a stable `code`; the CLI calls it and any other front-end would call it too.

This is the seam, modulo the one leak listed at the end of this page. A CLI step is more than
its command, though: `lanes detect` also renders the overlay and `fit run` also writes
`report.md`. So [`cli/tools.py`](../../src/gelpy/cli/tools.py) pairs each request model with a
handler that runs the command and the rendering its CLI step does, under one run-folder lock,
and the commands, `gelpy tool call` and the MCP server all run it. A front-end that wrapped
`commands.*` itself would drop the rendering. It is why the request models carry `description=`
on every field: those descriptions are the tool documentation an agent reads, so they are part
of the contract, not decoration.

**`render/` is where matplotlib is imported, and nowhere else except the interactive picker.**
<!-- claim: matplotlib-only-in-render -->
`render/__init__.py` calls `matplotlib.use("Agg")` at import time, so the whole package is
headless-safe by construction. Every renderer is a pure function of data: `render_lane_fit`
takes `(prepared_profile, baseline, LaneFitResult, out_path)`, `render_report` takes
`(Recipe, AnalysisResult, out_path)`. None of them opens a run folder, and nothing calls
`plt.show()`. Note that `render/` means *output artifact*, not *matplotlib*:
[`render/report.py`](../../src/gelpy/render/report.py) builds `report.md` from a
`string.Template` and imports nothing but stdlib and `schemas/`, and so does
[`render/html_report.py`](../../src/gelpy/render/html_report.py), which builds the static HTML
report ([html-report.md](html-report.md)) from a `ReportContent` and the bytes of the images that
`cli/_render.py` collects. Which folder a report may be written into is `runfolder/`'s business
(`report_folder.py`: the marker that says a folder is the report's), not the renderer's.

**`cli/` is an adapter.** Twenty commands; thirteen of them build one request, run the tool of
the same step and emit its result as JSON (`results show` also offers `--format csv|table`,
which format the same result themselves). The tool is where `lanes detect` calls
`run_lane_detection` and draws the overlay from its outcome, and where `fit run` writes the
report, so the typer command only parses argv and builds the request. `background show` is pure
rendering and `lanes pick` waits for clicks: neither is a tool, and `image_render` calls no
command function at all.
`schema`, `models list` and `tool list` touch no run folder, and `tool call` runs any tool from
one JSON object. `mcp` serves every tool over stdio:
[`cli/mcp_server.py`](../../src/gelpy/cli/mcp_server.py) advertises the registry's tools and
answers each call with `call_tool_safely`, one call at a time in a worker thread. It imports the
optional `mcp` package only when it starts, and a missing package is a `missing_dependency`
error. Every run-folder command takes the directory as an explicit positional
argument, so there is no ambient "current run" and a shell history is replayable as written.

Two commands reach past `commands.py` into `runfolder/` directly, because no command function
can hand back what they need: `lanes pick` constructs `RunPaths`/`Pipeline` itself to get the
image *array* to click on (and holds the run-folder lock while it waits for the clicks), and
`results show --format csv` calls `runfolder.results.results_csv_text` to render the live
assembly rather than read back a `results.csv` that a recipe edit may have staled.

The tools that write in several steps (`fit_run`, `fit_refit_lane`, `lanes_detect`,
`report_html`, and `background_remove` and `profiles_extract` with `plot`) take
`runfolder.lock.run_lock` once around all of them, so that a retried call cannot slip in between
a fit and its report. The lock is re-entrant, so the locked steps inside it run unchanged.

## The two exceptions in `cli/`

Rendering is the one thing the layering does not let `runfolder/` do, because doing it would
put matplotlib behind the command seam. So two `cli/` modules step outside the pattern:

- **`cli/_render.py`** — reads run-folder artifacts and calls the pure `render.*` functions.
  It is the only place that knows both the cache layout and the renderers. It does *not*
  import matplotlib itself; it imports the `render` package lazily inside each function, so
  the import (and the Agg backend selection) happens only when something is actually drawn.
  Every tool that produces a PNG, `report.md` or the HTML report goes through it.
- **`cli/_pick.py`** — `gelpy lanes pick`, the interactive human-in-the-loop lane-center
  picker. It is the sole module outside `render/` that imports matplotlib, and it must be:
  it calls `matplotlib.use(backend, force=True)` over `MacOSX`/`QtAgg`/`TkAgg` and blocks on
  `fig.ginput`, which is the opposite of what `render/` guarantees. Routing it through
  `render/` would mean a headless backend and no clicks. It is the last-resort fallback after
  `lanes detect` and overlay inspection — see [lane-placement.md](lane-placement.md).

## Seeing the boundaries hold

Two variables for every block below, set once. This block is skipped by the test that executes
the rest of the page, because that test sets both itself:

```bash
# doc-test: skip
export IMAGE=examples/data/example_gel_1.tif   # the example gel, 20 lanes
export RUN=/tmp/gelpy-architecture              # a scratch run folder
```

The import isolation is checkable in one shot:

```bash
python - <<'PY'
import sys
import gelpy.schemas                  # the contracts
print("sci stack after gelpy.schemas:",
      [m for m in ("numpy", "scipy", "matplotlib") if m in sys.modules])
import gelpy.runfolder.commands       # the command API
print("matplotlib after runfolder.commands:", "matplotlib" in sys.modules)
PY
```

```text
sci stack after gelpy.schemas: []
matplotlib after runfolder.commands: False
```

Because the command layer takes requests and not argv, the CLI is only one caller. Here is
the other one — a lane fitted from Python, with no argv anywhere in the call:

```bash
gelpy init "$RUN" "$IMAGE" --n-lanes 20 > /dev/null
gelpy background remove "$RUN" > /dev/null
python - <<'PY'
import os
from gelpy.runfolder import commands
from gelpy.schemas.requests import FitRunRequest, StatusRequest

run = os.environ["RUN"]
before = commands.status(StatusRequest(run_dir=run))
print("lanes:", before.n_lanes, "| already fit:", before.n_fit, "| polarity:", before.polarity)
summary = commands.fit_run(FitRunRequest(run_dir=run, lanes=[0]))
print("bands in lane 0:", summary.lanes["lane=0"]["bands"])
print("files written:", sorted(summary.files))
PY
test ! -f "$RUN/report.md" && echo "no report.md: commands.fit_run does not render"
```

```text
lanes: 20 | already fit: 0 | polarity: bright_on_dark
bands in lane 0: 3
files written: ['results_csv', 'results_json']
no report.md: commands.fit_run does not render
```

The two front-ends share one cache, because the cache is keyed on content, not on who asked.
The identical fit through the CLI recomputes nothing and additionally writes the report:

```bash
gelpy fit run "$RUN" --lane 0 \
  | python -c "import json,sys; d=json.load(sys.stdin); print('recomputed:', d['recomputed']); print('reused:', d['reused'])"
test -f "$RUN/report.md" && echo "report.md now exists: the CLI wrote it"
```

```text
recomputed: []
reused: ['background', 'fit:lane=0', 'image', 'profile:lane=0']
report.md now exists: the CLI wrote it
```

## Where the seam leaks

One place where the command layer is visible as a defect rather than a design. It matters to
anyone wrapping `commands.py` in a front-end of their own, and it is reproducible with the run
folder built above.

- **`plot` is inert at the command layer.** `FitRunRequest.plot`, `RefitLaneRequest.plot`,
  `DetectLanesRequest.plot`, `RemoveBackgroundRequest.plot` and `ExtractProfilesRequest.plot`
  are declared on the contracts, but `request.plot` is never read anywhere in `runfolder/`.
  `commands.fit_run(FitRunRequest(..., plot=True))` leaves `render/fits/` empty and writes no
  `report.md`. Every front-end honours the flag through the tool layer instead: the handlers in
  `cli/tools.py` render after the command returns, and a result names a rendered file
  (`overlay`, `files.report`, `files.plot`, ...) by its run-relative path, and only when the
  file was rendered.

## Deferred, so you do not go looking

Designed for, referenced in docstrings and in `CLAUDE.md`, and genuinely not implemented. A
grep for each turns up a comment at most:

- **`tune` / `refit-all`.** No such commands; the CLI tree is `init`, `status`, `schema`,
  `lanes`, `contrast`, `background`, `image`, `profiles`, `fit`, `results`, `report`, `models`,
  `tool`, `mcp`.
- **Notebook scaffolding.** `AnalysisResult.to_dataframe()` exists as the notebook bridge;
  nothing consumes it.
- **`replay` from `log/history.jsonl`.** Appended by the nine command functions that record
  intent or write fit results — not by `status`, `results show`, `lanes detect` without
  `--apply`, or the pure-render commands, all of which can still write cache files and PNGs
  into the folder. So the log is not a complete record of what touched the run folder. Read by
  nothing.
- **MW / ladder calibration.** Does not exist.

## Next

- Adding a peak shape, a fit strategy, or a background model, and what the registries want
  from you: [extending.md](extending.md).
- The run folder, the four hashes, and what a given edit invalidates: [concepts.md](concepts.md).
- The exhaustive surface — every command, every contract, every registry entry, every error
  code — is generated from the code under [reference/](../reference/). Do not learn it from
  prose; `gelpy --help`, `gelpy schema <Model>` and `gelpy models list` are the live version.
