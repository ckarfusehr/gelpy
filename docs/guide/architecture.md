---
title: Architecture
audience: both
summary: The five layers, the one-directional import rule, and what each boundary buys — including the request-to-result command seam an MCP server would wrap, and where it leaks.
subject:
  - src/gelpy/schemas/requests.py
  - src/gelpy/render/report.py
  - src/gelpy/runfolder/commands.py
subject_digest: 6a820c88e980cd82
---

`src/gelpy/` holds five layers and one rule. The rule is that imports only ever point
one way, from contracts towards the user interface. Everything a first-time reader needs to
predict where a piece of code lives follows from it.

## The layers

Stacked; each layer imports only from the layers above it, never sideways and never down.

```text
   schemas/     pydantic contracts. stdlib + pydantic only.
                14 files, ~1270 lines.
   ------------------------------------------------------------------
   core/        pure algorithms on ndarrays. numpy / scipy / skimage / sklearn.
                image_io, contrast, lanes, line_profiles, background/, peaks/.
                28 files, ~3200 lines.
   ------------------------------------------------------------------
   render/                       runfolder/
   data -> PNG or markdown       run-folder format, the hash-chain cache,
   matplotlib lives here         and f(request) -> result commands
   7 files, ~550 lines           11 files, ~1280 lines
   ------------------------------------------------------------------
   cli/         typer adapter: argv -> request, result -> JSON,
                and it triggers rendering. 5 files, ~1010 lines.
```

Measured, not asserted: an AST sweep over every `.py` file in `src/gelpy/` resolving each
relative import to its layer gives `core -> {schemas}`, `render -> {core, schemas}`,
`runfolder -> {core, schemas}`, `cli -> {core, render, runfolder, schemas}`. No edge points
upward, and `render/` and `runfolder/` do not know about each other. A sixth directory,
`utils/`, holds one function (`cm_to_inch`, used only by `render/` for figure sizing) and sits
outside the story.

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
```

The four hashes and what invalidates each are the subject of
[concepts.md](concepts.md) — read that for the caching story. This page is about the
boundaries.

## What each boundary buys

**`schemas/` imports standalone.** No numpy, no scipy, no matplotlib — the only third-party
import is pydantic, and the one pandas use (`AnalysisResult.to_dataframe`,
`schemas/results.py:43`) is a deliberate lazy import inside the method. So a thin client can
validate a request and discover the whole contract surface without installing the scientific
stack. That is not a hypothetical: `gelpy schema <Model>` is schema discovery over exactly
these models, and a future MCP server would advertise its tool signatures from them.

**`core/` is algorithms, not plumbing.** It takes arrays and small config objects and returns
arrays and results. It never reads a run folder, never writes a cache file, never decides what
to recompute. One exception, and it is the input funnel: `core.image_io.decode_image` calls
`skimage.io.imread` (`image_io.py:125`) to turn a path into pixels. That is the only
filesystem read in the layer.

**`runfolder/commands.py` is request-in / result-out.** Eleven command functions
(`init_run`, `set_image`, `set_lanes`, `detect_lanes`, `set_contrast`, `remove_background`,
`extract_profiles`, `fit_run`, `refit_lane`, `get_results`, `status`), each taking one
pydantic request from [`schemas/requests.py`](../../src/gelpy/schemas/requests.py) and
returning one pydantic result. No argv, no `print`, no `sys.exit` — it writes plenty of files
(`recipe.json`, `manifest.json`, `fit_config.json`, `results.csv`, the caches,
`log/history.jsonl`), it just never touches the terminal. Most domain failures are raised as
`GelpyError` subclasses, which `core.errors.to_error_response` turns into an `ErrorResponse`
with a stable `code`; the CLI calls it and any other front-end would call it too.

This is the seam. An MCP server is `mcp.tool()(commands.refit_lane)` plus a transport — the
docstring in `requests.py` says so, and the shape holds modulo the four leaks listed at the
end of this page. It is why the request models carry `description=` on every field: those
descriptions are the tool documentation an agent reads, so they are part of the contract,
not decoration.

**`render/` is where matplotlib is imported, and nowhere else except the interactive picker.**
<!-- claim: matplotlib-only-in-render -->
`render/__init__.py` calls `matplotlib.use("Agg")` at import time, so the whole package is
headless-safe by construction. Every renderer is a pure function of data: `render_lane_fit`
takes `(prepared_profile, baseline, LaneFitResult, out_path)`, `render_report` takes
`(Recipe, AnalysisResult, out_path)`. None of them opens a run folder, and nothing calls
`plt.show()`. Note that `render/` means *output artifact*, not *matplotlib*:
[`render/report.py`](../../src/gelpy/render/report.py) builds `report.md` from a
`string.Template` and imports nothing but stdlib and `schemas/`.

**`cli/` is an adapter.** Sixteen commands; twelve of them build one request, call exactly one
`commands.*` function, and emit the result as JSON (`lanes detect` merges an extra `overlay`
path into the dump; `results show` also offers `--format csv|table`, which format the same
result themselves). `background show` and `image show` call no command function at all — they
are pure rendering. `schema` and `models list` touch no run folder. Every run-folder command
takes the directory as an explicit positional argument, so there is no ambient "current run"
and a shell history is replayable as written.

Two commands reach past `commands.py` into `runfolder/` directly, because no command function
can hand back what they need: `lanes pick` constructs `RunPaths`/`Pipeline` itself to get the
image *array* to click on, and `results show --format csv` calls
`runfolder.results.results_csv_text` to render the live assembly rather than read back a
`results.csv` that a recipe edit may have staled.

## The two exceptions in `cli/`

Rendering is the one thing the layering does not let `runfolder/` do, because doing it would
put matplotlib behind the MCP seam. So two `cli/` modules step outside the pattern:

- **`cli/_render.py`** — reads run-folder artifacts and calls the pure `render.*` functions.
  It is the only place that knows both the cache layout and the renderers. It does *not*
  import matplotlib itself; it imports the `render` package lazily inside each function, so
  the import (and the Agg backend selection) happens only when something is actually drawn.
  Every command that produces a PNG or `report.md` goes through it.
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
export IMAGE=examples/data/folding_screen.tif   # the example gel, 20 lanes
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
the other one — a run folder created and fitted from Python, with no argv anywhere:

```bash
gelpy init "$RUN" "$IMAGE" --n-lanes 20 > /dev/null
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
reused: ['fit:lane=0', 'image', 'profile:lane=0']
report.md now exists: the CLI wrote it
```

## Where the seam leaks

Four places where the command layer is visible as a defect rather than a design. All four
matter to anyone wrapping `commands.py` in a second front-end, and all four are reproducible
with the run folder built above.

- **`plot` is inert at the command layer.** `FitRunRequest.plot` and `RefitLaneRequest.plot`
  are declared on the contracts, but `request.plot` is never read anywhere in `runfolder/`.
  Rendering happens in `cli/app.py`, after the command returns.
  `commands.fit_run(FitRunRequest(..., plot=True))` leaves `render/fits/` empty. An MCP server
  wrapping the command functions verbatim would silently produce no plots and no `report.md`.
- **`refit_lane` reports a plot path unconditionally.** `_result_files(paths, lane_index=...)`
  always adds `files["plot"] = "render/fits/lane_NNN.png"`, whether or not a PNG was rendered.
  Called with `plot=False`, the returned summary names a file that does not exist.
- **The `files` map mixes paths relative to the run root with the path as given.**
  `commands.py` relativizes to the run root (`"results_csv": "results.csv"`), then
  `cli/app.py` assigns `result.files["report"]` straight from `_render.write_report`, which
  returns the path as given. With an absolute `RUN_DIR` that is an absolute path; with a
  relative one it is relative to the *cwd*, not to the run root. Consumers must handle both.
- **Five domain failures come out as `internal_error`.** `to_error_response` has branches for
  `GelpyError`, `FileNotFoundError` and pydantic's `ValidationError`, then a generic fallback
  its own docstring reserves for unknown exceptions. But five checks in `commands.py` — the
  label-count mismatches in `set_lanes`/`detect_lanes` and the missing-`n_lanes` cases — still
  raise a bare `ValueError`, so `code: "internal_error"` is what a caller branching on the
  code sees for "you passed 2 labels for 20 lanes". Indistinguishable from a crash.

## Deferred, so you do not go looking

Designed for, referenced in docstrings and in `CLAUDE.md`, and genuinely not implemented. A
grep for each turns up a comment at most:

- **MCP server.** No module, no dependency. The seam described above is the whole preparation.
- **`tune` / `refit-all`.** No such commands; the CLI tree is `init`, `status`, `schema`,
  `lanes`, `contrast`, `background`, `image`, `profiles`, `fit`, `results`, `models`.
- **HTML report and notebook scaffolding.** `report.md` is the only report format.
  `AnalysisResult.to_dataframe()` exists as the notebook bridge; nothing consumes it.
- **`replay` from `log/history.jsonl`.** Appended by the nine command functions that record
  intent or write fit results — not by `status`, `results show`, `lanes detect` without
  `--apply`, or the pure-render commands, all of which can still write cache files and PNGs
  into the folder. So the log is not a complete record of what touched the run folder. Read by
  nothing.
- **MW / ladder calibration, run-folder locking.** Neither exists — no `flock`/`fcntl`/lock of
  any kind, so two commands writing one run folder concurrently are unguarded.

## Next

- Adding a peak shape, a fit strategy, or a background model, and what the registries want
  from you: [extending.md](extending.md).
- The run folder, the four hashes, and what a given edit invalidates: [concepts.md](concepts.md).
- The exhaustive surface — every command, every contract, every registry entry, every error
  code — is generated from the code under [reference/](../reference/). Do not learn it from
  prose; `gelpy --help`, `gelpy schema <Model>` and `gelpy models list` are the live version.
