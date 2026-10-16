# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What it does

gelpy is a gel electrophoresis image analysis package, rebuilt as an agent-native,
CLI-first tool. Data flows: source image → (optional) background-corrected image →
per-lane line profiles → peak fits → a browsable **run folder** (config, cached
intermediates, CSVs, plots, report). The whole pipeline is content-hash cached so a
targeted edit (e.g. refitting one lane) recomputes only what changed.

## Commands

- Local dev install: `pip install -e ".[dev,mcp]"` (hatchling backend; `pyproject.toml` is the
  source of truth). `dev` is the test and documentation suites; `mcp` is the optional package
  `gelpy mcp` serves with (the tests of the server skip without it, CI installs it).
  `pip install -e ".[notebook]"` adds the notebook libraries.
- Run the CLI: `gelpy ...` or `python -m gelpy ...`.
- Run all tests: `pytest tests/`
- Run one layer: `pytest tests/core`, `pytest tests/runfolder`, `pytest tests/cli`, etc.
- The `tests/e2e` replay uses `examples/data/example_gel_1.tif` (skipped if absent).
- Dev environment: the `gelpy_dev` conda env has the sci stack + pydantic/typer/pytest.
- No configured linter. CI (.github/workflows/ci.yml) runs the test and documentation suites.

## Architecture

Strictly one-directional dependencies: `schemas/` → `core/` → (`render/` + `runfolder/`) → `cli/`.
`utils/` (`atomic_write`, `cm_to_inch`) sits outside the layers: `render/` and `runfolder/` both
write through `atomic_write`.

- **`schemas/`** — pydantic v2 contracts. No numpy/scipy/matplotlib; importable
  standalone. Every model subclasses `CanonicalModel` (`extra="forbid"`), so an
  unknown field/name is a hard `ValidationError`. Every request derives from `RequestModel`,
  so a number in a request must be finite. Key models: `Recipe` (persisted
  intent; deliberately excludes fit config), `FitRunRequest`/`RefitLaneRequest`,
  `FitConfig` (strategy `method` + `baseline`/`shapes`/`priors`/`seed`; the seed keeps the
  bayesian tier cacheable), the discriminated `PeakParams` shapes, `BandFit`
  (`area`/`area_fraction`/`area_uncertainty`/`component_class`), `FitQuality`,
  `LaneFitResult` (`coverage`/`residual_fraction`/`confidence`/`confidence_reasons`;
  `to_rows()` emits true-area columns), `BaselineConfig`,
  `FitConfigStore`/`PerLaneFitConfig` (durable per-lane `(profile, fit)` intent → `fit_config.json`),
  `RunStatus`/`LaneStatus` (the `status` command's result; its `report` is a `ReportStatus`),
  `ErrorResponse` (carries a machine-readable `code`), `ToolSpec` (a tool as
  `gelpy tool list --json` describes it), `ReportMarker` (`report/.gelpy-report.json`).

- **`core/`** — pure algorithms on ndarrays. `image_io.canonicalize_image` is the single
  input funnel: any dtype/RGB(A)/multi-page source -> a canonical **float64 [0,1],
  signal-bright, finite** 2-D array + a recorded `ImageMeta` (channel + polarity + page). The
  file header is probed first, and a file over the pixel or page cap (`GELPY_MAX_PIXELS`,
  `GELPY_MAX_PAGES`) is refused before any pixel is decoded. What tifffile logs while it reads a
  file (it logs a damaged TIFF and carries on) is kept in `ImageMeta.reader_warnings`, which
  `init`, `image set` and `status` show; no decoder text reaches an error with the file's folders.
  `text.decode_utf8` decodes the text files a person hands gelpy or edits (labels, stylesheet,
  recipe.json, fit_config.json) and names the file, the line and the encoding when one is not UTF-8.
  Polarity is auto-detected (skew + tail-asymmetry), confidence-flagged, and overridable
  (`--invert`/`image set`); detection runs once at init, and the frozen `ImageMeta` (folded
  into `image_hash`) drives a deterministic `apply_image_meta` replay in the cached
  pipeline. Downstream stays polarity-agnostic — signal is always bright.
  `contrast`, `lanes`
  (`resolve_lane_selector` implements the `int | "re:" | "label:" | literal` grammar;
  `detect_lane_centers` fits an even N-lane grid to the y-`column_projection` so lane
  placement is image-derived and crop-invariant, never a geometric guess — N is ground
  truth, the grid is robust to skipped wells, and a `confidence` flag marks when to verify
  the overlay / set endpoints manually), `line_profiles`.
  `_registry.Registry` replaces the old if/elif string dispatch.
  `core/background/` is RANSAC/robust-plane 2-D background removal.
  **`core/peaks/` is the tiered peak-fitting engine** (`docs/decisions/
  2026-07-14-robust-peak-fitting-design.md`). It needs no per-lane detection tuning:
  `baseline.py` (conservative arPLS-weighted **polynomial**, order 2 — tracks drift but
  can't bend to a smear) subtracts a baseline; `detect.py` seeds by residual-argmax on the current
  residual (`strongest_bump`) — overlapping peaks and shoulders separate out over greedy
  iterations, so an up-front shoulder seeder proved unnecessary and was removed in 1.0;
  `driver.py` is the shared robust-MAP
  `least_squares` fit (mixed shapes jointly, soft priors as penalty rows, **returns
  covariance**); shapes are `PeakModel`s in the `peak_registry` (`gaussian`, `emg` with
  **signed** `tau` = tail direction, `bi_gaussian`, `pseudo_voigt`) each with a
  **closed-form `area`** and role-tagged params. `strategies/` register in
  `fit_strategy_registry`: `grow_select` (default — greedy residual growth with an AICc
  stop, consolidation, per-component shape selection, near-duplicate collapse),
  `sparse_seed` (NNLS deconvolution seeding), `bayesian` (seeded Metropolis for credible
  intervals); `--auto-escalate` climbs that ladder automatically on any lane whose
  `confidence` stays `low`, persisting the escalated config to the `FitConfigStore`. Areas are on the **true-signal basis** (`area_fraction = ∫component / ∫(y −
  baseline)`, with a `residual_fraction`); `classify.py` labels each band
  `sample|well|smear|dust|unknown` (five values — `unknown` is the
  fallback; all are counted, never dropped); `quality.py` emits `FitQuality` +
  a `confidence` (`ok|low`) + machine-readable `reasons` (coverage band, R², per-band
  uncertainty/identifiability) — `status:"ok"` only means it converged. A failed/empty
  fit returns a `LaneFitResult(status=...)`, never a raised error.

- **`render/`** — the ONLY place matplotlib is imported (`Agg` set in `__init__`).
  Every renderer is a pure `data → PNG` (or markdown, or HTML) function; nothing calls
  `plt.show()`. `report.py` builds `report.md`; `html_report.py` builds the static HTML report
  (stdlib only: no script, no inline style, every text value escaped) from a `ReportContent` and
  the bytes of the images that `cli/_render.py` collects.

- **`runfolder/`** — run-folder format + caching + orchestration (the crux).
  - `stages.py` holds the hash chain: `image → background → profile[lane] → fit[lane]`.
    Every hash is computed from the recipe + small config objects (never the image),
    so deciding what to recompute is a few sha256-over-small-JSON ops.
    `profile_hash` deliberately excludes slice/normalization/label; those feed `fit_hash`.
  - `cache.py` — content-addressed `get_or_compute_*` with atomic writes (`utils/atomic.py`;
    pass an open file object to `np.save`/`np.savez` so they don't mangle the temp filename)
    and corrupt-file recovery. The gelpy version is part of every key (`stages.code_version`),
    not a hash of the code: an edit within one version reuses old caches.
  - `lock.py` — the advisory lock (`flock` on `<run>/.gelpy.lock`) that every writer holds. The
    commands that write and the renderers in `cli/_render.py` take it, a reader never waits for
    it, and a second writer gets `run_locked` at once. A new writer wraps itself in `locked` /
    `locked_run_dir`; tests fail for a command or tool that forgot. Taking the lock also refuses
    a run whose own folders (`layout.WRITTEN_FOLDERS`) are links (`RunPaths.require_real_folders`):
    gelpy never writes through a link inside a run folder. Files go through `atomic_write`, which
    replaces a leaf link; the two it opens in place (the lock, `log/history.jsonl`) go through
    `layout.open_no_follow`.
  - `pipeline.py` — each `ensure_*` checks its own cache before invoking its upstream
    (a lazy thunk), so a fit cache-hit never re-decodes the image or re-runs RANSAC.
  - `commands.py` — pure `f(request) -> result` (no argv/stdout). This is the seam the tool
    layer (`cli/tools.py`) wraps, which the MCP server serves. Includes `status`
    (cold-orientation snapshot, no recompute) and `detect_lanes` (image-based lane placement
    → `LaneDetectionResult`; `--apply` persists via the same `_build_layout` path as
    `set_lanes`; `set_lanes` also
    takes `endpoints` as the manual fallback). Per-lane fit intent is durable:
    `fit_run`/`refit_lane` seed
    `fit_config_io.py`'s `FitConfigStore` (last-write-wins) — it is NOT on the
    disposable manifest, so it survives a manifest wipe and is hand/agent editable.
  - `results.py` — assembles `results.csv`/`results.json` from cached fits (no recompute);
    `_fit_is_current` recomputes each lane's expected `fit_hash` from the store to drop
    stale fits after a recipe edit. A fit is current when its expected artifact is cached, whatever
    the ledger says: `follow_the_cache` re-points the in-memory records of a manifest that a
    reader (`status`, `results show`, the plots, the report) has just loaded, so an undone edit
    makes the older fits current again. `read_fit_result` reads a cache file and names it, never
    what is in it, when it does not validate.
  - `report_folder.py` — which folder an HTML report may be written into (new, empty, or one
    that carries the marker `report/.gelpy-report.json`; never the run folder or a folder around
    it), what a rebuild and `init --force` remove (only the files the marker lists), and the
    fingerprint `status` compares to say whether the report is current.

- **`cli/`** — thin typer layer. `app.py` (commands), `_io.py` (output + `parse_lane_term`;
  errors go through `core.errors.to_error_response` → `ErrorResponse` with a stable `code`,
  shared with `gelpy tool call` and the MCP server), `_render.py` (rendering orchestration:
  reads caches, calls `render.*`, keeps `commands` matplotlib-free), `tools.py` (the tool layer:
  each workflow step as one JSON-in/JSON-out `Tool`, the command plus the rendering its CLI step
  does, under one run lock; the typer commands, `gelpy tool list/call` and the MCP server all run
  it; the tool names are public API and change only with a major version), `mcp_server.py`
  (`gelpy mcp`: the stdio MCP server over `tools.py`, one call at a time in a worker thread; the
  optional `mcp` package is imported only when it starts). Every command that
  touches a run folder takes the run directory as an explicit **positional** argument
  (`gelpy fit run RUN_DIR ...`) — there is no ambient "current run", so a shell history is
  self-contained and replayable; `schema`, `models list`, `tool list` and `mcp` are the
  run-folder-free commands (`tool call` takes the run directory in its JSON).
  `_pick.py` (the `lanes pick` command) is the one deliberate exception to "matplotlib
  only in `render/`": it's an interactive human-in-the-loop lane-center picker, so it
  selects a *real* (non-`Agg`) backend itself. It's the last-resort manual fallback after
  `lanes detect` + overlay verification. Command tree: top-level `init`/`status`/`schema`/`mcp`
  plus sub-apps `lanes` (`set`/`detect`/`pick`), `contrast`, `background` (`remove`/`show`),
  `image` (`show`/`set`), `profiles extract`, `fit` (`run`/`refit-lane`), `results show`,
  `report html`, `models list`, `tool` (`list`/`call`).

## Run folder format

```
<run_dir>/
  recipe.json              # editable intent — precious
  fit_config.json          # per-lane fit intent — precious, editable (sparse; only fit lanes)
  manifest.json            # cache ledger — disposable, regenerable
  .gelpy.lock              # advisory lock, held while a command writes the folder
  cache/{image,background,profiles,fits}/...
  render/*.png, render/fits/lane_NNN.png, render/fits/index.json
  results.csv / results.json / report.md
  report/                  # gelpy report html: index.html, report.css, theme.css, img/
    .gelpy-report.json     # marks the folder as the report's and lists the files it wrote
  log/history.jsonl        # append-only command log
```

The source image is referenced by path + sha256, not copied in by default.

## Where documentation lives

- `docs/guide/` — hand-written narrative (the WHY). Freshness-stamped against its subject files.
- `docs/reference/` — GENERATED from the code. Never edit; run `python tools/docs.py build`.
- `docs/walkthrough/` — GENERATED by running the analysis (`tools/build_walkthrough.py`).
- `docs/decisions/` — historical design records, each stamped with its status. NOT current truth.
- `plugin/` — the distributable Claude Code plugin; the single copy of `SKILL.md` lives here.

## Conventions

- Adding a background model or peak **shape**: subclass the base, register with the
  registry decorator (`background_registry` / `peak_registry`); a shape needs a closed-form
  `area`, role-tagged params, and a `PeakParams` model. Adding a fit **strategy** (tier):
  register a `fit(profile, config, *, lane_index, label) -> LaneFitResult` in
  `fit_strategy_registry` and reuse `strategies.grow_select.refine` for finalization. Do not
  add if/elif dispatch.
  Adding a lane **projection strategy** (the fourth registry): register
  `fn(image, n_lanes, *, expected_pitch, y_start, y_end) -> AdaptiveProjection` in
  `core.lanes.projection_strategies`; it is adopted per image only if its comb-SNR beats
  full height by `_ADAPTIVE_MARGIN`, so a new strategy can never make detection worse.
- Adding a request/result field: put it on the pydantic contract in `schemas/`.
- Adding a tool (a workflow step an agent can call): a request model (a `RequestModel`) and a
  result model in `schemas/`, a handler in `cli/tools.py` that runs the command and its rendering
  under the run lock, and a `Tool` in its registry. `tests/cli/test_tools.py` pins the names:
  adding one is a minor change, renaming or removing one a major.
- Keep matplotlib imports inside `render/` (and `cli/_render.py`, which drives it; plus
  `cli/_pick.py`, the interactive picker, which selects a real GUI backend itself).
- **Documentation is verified, not trusted.** `docs/reference/` is generated
  (`python tools/docs.py build`) and diffed by a test; every fenced `bash` block in
  `docs/guide/` and `docs/walkthrough/` is executed against the example gel; load-bearing
  facts cite a `<!-- claim: id -->` asserted in `tests/docs/test_claims.py`; and each guide
  page carries an AST digest of its subject files, so changing a default or signature fails a
  test naming the page (`python tools/docs.py accept <page>` after re-reading it). Prose that
  the runtime can print belongs in a `Field(description=...)` or a typer `help=`, not in a
  doc. See `docs/guide/extending.md`.
- `pytest tests/` must stay green. It sat red for two months before 1.0; CI now enforces it.
- Tests are treated as heuristic guardrails, not ground truth; the e2e replay against
  the real TIFF is the primary verification.

## Data in examples, tests and docs

- Examples, tests and documentation use only the public example gels in `examples/data/` or
  the synthetic suite in `tests/synthetic/`. Never include names, sample identities, file
  names, paths or results from analyses of private data: not in code, comments, docstrings,
  docs, test names or commit messages.
- Describe a motivation generically ("a gel with a faint right third"), not by the analysis
  it came from.
- Analyse other gels outside this repository; run folders, `agent_analysis/` and reports are
  output, not source. `.gitignore` ignores `runs/`, `run/`, `agent_analysis/`, `report.pdf`
  and local agent folders. `tests/test_repo_hygiene.py` fails on home-directory paths in
  tracked text, on tracked `recipe.json` / `fit_config.json` / `history.jsonl` outside
  `tests/`, and on binary files outside `docs/walkthrough/img/` and `examples/data/`.
- Agents keep specs, plans and other working notes outside the repository.

## Deferred (designed for, not built)

Agent-guided auto-tuning (`tune`/`refit-all` write into
`FitConfigStore` — no new persistence), notebook scaffolding (a template sibling of
`render/report.py`; `AnalysisResult.to_dataframe()` is the notebook bridge), `replay` from
`log/history.jsonl`, MW/ladder calibration.
