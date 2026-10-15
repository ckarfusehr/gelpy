# Changelog

All notable changes to gelpy are recorded here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is
[semantic](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- Tests only: CLI tests run with colour off and a fixed width (one autouse fixture in
  `tests/cli/conftest.py`, the docs test sets the same), so they no longer fail under GitHub
  Actions, where rich and typer force colour and split `--n-lanes` with escape codes.

## [1.1.0] - 2026-10-15

Every failure now answers with one structured error that names the next step; the workflow
steps can be called as tools (`gelpy tool`, `gelpy mcp`); a run can be written as a static HTML
page. Caches written by 1.0 are not reused: the gelpy version is part of every cache key, so the
first command on an older run folder recomputes what it needs, and its fits show as stale until
they are run again. **1.1 needs numpy 2.0 or later**: an environment pinned to numpy 1.x cannot
install it (see Changed).

### Added

- `gelpy tool list [--json]` and `gelpy tool call NAME`: the thirteen workflow steps as tools that
  read one JSON object and write one. Each has an input and an output JSON Schema (registry names
  as enums), MCP-style annotations and its path arguments with their roles (`run`, `input`,
  `output`). Tool names are public API: they change only with a major version.
- `gelpy mcp`: the same tools over the Model Context Protocol (stdio). Needs
  `pip install "gelpy[mcp]"`. A server runs one call at a time, reads included, in the order the
  calls arrive; a call that is cancelled while it is still queued never runs.
  Ctrl-C stops the server at once (exit 130), and an installed `mcp` 1.x is refused with a
  `missing_dependency` that names `pip install -U "gelpy[mcp]"`. The text of an answer keeps
  non-ASCII characters, as the CLI prints them.
- `gelpy report html RUN [--css FILE] [--out DIR]`: a static report folder (`index.html`, a
  stylesheet, images) with no scripts and no inline styles; every text value is escaped. Styling
  uses `--gelpy-*` CSS variables; `--css` adds a stylesheet that overrides them.
- The report folder carries an ownership marker, `.gelpy-report.json`, which lists the files a
  build wrote. A build writes only into a folder that is new, empty or carries a marker, and a
  rebuild replaces only the files the marker lists; any other non-empty folder is refused.
- A marker is a plain file, so it is not trusted. It may list only names a build writes
  (`index.html`, `report.css`, `theme.css`, and PNGs in `img/` and `img/fits/`), and some targets
  are refused whatever marker they carry: the run folder, the folders inside it other than its own
  `report/`, the folders that hold it, the folders that hold a `recipe.json` or a `manifest.json`,
  and a default `report/` that is a link.
- `init --force` removes only the files the marker lists, and unlinks a `report/` that is a link
  without touching its target. The temporary files of a build that was killed
  (`.<name>.<random>.tmp`) are removed by the next build and by `init --force`.
- `status.report` is `{path, current}` for the run's own `report/`, and the page prints the
  fingerprint it was built from.
- `gelpy --version`. The version is recorded in `manifest.json`, `results.json`, `status` and
  every `log/history.jsonl` entry.
- `status` lists `stale_stages`: cached stages computed under an earlier recipe or version.
- Input limits, checked from the file header before decoding: 100 million pixels summed over the
  pages, and 64 pages. `GELPY_MAX_PIXELS` and `GELPY_MAX_PAGES` change them. The lane count is
  capped at 1000 (`n_lanes`, the count of `--linspace`, `--x-center`, the labels, the terms of a
  lane selector) in requests, in `recipe.json` and in `fit_config.json`: over it, a
  `validation_error` names the field and the limit, and a recipe with more lanes does not load.
- An advisory lock per run folder: a second process writing the same run gets `run_locked`.
  Reads (`status`, `results show`) never wait for it.
- `init --force`; `--allow-raw` on `profiles extract`, `fit run` and `fit refit-lane`.

### Changed

- Every failure -- malformed command lines (exit code 2), decoder errors and bare `ValueError`s
  included -- is one `ErrorResponse` on stderr, never a traceback. `detail` names the next step
  (for `validation_error` it still carries the field errors).
- `init` refuses a folder that already holds a run (`run_exists`) unless `--force`.
- Profiles and fits refuse a run without background removal (`background_required`) unless
  `--allow-raw`.
- Paths are recorded relative to the run folder where possible (the source image, cache entries),
  so a run folder works from any working directory and can be moved together with its image. Run
  folders written by 1.0 still open from the directory they were made in.
- `image show --kind` accepts only `raw`, `linear`, `non_linear` and `corrected`.
- Text cells of `results.csv` that start with `=`, `+`, `-`, `@`, a tab or a carriage return are
  written with a leading `'`.
- `report.md` names the source image by file name and sha256 instead of a path.
- `lanes detect` runs the detection once when it also renders the overlay; `robust_plane` with
  `subsample > 1` no longer builds full-resolution coordinate grids.
- **`numpy>=2.0`: breaking for an environment pinned to numpy 1.x.** gelpy calls `np.trapezoid`,
  which numpy 1.x lacks; 1.0 accepted `numpy>=1.24.3`. The packages with compiled code are floored
  at the oldest releases that have numpy 2 wheels (scipy 1.13, scikit-image 0.23.1, scikit-learn
  1.4.2, matplotlib 3.8.4; pandas 2.2.2 in the extras). Pillow and tifffile are declared
  dependencies. Text files are read and written as UTF-8.
- **`typer>=0.19`** (was 0.12): older typer cannot build its command tree, or reports a missing
  argument wrongly, against current click.
  CI runs the suite once more on Python 3.10 with the lowest declared floors.
- The `gelpy-analyze` skill is split into `judgment.md` (tool-agnostic) and `SKILL.md` (commands
  and tool names).

### Fixed

- The generated `docs/reference/` pages and the page digests no longer depend on the Python or click version: types render through one fixed mapping (`float | None`, `list[str]`, `text`, `integer`), and a digest spells `lambda:` the same on Python 3.10.
- `results show` names each lane it leaves out because its fit file is damaged, and the command
  that fits it again (`fit refit-lane`): in `warnings` of the JSON, on stderr for `--format csv`
  and `table`. `AnalysisResult` has the new field `warnings` (default empty).
- Exit code 130 for Ctrl-C is documented next to 0, 1 and 2, and `gelpy --version` is in the CLI
  reference. The known limitations name two warnings that are not about the data (a CJK glyph
  missing from matplotlib's font, Accelerate's `matmul` warnings with numpy 2.0 to 2.2 on arm64
  macOS) and the width `lanes set` keeps.
- The summary PDF example of the `gelpy-analyze` skill shows the lanes as set in its lane QC
  figure, not the last detection, and the CSV of its component-fit example is guarded against a
  bare carriage return on Python 3.10, as `results.csv` is. `judgment.md` says what a width that
  ran to a bound means and what to do.
- A run folder whose path is too long for the file system is `not_found` on Python 3.10 too (it
  was an `os_error` there).
- Prose: American spelling in the walkthrough, `image-input.md` and the agent prompts.
- Results carry the lanes' current labels; a lane relabelled after fitting kept its old label.
  Fit plots carry the current label too, and `fit run --plot`, `fit refit-lane --plot` and
  `report html` draw a plot again after a relabel.
- `refit-lane` named a plot file even when it rendered none.
- Requests refuse numbers that are not finite (`NaN`, `inf`). They used to be stored as `null`,
  and the run folder no longer loaded.
- Figures in `render/`, `report.md` and the HTML report are written atomically and never through a
  link, and the report refuses a plot that is a link.
- gelpy never writes through a link inside a run folder. `results.csv` is written atomically like
  the other files; `log/history.jsonl` and `.gelpy.lock` are opened without following a link, and
  one that is a link is refused; a link where `cache/`, `render/`, `log/` or a folder below them
  belongs is refused with `invalid_value` before anything is written (`init --force` still
  removes a linked `render/`). The run folder itself may be a link.
- `status` and `results show` judge a fit by whether the fit the recipe calls for is in the cache,
  as `stale_stages` always did: after an edit is undone the fits are current again at once, where
  `n_stale` and `results show` waited for a fit command. `results show` leaves out a current fit
  whose file does not validate, as `status` counts it missing, instead of failing; a fit record
  that names no lane is an `invalid_value` that names `manifest.json`. An error about a cache file
  names the file and never what it holds. A `recipe.json` or `fit_config.json` saved in another
  encoding is refused by name, line and encoding, not as a bug in gelpy. `lanes pick` answers
  `display_required` (no display) and `pick_cancelled` (window closed early), not `internal_error`.
  A deleted upstream cache file stays in `stale_stages` until a command that needs it runs.
- The report folder. A marker may list file names of at most 255 bytes, and a name the file system
  finds too long no longer stops a rebuild or `init --force`. A leftover of a killed build that
  cannot be removed (an immutable flag) is skipped, no longer aborting the build and every later
  one. A build never writes over an existing file its marker does not list: it is refused, with the
  files named. Pruning compares files, not names, so a marker that lists `img/LINEAR.png` no longer
  costs the new `img/linear.png` on a file system that ignores case. `--out RUN/report` follows the
  same link rule as the default, and a `.DS_Store` does not make a folder non-empty.
- Image input. A stack of three or four pages written with a length-1 axis, which tifffile 2026.5.2
  and later report without it, is counted by its pages and pixels again, so the input caps see it.
  `init`, `image set` and `status` carry what tifffile logged while it read the file in
  `warnings` (a stack whose page list is damaged comes back short, and only the log said so). A
  limit of 19 digits or more in `GELPY_MAX_PIXELS` / `GELPY_MAX_PAGES` is refused by name. An error
  no longer carries the folders of the image (Pillow's `cannot identify image file '<path>'`). A
  labels file or stylesheet with CR line endings names the right line. An image typed with `..`
  after a linked folder is recorded as the file that was read. A deflate TIFF that inflates past
  its header is a known limitation, documented in `known-limitations.md`.
- The skill's references to `lanes detect --no-apply`, to `--protect-well` and `--labels` (flags
  that do not exist) and to `InitRequest` (the contract is `InitRunRequest`), and its claim that
  `image show --kind raw` draws the lane overlay (only `--lanes` does).

## [1.0.0] - 2026-10-09

**1.0 is not compatible with 0.x.** The `Gel` object API is gone, not deprecated, and old
`.pkl` gel objects cannot be loaded. If you have 0.x notebooks,
[docs/guide/migrating-from-0.x.md](docs/guide/migrating-from-0.x.md) gives the command that
replaces each method.

gelpy is now a CLI-first, agent-native tool: a content-hash cached pipeline that turns a gel
image into per-lane band intensities in a browsable run folder. Every command takes the run
directory explicitly, emits JSON, and recomputes only what a change invalidated.

### Removed (breaking)

- **The `Gel` object and the notebook API.** `from gelpy import Gel`, `gel.setup_gel()`,
  `gel.show_line_profiles()`, `gel.save()` and `Gel.load()` no longer exist. The run folder
  replaces the pickled gel object as the thing you share: a directory of plain files
  (`recipe.json`, `results.csv`, `report.md`, `render/*.png`) that a collaborator can read
  without Python. Re-running any command reproduces the analysis exactly.
- **Five `fit` flags, without aliases:** `--model`, `--maxima-threshold`,
  `--maxima-prominence`, `--peak-width`, `--sigma`. The tiered fit engine chooses the peak
  count (AICc) and each component's shape itself, so there is no per-lane detection tuning,
  and a loose pass followed by a tight pass is no longer needed. The nearest current options
  are `--method` (strategy tier), `--shapes` (allowed shapes), `--baseline` / `--poly-order`
  and `--auto-escalate`; they control different things.
- **Every `results.csv` column name except `label` and `band_number`.** The columns are now
  `lane_index`, `label`, `band_number`, `component_class`, `shape`, `area`, `area_fraction`,
  `area_uncertainty`, `maximum_position`, `residual_fraction`, `confidence`, `status`,
  `method` and `param_*`. Code that reads the 0.x columns needs updating.
- `resources/ladders/*.png` and the unused `examples/data/folding_screen_RGB.tif` (~19 MB).
  MW/ladder calibration is still deferred; if it is added, the reference images can be
  recovered from git history.
- The pre-rewrite example notebook and `examples/cli_walkthrough.sh`. The notebook used the
  removed API; the script used the removed flags and aborted at line 26 of 42. Their
  replacements, [docs/walkthrough/](docs/walkthrough/) and `examples/walkthrough.sh`, are
  generated by running the commands they contain, and CI runs those commands again.
- `detect.py:ridge_candidates` / `second_derivative_centers`: implemented, never called.

### Added

- **Lane placement from the image.** `gelpy lanes detect` fits an even N-lane grid to a 1-D
  projection of the image with a matched filter and a soft `width/N` prior, renders a
  verification overlay, and reports a `confidence` with a `confidence_reason`. Row aggregation
  is pluggable (`--projection-strategy`). The default, the adaptive multi-band `d4_union`, is
  used on an image only when it beats full height, so it is never worse. `gelpy lanes pick` is
  the interactive fallback for a human.
- **A tiered peak-fitting engine** that needs no per-lane tuning: `grow_select` (greedy
  residual growth, AICc stop, shape chosen per component), `sparse_seed` (NNLS deconvolution
  seeding) and `bayesian` (credible intervals). `--auto-escalate` moves any low-confidence
  lane up these tiers. Four peak shapes (gaussian, EMG, bi-gaussian, pseudo-Voigt) are fitted
  jointly, each with a closed-form area.
- **Quality reporting.** Every band is classified (`sample|well|smear|dust|unknown`) and
  counted, never dropped. Area fractions are on the true-signal basis (relative to the lane's
  baseline-subtracted signal), with a `residual_fraction`. Each lane carries a `coverage` and a
  `confidence` (`ok|low`) with machine-readable `reasons`. `status: "ok"` only means the fit
  converged; `confidence` is the quality signal.
- **`robust_plane` background removal** as the default: an IRLS low-order surface fit, with no
  hand-placed stripes, unbiased when bands cover much of the image. `plane_ransac` remains as
  a fallback. `gelpy background show` renders the surface, the weights and before/after
  slices.
- **Uniform image input.** `canonicalize_image` converts any dtype, RGB(A) or multi-page
  source to a canonical float64 `[0,1]` image, signal-bright and finite. Polarity is
  auto-detected with a confidence flag, recorded in the run, and can be overridden
  (`init --invert`, `image set`).
- `gelpy status`: an overview of a run (source, background, polarity, per-lane fit and stale
  state) that recomputes nothing.
- `fit_config.json`: per-lane fit settings, sparse and editable by hand or by an agent. It is
  kept apart from the disposable manifest, so it survives a cache wipe.
- For agents: `gelpy schema <Model>` prints any contract, `gelpy models list --kind
  fit|shape|background` lists the registries, and errors carry a stable `code`.
- **Documentation checked by tests.** `docs/reference/` is generated from the code; every
  command block in the guide and the walkthrough is executed by `tests/docs`; key facts cite a
  claim id that a named test asserts; and each guide page records a digest of the public
  surface it describes, so a changed default or signature flags the page for review.
- **A Claude Code plugin** (`plugin/`) with the `gelpy-analyze` skill and the example scripts it
  adapts (component fit, PDF report), plus `tools/sync_plugin.py` to vendor it into a plugin
  marketplace repository.
- GitHub Actions CI that runs the test suite and the documentation checks.

### Fixed

Found before the release by checking every document against what the code does. Most of these
bugs were in development versions of 1.0, not in 0.0.4.

- `image show --kind corrected` after `background remove`, two commands the README documented,
  crashed with `ValueError: Image Correction methods work correctly only on images with
  non-negative values`. The corrected image keeps signed residuals on purpose; the display now
  clamps them, and the cache stays signed.
- `image show --kind raw` was not raw: it took the same gamma path as `non_linear`, so the two
  images differed only in their title. `raw` now applies no display contrast.
- `gelpy status` failed with `validation_error` on any run with a lane whose fit came back
  `empty`: `LaneStatus.status` could not represent a value that `LaneFitResult.status` allows.
- `gelpy lanes detect --smoothing-window <any value>` crashed with `TypeError: 'float' object
  cannot be interpreted as an integer`. The value is a float throughout, but only the default
  was rounded to an integer; now every value is.
- `gelpy schema ImageMeta | SetImageRequest | RobustPlaneParams` reported that the contract did
  not exist, because the three were missing from `schemas.__all__`. A test now checks that
  every model is reachable through `gelpy schema`.
- `_SMOOTHING_WINDOW_FRACTION` is 0.7, but its comment and two schema descriptions said 0.35.
  Measured on `folding_screen`: 0.7 scores 0.98 composite with 0% false-ok and 0.019×pitch
  error; 0.35 scores 0.94 with 2.67% false-ok and 0.044×pitch.
- `core/peaks/baseline.py` said the default baseline is arPLS (it is order-2 `poly`), and the
  `BaselineConfig` docstring contradicted its own `poly_order` field about order 3. Both now
  carry the measured smear-absorption table: `none` 0.0%, poly-0 2.1%, poly-2 5.5%,
  poly-3 13.1%, `arpls` 34.4%, `asls` 58.5%.
- `tests/synthetic/harness.baseline_detect` inherited the module default instead of pinning
  `projection_strategy="full"`, so two tests compared the shipped detector with itself and
  could not fail. Three other lane tests had never passed since they were added: they asserted
  a sub-pixel grid phase that was never reached. They now assert the measured, pitch-relative
  accuracy, and the numbers are in `docs/guide/known-limitations.md`.
- The source distribution shipped the example TIFFs, and a blanket `*.csv` in `.gitignore`
  silently kept every results file out of git.

### Changed

- The version is single-sourced from `src/gelpy/__init__.py`; `pyproject.toml` reads it via
  hatch.
- Dependencies: pydantic and typer are new; pandas, seaborn and ipykernel moved to the
  optional `notebook` extra.
- Documentation lives in `docs/` (guide, generated reference, walkthrough, decisions), with an
  mkdocs-material config. Historical design specs are labelled as history, so they are not
  mistaken for current behaviour.

## [0.0.4] - 2024-09-27

Final release of the notebook-driven `Gel` API, marked deprecated at the time. Superseded by
1.0.0; see the migration guide.
