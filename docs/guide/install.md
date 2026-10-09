---
title: Install
audience: both
summary: Create an environment, install the checkout editable, and verify it with five checks whose output you can compare against this page.
subject:
  - src/gelpy/__init__.py
  - src/gelpy/__main__.py
subject_digest: dcd97f6098ac18e8
---

## What you need

- **Python >= 3.10.** That is the `requires-python` floor declared in `pyproject.toml`. CI
  (`.github/workflows/ci.yml`) runs the suite on Python 3.11 and 3.12, so the declared floor
  itself is never exercised: 3.10 is a declaration, not a tested guarantee.
  Everything measured on this page was measured on Python 3.14.4.
- **A checkout, not PyPI.** The newest release on PyPI is 0.0.4 (checked 2026-09-10), which is
  the pre-rewrite `Gel` object API. `pip install gelpy` therefore gets you 0.x, which is
  incompatible with everything in this documentation.
- The scientific stack arrives as declared dependencies (numpy, scipy, scikit-image,
  scikit-learn, matplotlib) plus pydantic and typer. The version floors are in
  `pyproject.toml`; do not trust a copy of them written anywhere else.

## Install

```bash
# doc-test: skip
conda create -n gelpy python=3.11
conda activate gelpy
cd /path/to/GelPy               # the checkout

pip install -e .                # package + runtime dependencies
pip install -e ".[dev]"         # ...and pytest, to run the test suite
pip install -e ".[notebook]"    # ...and pandas/seaborn/ipykernel, for your own notebooks
```

The documentation test suite does not execute that block: it would install into the
environment running the tests. Every block below it is executed, in order, against the real
example gel.

Two notes on the extras:

- `.[dev]` declares pytest, pandas, nbclient, nbformat and mkdocs-material. Nothing under
  `tests/` or `tools/` imports the last four on this checkout, so what the suite actually needs
  is pytest. The extra is wider than the suite.
- `.[notebook]` installs libraries, not a feature. It exists so `AnalysisResult.to_dataframe()`
  (`src/gelpy/schemas/results.py`, pandas lazy-imported) works and so you can plot in a
  notebook of your own. Notebook *scaffolding* — a command that writes one for you — is not
  implemented.

## Verify

Set two variables once, in one shell, from the repo root (`$IMAGE` below is a relative path).
Every block after this one uses them:

```bash
export IMAGE="${IMAGE:-examples/data/folding_screen.tif}"
export RUN="${RUN:-/tmp/gelpy-install-check}"
```

The `:-` defaults are there because the documentation test harness sets both itself; if you
already have them set, this leaves them alone.

**1. The entry point exists, and what version it is.**

```bash
gelpy --help
python -c "import gelpy; print(gelpy.__version__)"
```

Expect the command tree and `1.0.0`. There is no `--version` flag on the CLI — `gelpy
--version` exits 2 with "No such option". Ask Python for the version instead; `pyproject.toml`
reads it out of `src/gelpy/__init__.py`, so there is only one copy of it.

**2. Which install you are actually running.**

```bash
which gelpy
python -c "import gelpy; print(gelpy.__file__)"
```

```text
/path/to/env/bin/gelpy
/path/to/GelPy/src/gelpy/__init__.py
```

`gelpy` and `python -m gelpy` are the same code path — `src/gelpy/__main__.py` calls
`cli.app.main`, which is also the `[project.scripts]` entry point. They can nonetheless be
*different installs*: `gelpy` is whatever is first on `PATH`, `python -m gelpy` is whatever the
interpreter you named can import. With more than one environment around, prefer `python -m
gelpy`: it targets the checkout you are testing. gelpy's own test suite invokes it that way for
exactly that reason (`tests/docs/conftest.py`).

The second line printing a path inside your working tree is what "editable" means: edits and
`git pull` take effect with no reinstall. A changed dependency list still needs one.

**3. The scientific stack loads and the registries are populated.**

```bash
python -m gelpy models list --kind fit
```

```text
{
  "names": [
    "bayesian",
    "grow_select",
    "sparse_seed"
  ],
  ...
}
```

Three strategy names means numpy, scipy, scikit-image and scikit-learn all imported cleanly —
bringing up the CLI pulls in the whole fit engine. Matplotlib is *not* in that set: it is
imported only when something renders, so a broken matplotlib surfaces at your first plot, not
here. `--kind shape` and `--kind background` list the other two registries; see
[reference/registries.md](../reference/registries.md).

**4. It decodes a real image.**

```bash
gelpy init "$RUN" "$IMAGE" --n-lanes 20
gelpy status "$RUN"
```

```text
{
  ...
  "image_width": 4688,
  "image_height": 1364,
  ...
  "polarity": "bright_on_dark",
  ...
  "n_lanes": 20,
  "n_fit": 0,
  ...
}
```

This reads the 16-bit TIFF, auto-detects polarity, and writes `recipe.json`. Nothing is fitted
yet — `n_fit: 0` is correct here. To go further, follow
[analyze-a-gel.md](analyze-a-gel.md).

**5. Schema discovery works with pydantic alone.**

```bash
python -c "
import sys, gelpy.schemas
heavy = [m for m in ('numpy', 'scipy', 'matplotlib') if m in sys.modules]
print('heavy modules pulled in:', heavy or 'none')
print('contracts available:', len(gelpy.schemas.__all__))
"
```

`gelpy.schemas` imports standalone — no numpy, scipy or matplotlib — so a thin client, an MCP
server say, can do schema discovery and validation on its own.
<!-- claim: schemas-import-without-scientific-stack -->
That is a deliberate constraint on the package, not an accident of import order; a test asserts
it in a subprocess.

## Run the tests

A cheap check that the install is coherent:

```bash
python -m pytest tests/schemas tests/cli -q
```

```text
27 passed in 3.20s
```

The count is the part to compare; the seconds are not reproducible to two digits.

The whole suite is `pytest tests/ -q`. Measured on Python 3.14.4, 2026-09-10:

- Everything except `tests/docs/` — the layers that test the code — is **199 tests, all
  passing, in about 6.5 minutes**. Three tests are ~90% of that wall clock: the lane-strategy
  comparison (`tests/core/test_lane_strategy_d4.py::test_d4_harness_composite_at_least_matches_baseline`,
  ~4 min), the end-to-end replay (`tests/e2e/test_replay.py`, ~1 min) and the real-image fit
  smoke test (`tests/synthetic/test_fit_real.py`, ~50 s). Everything else is seconds. So if you
  want a fast loop, name the layer you are working on rather than waiting on `tests/`.
- Read those durations as an order of magnitude, not a benchmark. Repeated runs of the same
  tests on the same machine and interpreter spread by 10-40% here, and the count itself moves
  whenever a test lands — three arrived in the two commits before this measurement, taking it
  from 196 to 199. If your numbers differ, that is not a broken install.
- The end-to-end replay drives the real 20-lane TIFF through the whole pipeline and is the
  primary verification; the unit layers are guardrails, not ground truth.
- `tests/docs/` verifies the documentation against the code — it executes every command block,
  regenerates the reference and diffs it, and asserts each cited claim. It parametrizes over
  pages, so the collected total grows as documentation is added.
- The two example TIFFs are tracked in git (about 17 MB) but excluded from the sdist
  (`[tool.hatch.build.targets.sdist]` in `pyproject.toml`), so tests that need them run from a
  checkout and skip when they are absent.
- `pytest -m "not slow"` deselects the documentation block replay
  (`tests/docs/test_examples_execute.py`) and nothing else — it is the only test carrying the
  marker. In particular it does **not** skip the ~1-minute end-to-end replay, despite the
  marker's description in `pyproject.toml` ("end-to-end runs against the real example TIFFs").

## Coming from 0.x

1.0 is not compatible with 0.x. The `Gel` object API is gone, not deprecated; pickled `.pkl`
gel objects cannot be loaded; every `results.csv` column except `label` and `band_number` was
renamed; five `fit` flags were deleted with no aliases. The per-method replacement commands are
in [migrating-from-0.x.md](migrating-from-0.x.md), and the full list of removals is in
[CHANGELOG.md](../../CHANGELOG.md).

## If it does not work

- `gelpy: command not found` — the entry point is installed with the package. Re-run
  `pip install -e .` in the *active* environment, or use `python -m gelpy`, which needs no
  entry point.
- `python -m gelpy` fails with `No module named gelpy` while `gelpy` works — you have two
  environments. Compare `which gelpy` against `python -c "import gelpy; print(gelpy.__file__)"`.
- A command runs but the numbers look wrong: that is not an install problem. Start at
  [troubleshooting.md](troubleshooting.md).
