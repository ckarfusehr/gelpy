---
title: Extending gelpy
audience: both
summary: The exact contract a new peak shape, fit strategy, background model or lane projection strategy must satisfy, the layering rules that constrain where it lives, and how the documentation checks react when you change any of it.
subject:
  - src/gelpy/core/peaks/base.py
  - src/gelpy/core/background/base.py
subject_digest: 11cb0fcc83773a66
---

## Four registries

Every pluggable family is a `core._registry.Registry`: a name maps to a factory, an optional
pydantic config model, and a one-line description. Selection is always a registry lookup —
nothing dispatches on a model name to *choose* an implementation.

| family | registry | selected by | entry is |
|---|---|---|---|
| peak shape | `core.peaks.peak_registry` | `fit run --shapes` | a `PeakModel` subclass, instantiated by the caller |
| fit strategy | `core.peaks.strategies.fit_strategy_registry` | `fit run --method` | a plain function, called directly |
| background model | `core.background.background_registry` | `background remove --model` | a `BackgroundModel` subclass, instantiated |
| lane projection | `core.lanes.projection_strategies` | `lanes detect --projection-strategy` | a plain function, called directly |

Two hand you a class, two a callable. Check which you are writing before copying a skeleton.

Three call sites do special-case a name *after* selection, for diagnostics or seeding:
`cli/_render.render_background` (a `plane_ransac`-only stripe overlay),
`runfolder/commands.remove_background` (a `plane_ransac`-only `stripes` shim) and every fit
strategy, which prefers `gaussian` as its seed shape when that shape is allowed. None of them
decide *which* model runs.

Registration happens at **import time** and there is no entry-point plugin discovery (nothing
reads `importlib.metadata`). A new entry must therefore live in the source tree and be imported
by its family's package `__init__` — `core/peaks/__init__.py`,
`core/peaks/strategies/__init__.py`, `core/background/__init__.py` — or, for a projection
strategy, be defined in `core/lanes.py`, the only module that registers one.

Current entries: [reference/registries.md](../reference/registries.md), or at the terminal:

```bash
# doc-test: skip
# The doc test sets these two; set them yourself if you are following along.
export IMAGE=examples/data/folding_screen.tif
export RUN=/tmp/gelpy-extending
```

```bash
gelpy models list --kind shape --format text
gelpy models list --kind fit --format text
```

Projection strategies are the one family `models list` cannot show — it accepts only `fit`,
`shape` and `background`:

```bash
# doc-test: expect-failure
gelpy models list --kind projection
```

## Rules that are not negotiable

- **Never add `if/elif` dispatch on a model name.** Register instead. Among the pluggable
  *model* families the 1-D baseline estimator is the one still not a registry: a `Literal` in
  `schemas/baseline.py` plus an `if`-chain in `core/peaks/baseline.estimate_baseline`. That is
  the exception, not the pattern to copy. Three smaller option sets are string-dispatched too —
  profile normalization (`core/line_profiles.normalize_profile`), channel collapse
  (`core/image_io._collapse_with`) and contrast mode (`core/contrast.rescale_linear`) — but
  those are closed enumerations, not extension points. If what you are adding is a model, it
  goes in a registry.
- **A new request or result field goes on the pydantic contract in `schemas/`**, never into a
  loose dict. Contracts are `extra="forbid"`, so an unknown field is a hard `ValidationError`
  instead of a silently dropped key.
- **Layer direction is `schemas/` → `core/` → (`render/` + `runfolder/`) → `cli/`, never
  upward.** Nothing violates it today: no module under `core/` imports `runfolder`, `render` or
  `cli`; `runfolder/` imports no `render`, which is what keeps `runfolder/commands.py`
  matplotlib-free, so the command API (which the tool layer in `cli/tools.py` wraps) imports
  without the plotting stack; `schemas/` imports nothing at module
  level but pydantic, typing and its own siblings (`AnalysisResult.to_dataframe` lazy-imports
  pandas inside the method, which is what keeps the package importable without it).
- **matplotlib is imported under `render/` only.**
  `tests/docs/test_claims.py::test_claim_matplotlib_only_in_render` scans the package and
  exempts `render/`, `cli/_render.py` and `cli/_pick.py`. Of those two exemptions only
  `cli/_pick.py` actually imports it (it selects a GUI backend for the interactive lane
  picker); `cli/_render.py` orchestrates rendering without importing matplotlib.
- **Be deterministic.** Cache keys are sha256 over small JSON config objects, never over
  results: `fit_hash` folds in the whole `FitConfig` dump, `background_hash` folds in your
  *validated* config-model dump. Randomness must come from `FitConfig.seed` (as
  `strategies/bayesian.py` does), or your strategy returns different answers under one key.
- **The cache key carries the package version, not a hash of the code.** `image_hash` folds in
  `gelpy.__version__` and every other hash chains off it, so a new version never reuses an old
  cache. An edit to the code of an existing entry within one version keeps its key, and a run
  folder that already holds its results reuses them: bump the version, or delete the run folder's
  `cache/` (and `manifest.json` with it), after you change what an existing entry computes.

## A peak shape

`core/peaks/base.py` is the entire contract: seven abstract methods, two optional overrides.
`area(*params)` must be **closed form** — gelpy never numerically integrates your shape for
you, and that number is what `results.csv` reports. (The one numerically-integrated `area` in
that file is the smear region fallback, which emits `shape="region"` and no params; it is a
property of `assemble_result`, not of your shape.) `single_peak` must be non-negative for a
non-negative amplitude.

Of the two overrides, `peak_maximum_x` defaults to the `center` parameter — override it for an
asymmetric shape, `_grid_argmax` exists for that. `effective_width` defaults to the first
`width`-role parameter, or 1.0 if the shape has none, and drives **windows, not labels**: the
±3·width window `assemble_result` cuts around each component (which gates its shape-R² and so
the smear region-integral fallback), `_common._identifiability`'s pairwise-overlap flag, and
`grow_select`'s proposal, consolidation and prune windows. Get it wrong on a tailed shape and
every one of those is mis-sized, silently.

It does *not* classify. `BandFit.component_class` (`sample|well|smear|dust|unknown`) comes from
`core/peaks/classify.py`, which uses its own shape-agnostic width, `area / (height · √(2π))` —
the sigma of a Gaussian with this component's area and height. Deliberately so: a broad smear
the fitter described as a narrow-σ pseudo-Voigt still has a large area/height ratio, so the
label survives a bad shape choice. (`base.py`'s `effective_width` docstring still says "for
classification". It is stale; the code above is what runs.)

Roles are why the driver never branches on your shape: `driver._component_bounds` and
`driver._prior_rows` read `param_roles()` and nothing else.

| role | bounds | prior |
|---|---|---|
| `amplitude` | `[0, inf)` | — |
| `center` | seed ± `max(2*seed_scale, 4.0)` px, clipped to the profile | — |
| `width` | `[MIN_WIDTH, max_width]`, `max_width = max(max_width_frac * length, 2*MIN_WIDTH)` | log-normal, anchored to the seed scale |
| `tail` | `±max_width`, signed (the sign is the tail direction) | shrinkage toward 0 |
| `unit` | `[0, 1]` | — |

An unrecognized role string is not an error — it falls through to `(-inf, inf)` with no prior.
If your parameter is a width, tag it `width`, or it is neither bounded nor anchored.

```python
# src/gelpy/core/peaks/lorentzian.py
import numpy as np
from ...schemas.peaks import LorentzianPeakParams, PeakParams
from .base import PeakModel, peak_registry


@peak_registry.register("lorentzian", description="Pure Lorentzian peak.")
class LorentzianFitModel(PeakModel):
    kind = "lorentzian"

    def params_per_peak(self) -> int: return 3
    def param_names(self) -> list[str]: return ["amplitude", "center", "gamma"]
    def param_roles(self) -> list[str]: return ["amplitude", "center", "width"]

    def single_peak(self, x, amplitude, center, gamma):
        return amplitude * gamma**2 / ((x - center) ** 2 + gamma**2)

    def area(self, amplitude, center, gamma) -> float:
        return float(np.pi * amplitude * gamma)

    def initial_guess(self, center, height, scale) -> list[float]:
        return [float(height), float(center), float(scale)]

    def build_params(self, params) -> PeakParams:
        return LorentzianPeakParams(amplitude=params[0], center=params[1], gamma=params[2])
```

The class alone is not enough — five more things have to line up:

1. **`schemas/peaks.py`** — a `CanonicalModel` with `kind: Literal["lorentzian"]`, one field per
   fit parameter, added to the `PeakParams` discriminated union. Skip it and the fit runs to
   completion, then dies in assembly: `BandFit` rejects the params with `Input tag 'lorentzian'
   ... does not match any of the expected tags` (verified by registering the shape above
   in-process). Note it *raises*: "a bad lane returns `LaneFitResult(status=...)` instead of
   raising" covers data-dependent failure, not a mis-declared shape.
2. **`param_names()` must equal the pydantic field names, in order** (minus `kind`).
   `render/fits._component_curve` rebuilds a component as
   `tuple(float(dumped[name]) for name in model.param_names())` and `to_rows` emits one
   `param_<field>` column each. That is `param_names()`'s only consumer in the package, so a
   mismatch surfaces as a `KeyError` at plot time and nowhere earlier. All four built-ins
   agree; nothing enforces it.
3. **`schemas/__init__.py`** — list it in `__all__` (which
   `test_claim_every_schema_is_discoverable` asserts for every `CanonicalModel` under
   `schemas/`) *and* import it: `gelpy schema` getattrs each `__all__` name off the package, so
   an entry with no import passes the test and fails at runtime.
4. **`schemas/fit_config.DEFAULT_SHAPES`** — the default `--shapes` menu; a shape outside that
   tuple is selectable but never tried by default.
5. **`tests/core/test_peaks.py`** asserts `set(peak_registry.names())` *exactly*, so adding a
   shape fails it until you update it. Add yours to `test_closed_form_area_matches_numeric`
   too: that list is hand-written, not parameterized over the registry, and a wrong closed-form
   area is the one bug that silently corrupts every number gelpy reports.

Caveat: `schemas/peaks.ShapeKind` is a `Literal` of the four built-in names that nothing
imports. Dead today, so forgetting it breaks nothing — and nothing tells you it went stale.

## A fit strategy

Register a function with exactly this signature; `runfolder/pipeline` looks it up by
`FitConfig.method` and calls it with the sliced, normalized profile.

```python
# src/gelpy/core/peaks/strategies/single_peak.py
import numpy as np
from ....schemas.fit_config import FitConfig
from ....schemas.peaks import LaneFitResult
from ..driver import Component
from ._common import assemble_result
from .grow_select import prepare, refine
from .registry import fit_strategy_registry


@fit_strategy_registry.register("single_peak", description="One component at the global argmax.")
def fit(profile: np.ndarray, config: FitConfig, *, lane_index: int, label: str) -> LaneFitResult:
    profile, baseline, signal, x, noise, models, kwargs = prepare(profile, config)
    model = models.get("gaussian") or next(iter(models.values()))
    i = int(np.argmax(signal))
    seed = Component(model, tuple(model.initial_guess(float(i), float(signal[i]), 4.0)), float(i), 4.0)
    final = refine(signal, x, [seed], config, models, noise, kwargs)
    return assemble_result(
        profile, baseline, final, config, lane_index=lane_index, label=label, method="single_peak"
    )
```

Reuse all three helpers; do not reimplement them. `prepare` baseline-subtracts, estimates the
noise sigma, instantiates the allowed shapes and builds the driver kwargs from the priors.
`refine` consolidates, selects a shape per component, collapses near-duplicates, prunes and
joint-refits — all three built-in tiers funnel through it, so tiers differ only in how components
are *proposed*, never in how they are finalized. `assemble_result` does true-area accounting
(`area_fraction = area / ∫(y − baseline)`), per-band uncertainty from the covariance,
classification, the smear region-integral fallback, quality, and the `confidence` vector
everything downstream reads.

Registering that strategy in-process and running it on a stated synthetic — length 400, two
Gaussians of σ = 10 at 140 and 260, amplitudes 1.0 and 0.6, flat 0.02 offset, no noise:

```text
single_peak: status=ok  bands=1  coverage=0.625  confidence=low
             reasons: ['coverage 62%: 38% of the signal is unmodeled', 'poor fit (R2=0.682)']
grow_select: status=ok  bands=2  coverage=1.000  confidence=ok
```

`status="ok"` only means the `least_squares` call converged, and 0.625 is not a quality score:
`coverage = Σ band area / ∫(y − baseline)`, so with one kept band it reduces to that band's
share of the signal — here 1.0/(1.0 + 0.6), the two σ being equal. What the tier gets for free
is the flagging: a deliberately under-fitting strategy comes back `confidence = "low"` with
machine-readable reasons, and you wrote none of that.

Two things a new tier does not get for free. `--auto-escalate` ignores it:
`runfolder/commands._TIER_LADDER` is the hard-coded `["grow_select", "sparse_seed", "bayesian"]`,
and a method outside that list falls to index 0, so escalating from it restarts at
`sparse_seed`. And `--method`'s help string enumerates the three built-ins by hand while
`tests/docs/test_claims.py` asserts `set(fit_strategy_registry.names())` exactly — both need
editing. Strategy-specific knobs go in `FitConfig.params` (`dict[str, Any]`): no built-in reads
it today, but it is folded into `fit_hash`, so whatever you put there is cached correctly. The
`config_model=` slot is unused by this family, which is why `models list --kind fit` reports an
empty schema for every tier.

## A background model

`core/background/base.py`: a `name`, one `remove` method, one result container.

```python
# src/gelpy/core/background/flat_median.py
import numpy as np
from ...schemas.background import FlatMedianParams
from .base import BackgroundFitResult, BackgroundModel, background_registry


@background_registry.register(
    "flat_median", config_model=FlatMedianParams,
    description="Subtract one global percentile level.",
)
class FlatMedian(BackgroundModel):
    name = "flat_median"

    def remove(self, image: np.ndarray, params: FlatMedianParams) -> BackgroundFitResult:
        level = float(np.percentile(image, params.percentile))
        plane = np.full_like(image, level)
        return BackgroundFitResult(
            corrected_image=image - plane, background_plane=plane, params={"level": level}
        )
```

This is the one family where `config_model=` matters, and it earns its keep three times:
`runfolder/commands.remove_background` validates incoming params against it,
`runfolder/stages.background_hash` hashes the *validated* dump (so a hand-edited `recipe.json`
cannot miss the cache), and `models list --kind background` prints its JSON schema. Give every
field a default and a `Field(description=...)`: zero-config is the intended path, and that
description is what both a user and the generated reference read.

- `runfolder/pipeline.ensure_background_image` caches **only** `corrected_image` — its compute
  thunk returns `{"corrected": result.corrected_image}`, so `params`, `weights` and
  `inlier_fraction` never reach the cache. A new diagnostic scalar has nowhere to live yet — it
  needs a field on a `schemas/` contract and a change to that compute thunk.
- `cli/_render.render_background` special-cases `plane_ransac` by name for its stripe overlay
  and sends every other model down a generic path that re-runs `remove` and plots
  `background_plane`, `weights` and `corrected_image`. Fill in `background_plane` and the
  diagnostic figure works; fill in `weights` (per-pixel background trust in `[0, 1]`) and it
  also shows which pixels your fit trusted.
- Do not clip negatives. `RobustPlaneParams.clip_negative` defaults to `False` deliberately:
  clamping rectifies noise into a positive pedestal that biases every downstream integral.

## A lane projection strategy

The fourth registry, and the one the repository's own `CLAUDE.md` conventions section leaves
out of its list of extension points. A projection strategy decides
how the 2-D image collapses to the 1-D signal that `detect_lane_centers` fits an even N-lane
grid to. It is a function, keyword-only after `n_lanes`, returning an `AdaptiveProjection`:

```python
# in src/gelpy/core/lanes.py (the only module that registers these)
@projection_strategies.register("top_third", description="Project over the top third only.")
def _strategy_top_third(
    image: np.ndarray, n_lanes: int, *,
    expected_pitch: float, y_start: int | None, y_end: int | None,
) -> AdaptiveProjection:
    lo = 0 if y_start is None else y_start
    hi = image.shape[0] if y_end is None else y_end
    cut = lo + (hi - lo) // 3
    return AdaptiveProjection(column_projection(image, lo, cut), y_band=(lo, cut))
```

`AdaptiveProjection` carries the `projection` plus two optional diagnostics the overlay
surfaces: `row_weights` (the per-row weights you used) and `y_band` (a `(y_start, y_end)`
window, or a list of them). Return them if you have them — they are what makes a wrong
detection debuggable.

Your projection is not taken on trust. `detect_lane_centers` always computes the `"full"`
full-height baseline, runs the same detection core on both, scores each with `_comb_snr`, and
adopts your candidate only if it beats full height by `_ADAPTIVE_MARGIN` (0.02, in normalized
projection units) **and** does not turn a full-height `confidence == "ok"` into `"low"`.
Otherwise the result silently reverts, and `projection_source` says which projection won.

Registering the strategy above and detecting on two synthetic gels — both 300 × 600, six
σ = 12 px lane Gaussians on a 100 px pitch, amplitude 0.6 over a 0.05 floor, Gaussian noise
σ = 0.02; gel A has its bands in rows 20–280, gel B in rows 20–90 with a diagonal streak
(amplitude 0.9, σ = 60 px, sweeping x from 40 to 560) filling every row below:

```text
gel A (bands full height):   full 0.992  ->  source=full        (candidate had no headroom)
gel B (bands top third +
       diagonal streak):     full 0.332  ->  source=top_third   (adopted at 0.814)
```

Both detections keep `confidence = "ok"` and land within 3 px of the true centers. The point is
the fallback, not the numbers: demonstrating a new strategy needs a gel where the full-height
mean is genuinely worse, and verifying it means reading `projection_source`, not the centers.
`tests/core/test_lane_strategy.py`'s `_band_gel` is a readable construction of exactly that
case, and its fixture pops anything registered during a test back out of the registry so it
cannot leak into the next one.

## Smoke-testing a new entry

A wrong name is never a silent no-op: `fit run --method nope` exits 1 with a `registry` error
whose `detail` names the next step and whose `available` lists the registered names:

```console
{
  "error": "Unknown fit-strategy model 'nope'.",
  "code": "registry",
  "detail": "Use one of the names in `available`.",
  "available": [
    "bayesian",
    "grow_select",
    "sparse_seed"
  ]
}
```

Beyond that, one lane exercises a shape, a strategy and a background model end to end —
substitute your own names:

```bash
gelpy init "$RUN" "$IMAGE" --n-lanes 20 --format text
gelpy background remove "$RUN" --format text
gelpy lanes detect "$RUN" --projection-strategy full --format text
gelpy fit run "$RUN" --lane 0 --shapes gaussian,emg --method grow_select --format text
```

Watch `recomputed` to confirm your entry is hashed into the chain: repeating the last command
unchanged reports `"recomputed": []` and reuses image, background, profile and fit. Judge the
fit by `coverage`, `confidence` and `confidence_reasons` — `status` only says it converged.

## Keeping the documentation honest

Four mechanisms guard the docs; three of them can name your change.

**The reference is generated.** `docs/reference/cli.md`, `contracts.md`, `registries.md`,
`errors.md` and `run-folder.md` come out of the code, carry a `<!-- GENERATED ... -->` first
line, and are byte-compared by `tests/docs/test_reference_is_current.py`. A new registry entry,
contract field or CLI option therefore needs a rebuild. Never hand-edit those files.

```bash
# doc-test: skip
# Not executed by the doc test: it rewrites files in the repository.
python tools/docs.py build
```

**Every guide page carries a freshness digest.** `subject:` lists the source files a page
describes; `subject_digest` hashes their *public surface* — module, class and function names,
full signatures including annotations and defaults, docstrings, literal module- and class-level
assignments, and `Field(...)` text. Function bodies, comments and definition order are
deliberately invisible. When `tests/docs/test_freshness.py` names your page, re-read it against
its subjects, fix what is now false, and only then re-stamp:

```bash
# doc-test: skip
# Also not executed: it rewrites the page's front matter.
python tools/docs.py accept docs/guide/extending.md
```

To see what moved, `python tools/docs.py surface <files>` prints the text being hashed and
`digest` prints the hash — here, this page's own subjects:

```bash
python tools/docs.py digest src/gelpy/core/peaks/base.py src/gelpy/core/background/base.py
```

Re-stamping without reading defeats the mechanism, which is why the digest ignores edits that
cannot falsify prose.

**Claim tests cover what no digest can see.** The digest moves for a changed signature, default
or `Field` description. It does not move when behaviour *inside* a function changes — if
`area_fraction` switched denominators while keeping its name, no digest would notice. That is
what `tests/docs/test_claims.py` is for: prose states the fact with an inline
`<!-- claim: some-id -->` and `test_claim_some_id` asserts it against the code, with a bijection
test forbidding a cited claim with no test or a test with no citation. Rule of thumb: a
signature, a default or a registered name needs no claim (the digest and the generated
reference own it); a behavioural invariant a reader would act on does.

**Every command block is executed.** `tests/docs/test_examples_execute.py` runs the ```bash
blocks of each page under `docs/guide/` and `docs/walkthrough/` in document order, as one shell
per page, against `examples/data/folding_screen.tif`, with `$RUN` and `$IMAGE` preset. A block
that cannot run headless starts with `# doc-test: skip`; one that must fail starts with
`# doc-test: expect-failure`. Transcripts go in ```text or ```console fences, which are never
executed. A static sweep in the same file parses the CLI and fails on any option mentioned
anywhere in the prose — inline code, a table, a skipped block — that the CLI does not have.

Two consequences worth internalizing:

- **Put prose where the runtime can print it.** A pydantic `Field(description=...)`, a typer
  `help=`, a registry `description=`: each reaches the user through `gelpy schema <Model>`,
  `--help` and `models list`, *and* is copied into the generated reference automatically. The
  same sentence in a hand-written page has to be maintained by hand. Write it once, in the code.
- **Every source module must be some page's subject.**
  `test_every_source_module_is_described_by_some_page` fails when a new module belongs to no
  page, so a subsystem cannot appear without someone deciding where it is explained. Add it to
  the closest page's `subject:` list (which moves that page's digest, so re-read the page), or
  to the exemption list in that test with a reason.
