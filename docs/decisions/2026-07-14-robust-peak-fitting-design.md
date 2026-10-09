# Robust peak-fitting engine (line profile → area fractions)

> **HISTORICAL DESIGN RECORD -- SHIPPED WITH DIVERGENCES**. Read the
> divergences before trusting any specific step:
>
> - Step 1 specifies an arPLS baseline retained inside the joint fit. What shipped
>   **subtracts** a conservative order-2 **polynomial** before fitting; arPLS was measured
>   absorbing ~34% of a broad smear against the polynomial's 5.5%.
> - The problem statement asserts the profile still sits on a structured, drifting baseline
>   after 2-D correction. On real gels where `robust_plane` did its job it does not; see
>   `2026-07-15-baseline-redundancy-and-well-boundary.md`.
> - `detect.py`'s second-derivative shoulder seeder was built, never wired into any
>   strategy, and removed in 1.0. Seeding is residual-argmax only.
>
> Current behaviour: `docs/guide/fit-engine.md`.


Status: approved design, in implementation. Branch `feat/robust-peak-fitting`.
**No backward compatibility** — the old `find_peaks`+`curve_fit` path, `FitType`, and legacy
CSV columns are removed, not preserved. Tests are rewritten to the new contract.

## Problem

A per-lane 1-D profile `y(t)` (intensity vs migration row), after 2-D background removal +
slice + normalization, contains an *unknown* number of overlapping components of
*heterogeneous* shape in the same trace:

- symmetric Gaussian-like bands,
- exponentially-tailed (EMG) bands (smearing / non-specific interaction),
- broad shallow **smears** (large width, low amplitude) — genuine signal, matter for yield,
- a narrow sharp **well/pocket aggregate** near the migration origin — genuine, valuable signal,
- **dust** spikes (a few px) — noise,

all on a **structured (non-flat, drifting/curved) baseline** even after 2-D correction.

Today's engine (`core/peaks/fitting.py`) sets N = number of `find_peaks` local maxima on a
smoothed trace, fits one shape (gaussian XOR emg) to all of them, caps widths with one global
5× multiplier, discards `curve_fit` covariance, and reports each area as a fraction of the
*fitted* total. Shoulders are structurally unfittable; there is no quality metric, no baseline
model, no artifact labeling, and missed signal silently rebalances the fractions to sum to 1.

## Confirmed decisions

1. **AUC basis = true integrated area + explicit residual.** Each component's area is a
   fraction of `A_sig = ∫(y − b)`, and a lane-level `residual_fraction = 1 − ΣA_k/A_sig`
   exposes unmodeled signal. Fractions never silently rebalance.
2. **Inclusive — never remove signal.** Everything is modeled and counted. `component_class ∈
   {sample, well, smear, dust, unknown}` **labels** components; it does not exclude them. The
   well is a labeled, counted band. Dust stays in unless a future robust classifier removes it
   (prefer flag-over-remove).
3. **Smears are signal, not baseline.** Baseline is conservative (true drift only). A
   continuous smear is **one broad component** with its own area fraction (the yield number).
   Hybrid representation: try a broad parametric shape; if it fits poorly, fall back to
   integrating the contiguous smear region directly.
4. **Principled redesign, robustness is THE requirement.** Iterative/algorithmic/Bayesian
   complexity is acceptable. Per-lane compute budget: target ≤5 s, hard cap 30 s. A higher
   tier that fits the budget may become the default.
5. **Validation = synthetic ground-truth profiles (known areas, structured baselines,
   convolved/overlapping peaks, sharp well, smears, dust) + curated real lanes + eyeball.**

## Area accounting (the correctness core)

Over the fit window `t = 0..L−1`:
- baseline `b(t)` (low-DOF, arPLS prior refined jointly),
- signal `S(t) = y(t) − b(t)`, total `A_sig = ∫S dt` (trapezoid),
- model `M(t) = Σ_k c_k(t)`, each `c_k ≥ 0`, `A_k = ∫c_k dt` (closed-form parametric; trapezoid
  for a region-integrated smear),
- `area_fraction_k = A_k / A_sig`, `coverage = ΣA_k/A_sig`, `residual_fraction = 1 − coverage`.

## Architecture (respects `schemas → core → render/runfolder → cli`)

`core/peaks/` gains a strategy layer, mirroring the `projection_strategies` registry built for
lane detection:

- `base.py` — `PeakModel` ABC + `peak_registry` (kept, extended).
- shapes: `gaussian.py`, `emg.py` (tail-shrinkage), `bi_gaussian.py`, `pseudo_voigt.py`.
  Each: `single_peak`, **closed-form `area`**, analytic Jacobian where feasible, param-index
  metadata, prior hooks, `build_params`.
- `baseline.py` — arPLS / AsLS (scipy.sparse Whittaker + iterative reweight) + polynomial;
  well-region protection.
- `detect.py` — multi-scale Ricker CWT + 2nd-derivative seeding → `(position, scale)`
  candidates incl. shoulders and broad smears; and `strongest_residual_bump(...)` used by the
  greedy loop.
- `driver.py` (was `fitting.py`) — the shared robust-MAP `least_squares` engine: composite
  model (peaks + low-DOF baseline), robust loss (`soft_l1`), soft priors implemented as
  appended penalty residuals, non-negativity bounds, returns params **and covariance**.
- `quality.py` — `FitQuality` + confidence/reasons.
- `classify.py` — position/width → `component_class`.
- `strategies/` — `fit_strategy_registry`: `grow_select` (Tier 1), `sparse_seed` (Tier 2),
  `bayesian` (Tier 3). Signature `fit(profile, config, *, lane_index, label) -> LaneFitResult`.

1-D baseline is **fit-time** (per-lane, coupled to peaks); its config feeds `fit_hash`. The
2-D background stage is unchanged.

## Data model (`schemas/`, no back-compat)

- `peaks.py`:
  - Shape params: `GaussianPeakParams`, `EmgPeakParams`, `BiGaussianPeakParams`,
    `PseudoVoigtPeakParams` (discriminated by `kind`). `PeakParams = Union[...]`.
  - `BandFit`: `band_number`, `component_class`, `area` (absolute), `area_fraction`,
    `area_uncertainty`, `maximum_position`, `centroid`, `params: PeakParams | None`
    (None for region-integrated smear), `region: (lo,hi) | None`, `shape_r2`, flags
    (`at_bound`, `low_identifiability`).
  - `LaneFitResult`: `lane_index`, `label`, `status ∈ {ok, empty, failed}`, `error`,
    `method`, `bands`, `baseline: BaselineFit`, `coverage`, `residual_fraction`,
    `signal_area`, `quality: FitQuality`, `confidence ∈ {ok, low}`, `confidence_reasons`.
    `to_rows()` emits the new columns (no legacy names).
  - `FitQuality`: `reduced_chi2`, `r2`, `residual_autocorr` (Durbin–Watson), `noise_sigma`,
    `n_components`, `aicc`, `bic`.
- `baseline.py` (new): `BaselineConfig(method ∈ {arpls, asls, poly, none}, lam, p, poly_order,
   protect_well)`, `BaselineFit(method, coeffs|None, values_hash|None)`.
- `fit_config.py`: `FitConfig(method: str = "grow_select", baseline: BaselineConfig,
   shapes: list[str] = [...allowed], priors: PriorConfig, seed: int = 0,
   params: dict[str, Any] = {})` validated per-method via the registry. `PerLaneFitConfig`
   holds `profile: LineProfileConfig` + `fit: FitConfig`. `FitConfigStore` unchanged in shape.
- `results.py`: `RESULT_COLUMNS` updated; `AnalysisResult` unchanged in shape.
- `requests.py`: `FitRunRequest`/`RefitLaneRequest` carry `FitConfig` + overrides + `method` +
  `auto_escalate`.

## Tier 1 — `grow_select` (default, deterministic)

1. Baseline: arPLS (conservative `lam`) → `S`; keep a low-DOF baseline in the joint fit.
2. Greedy growth: start empty. Each round, `strongest_residual_bump(S − M)` (multi-scale) →
   if strength > `k·σ_noise`, seed a component at its position and scale, joint robust-MAP
   refit, compute AICc; accept iff AICc improves, else stop. `σ_noise` from MAD of the
   high-frequency residual.
3. Consolidation (anti-shatter): merge adjacent components whose between-valley depth is below
   threshold (one smooth hump), drop components with amplitude→0 or huge area-uncertainty;
   re-fit, keep iff AICc not worse.
4. Per-component shape selection: try allowed shapes on each component, pick by local AICc.
5. Final joint robust-MAP refit with soft priors (width log-normal around seed scale,
   amplitude ≥ 0, tail shrinkage) → covariance.
6. Classify (`well/smear/dust/sample`); hybrid smear (parametric → region-integral fallback);
   compute areas, fractions, residual, quality.

## Tier 2 — `sparse_seed` (deconvolution seeder, deterministic)

NNLS (`scipy.optimize.nnls`) over a shifted multi-width Gaussian dictionary → non-negative
spike map → cluster into `(position, scale)` seeds → hand to the grow_select
refit/consolidate/shape-select/quality machinery (skip greedy discovery). For heavily
overlapped lanes where Tier-1 seeding is ambiguous.

## Tier 3 — `bayesian` (escalation, seeded → reproducible)

Seeded sampler (emcee, optional dep `gelpy[bayesian]`, lazy import) around the grow_select
structure; priors: non-negativity (bounded), width log-normal, tail half-normal shrinkage.
Robust N via per-N comparison (N*−1..N*+1) by WAIC/AICc. Posterior means + credible intervals
on `area_fraction`. RNG seeded from `FitConfig.seed` → cacheable. Benchmark per-lane time; if
≤5 s eligible as default, else escalation-only (≤30 s cap).

## Escalation UX

`fit run --method grow_select` (default). Agent reads `quality`/`confidence`, then
`refit-lane --method sparse_seed|bayesian` on flagged lanes. Optional `--auto-escalate`
climbs the ladder until confidence `ok` or max tier (bounded, logged). Off by default.

## Quality vector → confidence + reasons

`coverage`, `residual_fraction`; reduced χ² (MAD noise); **residual autocorrelation**
(Durbin–Watson) to catch structured residual when χ² looks fine; per-peak area/width
uncertainty (covariance), `at_bound` flag; **identifiability** (design-matrix condition
number) → `low_identifiability`. → `confidence: ok|low` + machine-readable `reasons`
(e.g. "structured residual near t≈420: likely unmodeled shoulder"; "peaks 2&3 not separately
identifiable").

## Caching / determinism

`stages.fit_hash` folds the full `FitConfig` (method + baseline + priors + shapes + `seed`).
Tier 3 is seeded → deterministic → cacheable like every other stage.

## Validation harness (`tests/synthetic/`)

- `profiles.py` — `ProfileSpec` + deterministic generator with ground-truth areas: structured
  baseline (poly + smooth drift), K components of chosen shape/position/width/area, sharp
  near-origin well, broad smears (known area), dust spikes, shoulder-separation ladder,
  convolved/overlapping doublets, read noise.
- `fit_conditions.py` — `SUITE` grouped by stressor (clean band; doublet-resolvability ladder;
  shoulder; tailed; smear+sharp yield case; well+bands; dust-on-band; low-SNR; heavy baseline
  drift; many-peak ladder).
- `fit_harness.py` — scoring: area-fraction error (components matched to truth), N-recovery +
  over/under-split rate, residual-fraction accuracy, flag correctness (`false_ok` worst,
  `false_low` lesser), shape-ID accuracy, position error, per-tier timing. `evaluate` +
  `format_table`.
- `fit_real.py` — curated real lanes from `examples/data/*.tif` + eyeball annotations +
  contact-sheet renderer.

Acceptance: `grow_select` beats a naive baseline on area error + N-recovery + flag-correctness
across the SUITE; `false_ok ≈ 0`; smear-shatter below threshold; Tiers 2/3 help hard cases
without regressing easy ones; real profiles don't regress and the yield case recovers the
band/smear split; same spec+seed → byte-identical (incl. seeded Tier 3).

## Render / CLI

`render/fits.py`: baseline curve + per-component shapes colored by `component_class` + residual
panel + quality/confidence annotation (matplotlib stays in `render/`). CLI `fit run` /
`refit-lane` gain `--method`, baseline/prior/shape flags, `--auto-escalate`; `models list`
shows fit strategies + shapes.

## Build order (task list #1–13)

schemas → shapes → baseline → driver → seeder → grow_select → quality → harness (iterate
grow_select to pass) → sparse_seed → bayesian → wire (runfolder/cli/render, benchmark, set
default) → full verification.

## Risks → mitigations

- Smear shatter → consolidation + broad multi-scale seeding + AICc; harness metric guards it.
- Baseline eating a smear → conservative baseline + well protection; explicit smear-on-drift case.
- EMG/Voigt overflow → errstate guards + bounded/shrinkage params + analytic forms.
- Tier-3 compute → flagged-lanes-only, seeded, cached; never in the default broad pass.
- Doublet over-split → identifiability flag marks speculative splits.
