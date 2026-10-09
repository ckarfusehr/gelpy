# Robust background fitting — design

> **HISTORICAL DESIGN RECORD -- SHIPPED** as `robust_plane`, now the default
> background model. Current behaviour: `docs/guide/background.md`.


Date: 2026-07-12
Branch: `feat/robust-background-fit`

## Problem

The only background model, `plane_ransac`, fits a 2D plane `z = a·x + b·y + c`
through **two hard-coded horizontal stripes** (rows `(20,+10)` and `(H-15,+10)`),
using RANSAC to reject band pixels that leak into those stripes, then subtracts the
plane and **clips negatives to zero**.

Three problems, in priority order:

1. **Hard-coded stripes make performance gel-specific.** The stripe rows are a guess
   about where background lives. When bands, ladders, or diffuse halos fall in those
   rows (framing-dependent), the "background" sample is contaminated. This is the
   main robustness failure.
2. **Clip-to-zero is a statistical bias.** After subtraction, true background pixels
   are symmetric noise centered on 0. Clamping the negative half rectifies that noise:
   its expectation becomes `+E|noise|/2 > 0`, larger where noise is larger. This injects
   a spatially-varying positive pedestal into exactly the regions meant to be zero, and
   destroys the only evidence of over-subtraction. Downstream Gaussian/EMG integrals
   then sit on and over-count that floor.
3. **A plane cannot represent curvature.** Transilluminator vignetting produces
   low-order curvature the plane misses.

## Grounding in real data

Three real gels (two not distributed with gelpy, plus the Cy3 example in `examples/data/`), robust low-envelope surface fit:

| Gel | Planar tilt (of dyn. range) | Quad improves bg-RMS | Robust σ (of range) | Signal pixels |
|---|---|---|---|---|
| gel A (not distributed) | x 0.1%, y 1.1% | 41.5% | 0.40% | 27% |
| gel B (not distributed) | x 0.2%, y 1.2% | 25.1% | 0.43% | 39% |
| 2023 cy3 scan        | x 1.0%, y 1.8% |  9.5% | 0.27% | 42% |

Findings that drive the design:

- **Background is offset + gentle gradient**, with real but weak tilt — a plane
  captures the bulk. Genuine low-order **curvature** exists (quad cuts residual 9–42%),
  but some of that gain is the quadratic bending into diffuse haze → curvature is an
  **opt-in**, not the default. Overfitting risk is real; plane stays default.
- **Smears/halos are always positive** (vertical trails off bands; broad sub-row halos).
  Signal only ever *adds*. This positivity asymmetry is the physical lever.
- **27–42% of pixels are signal.** The median/lowest-half is pulled up by signal, so the
  estimator must key on the **sub-background (negative-residual) side**, where only noise
  lives. This is the concrete argument for a probabilistic mixture that self-scales to
  the noise floor over a fixed-quantile threshold.
- Low-side residual histograms are near-Gaussian → a Gaussian background-noise
  likelihood is justified.

## Approach: one estimator, two weight modes (A is C's degenerate case)

Physical model, used throughout:

```
image(x,y) = B(x,y) + S(x,y) + noise,   noise ~ N(0, σ²)
  B = low-order polynomial surface   (order 1 = plane, default; order 2 = quad, opt-in)
  S = sample signal ≥ 0              (bands + smears; strictly non-negative)
```

Estimate `B` by **iterative reweighted least squares (IRLS / EM)**:

1. **M-step** — weighted least-squares fit of polynomial `B_θ` with per-pixel weights
   `w_i`. Plane = 3-column design `[1, x, y]`; quad = `[1, x, y, x², xy, y²]`. Solve the
   `k×k` normal equations `(AᵀWA)θ = AᵀW z`. Coordinates normalized to `[-1,1]` for
   conditioning.
2. **E-step** — recompute weights from the residual `r_i = z_i − B_θ(x_i,y_i)`:
   - **`weight_mode="asymmetric"` (A)** — Eilers ALS weight: `w = p` if `r>0` else `1`,
     with small `p` (default 0.02). No σ needed. Simple, fast, robust baseline.
   - **`weight_mode="probabilistic"` (C, default)** — two-component mixture
     `w_i = φ(r_i;0,σ²) / [φ(r_i;0,σ²) + π · U₊(r_i)]`, where `φ` is the Gaussian
     background likelihood, `U₊` a one-sided positive-signal density (exponential for
     `r>0`, 0 for `r<0`), `π` the signal prior. σ is estimated robustly **from the
     negative residual side** each iteration (`σ = 1.4826 · MAD` of `r[r<0]`, reflected).
     Yields `w≈1` for `r<0`, smoothly `→0` for bright pixels.
3. Iterate to weight convergence: stop when `max|Δw| < tol` or `max_iter` reached
   (≤ ~15 iterations in practice).

Both modes share the M-step and the surface model. C swaps in the probabilistic weight.
This is "A first, inside a C-ready harness that also runs A-only," with **no coupling to
lane-detection code** — the estimator stands alone on the positivity asymmetry.

**Optional contiguity prior (deferred behind a flag).** Bands/smears are spatially
contiguous; isolated hot pixels are not background. Blurring the weight field between
iterations (separable box/Gaussian, O(N)) makes weights spatially coherent. `off` by
default; add only if validation shows isolated-pixel or halo-edge failures.

**Determinism.** No RNG anywhere (RANSAC's `random_state` hack is gone). Same input →
same output, which the content-hash cache requires.

**Runtime.** M-step normal equations are O(N·k²), k∈{3,6}; weights are elementwise.
Optional `subsample` stride fits the smooth surface on every n-th pixel (lossless for a
low-order surface, evaluated on the full grid). Expected sub-second to ~1–2 s per gel;
`subsample` is the escape hatch for very large images.

## Cross-cutting fix: stop clipping the analysis array

- Cached corrected image = **signed residual** `image − B` (baseline centered at 0, no
  rectification bias, over-subtraction stays visible as negatives).
- `clip_negative: bool = False` config flag; when `True`, restores legacy clamp.
- The **renderer** clips for display only (already recomputes its own view), so figures
  stay non-negative without biasing the analysis path.
- Downstream check: peak fits run on a near-zero-centered baseline; validation confirms
  Gaussian/EMG fits are unharmed (expected: improved, since the floor is no longer
  rectified positive).

## Code changes

New, no breaking changes to the DAG/cache contracts:

- `core/background/robust_plane.py` — `RobustPlane(BackgroundModel)` + the IRLS engine.
  Engine split into small pure helpers: `polynomial_design(gx,gy,order)`,
  `weighted_lstsq(A,z,w)`, `asymmetric_weights(r,p)`, `probabilistic_weights(r,σ,π)`,
  `robust_sigma(r)`, and `fit_surface(image, params) -> (surface, weights, diagnostics)`.
  Registered as `"robust_plane"`.
- `schemas/background.py` — `RobustPlaneParams(CanonicalModel)`:
  `order:int=1`, `weight_mode:str="probabilistic"`, `asymmetry_p:float=0.02`,
  `signal_prior:float=0.3`, `max_iter:int=15`, `tol:float=1e-3`, `subsample:int=1`,
  `clip_negative:bool=False`, `contiguity:float=0.0`. All defaults → zero-config works.
- `schemas/recipe.py` / `schemas/background.py::BackgroundStageConfig.model` /
  `schemas/requests.py::RemoveBackgroundRequest` — default model → `"robust_plane"`.
  Generalize `RemoveBackgroundRequest` to carry `params: dict[str,Any]` (validated per
  model); keep `stripes` as a `plane_ransac`-only shim.
- `runfolder/commands.py::remove_background` — pass generic `params` through the
  registry validation (already does this; drop the stripes-specific branch to generic).
- `cli/app.py` `background remove` — expose common knobs (`--order`, `--mode`,
  `--no-clip/--clip`) plus `plane_ransac` back-compat; `--param k=v` passthrough.
- `render/background.py` — diagnostic generalized: show the fitted surface + the
  **weight map** (which pixels the fit trusted) instead of stripe overlays; keep the
  original/background/corrected profile slices. Weight map is the key new diagnostic —
  it makes "did it correctly ignore the smears?" eyeball-able.
- `cli/_render.py::render_background` — call the estimator to get surface+weights for the
  configured model (works for both models; stripe path retained for `plane_ransac`).
- `plane_ransac` stays registered for back-compat.

`stages.background_hash` already hashes validated params generically → no change needed;
switching model/params invalidates the background cache and everything downstream, as
designed.

## Testing strategy (efficient: minimal during build, ramp at validation)

**Synthetic (primary correctness proof).** Extend the `_synthetic_gel` helpers from
`tests/core/test_lanes.py` with a known background surface + noise + planted bands, and
assert the estimator recovers the surface within tolerance while excluding signal:

- Flat + tilted-plane background → recover coefficients; corrected background ≈ 0-mean.
- Curved (quadratic) background → `order=2` recovers it; `order=1` leaves bounded bias.
- **Heavy smear trails** (positive, contiguous, off-band) → excluded from the fit
  (weights low there; surface unaffected).
- **Band touching top/bottom edge** (the case that breaks fixed stripes) → still correct.
- **High signal fraction (>40%)** → probabilistic mode stays unbiased where asymmetric-
  with-loose-p or median would ride up.
- **Clip vs no-clip**: no-clip corrected background has ~zero mean (no positive pedestal);
  legacy clip shows the pedestal.

**Real (verification).** Run all three models (`plane_ransac`, `robust_plane` A,
`robust_plane` C) on the three gels; emit weight-map + corrected diagnostics; confirm the
surface sits at inter-lane level and smears/halos are excluded. Compare bg-RMS.

**Test discipline** (per request): minimal/no tests while building the engine; unit-test
the pure helpers once they stabilize; ramp synthetic edge-cases during validation; run the
full relevant suite (`pytest tests/core tests/runfolder`) only at the end, plus the e2e
replay if the TIFF is present.

## Out of scope

- Coupling to lane detection (explicitly rejected — keep background self-contained).
- Per-lane (1D) background models (requirement is a single uniform surface for all lanes).
- Making `order=2` or the contiguity prior the default (opt-in only, pending validation).
```
