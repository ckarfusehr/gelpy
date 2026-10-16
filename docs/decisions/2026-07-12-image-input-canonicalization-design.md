# Robust, agent-first image input — design

> **HISTORICAL DESIGN RECORD -- SHIPPED.** This is what was intended, not a description of
> current behaviour. For how image input works now, see `docs/guide/image-input.md`.


Date: 2026-07-12
Branch: `feat/robust-image-input` (proposed)

## Problem

All image input flows through `core/image_io.load_image(path)`:

```python
image = io.imread(path)
if image.ndim == 2:
    return image                      # native dtype/range, UNCHANGED
if image.ndim == 3 and image.shape[2] in (3, 4):
    if image.shape[2] == 4: image = image[:, :, :3]
    return color.rgb2gray(image)      # float64 in [0,1], luminance-weighted
raise ValueError(...)                 # anything else
```

It is called at init (`recipe_io.build_image_ref`, which records `ImageRef{sha256,H,W,dtype}`)
and by `pipeline.ensure_image` (cached as `.npy`). Three problems, in priority order:

1. **Polarity is a hidden, load-bearing, unchecked global assumption.** Every downstream
   stage assumes **signal = high pixel value** (bright band on dark background):
   - `column_projection` + `detect_lane_centers` → `find_peaks` finds *maxima* (`lanes.py:489`)
   - `robust_plane`: "signal only ever *adds* intensity, so background lives on the low
     side"; `robust_sigma` keys on `residual < 0` (`robust_plane.py:59-70`)
   - `plane_ransac`: `corrected[corrected < 0] = 0` (`plane_ransac.py:70`)
   - gaussian/emg peak models fit maxima

   This is correct for fluorescence/SYBR/GelRed (bright-on-dark). It is **silently wrong**
   for the very common Coomassie/silver-stained gel photographed or scanned in transmitted
   light (dark-on-light): the background surface fits *onto the bands*, peaks vanish or land
   wrong — all while reporting `status:"ok"`. There is no `--invert`, and nothing tells the
   agent which polarity is expected.

2. **Scale is inconsistent across formats.** Grayscale input keeps native magnitude
   (uint8 0–255, uint16 0–65535, float, signed int); RGB(A) collapses to float [0,1]. The
   same physical gel as a 16-bit TIFF vs. an RGB PNG yields band areas differing by ~65535×.
   Scale-dependent defaults (`--maxima-threshold 0.00025` in the README) are meaningful for
   exactly one scale.

3. **Fragile channel handling and unhandled inputs.** `rgb2gray`'s luminance weights
   (0.21R + 0.72G + 0.07B) are physically arbitrary for a gel. Multi-page/stacked TIFFs,
   non-finite pixels (NaN/Inf poison percentiles and fits), indexed/palette PNGs, 1-bit,
   signed int, and EXIF-rotated phone photos are unhandled or misinterpreted.

**Goal (agent-first):** deterministic code that Just Works when it can, and — when it must
guess — records what it assumed and raises a confidence flag, so the using agent intervenes
*only* when flagged. Same idiom as `detect_lane_centers`.

## Design decisions (settled)

1. **Polarity:** auto-detect + record + confidence flag + explicit override. Never invert
   silently on low confidence; a `low` result emits an ignorable warning.
2. **Canonical representation:** a single **2-D float64 array in [0, 1]**, dtype-aware via
   `img_as_float`, signal-bright, finite. Produced for *every* input.
3. **Colored-RGB default channel strategy:** **mean** (equal channel weight), overridable.
   Rationale: the intensity→ground-truth mapping of a colored image depends on the dye
   absorption / pseudo-color LUT, which is unknowable from the image alone, so default to the
   least-biased estimate — an equal-weight mean, not perceptual luminance (human-eye weighting
   has no physical meaning for quantitation). A dominant-channel pick is rejected as the
   *default* because it goes non-monotonic under a colormap (e.g. viridis' blue channel peaks
   at both low and mid values). `luminance`, single-channel, `dominant`, and `max` remain
   available as overrides. Genuine single-dye fluorescence should be loaded from its native
   grayscale TIFF, where this branch never runs.

The consequence of (2): **downstream (`background`, `lanes`, `peaks`) is untouched** — its
"signal is bright, scale is [0,1]" assumption is now always true by construction. No model
changes, no scale-dependent-threshold surprises.

## Grounding: the physics of polarity

A gel image is `pixel = background + signal`, where background occupies the **majority of
the image area** (bands/wells/smears are a spatial minority) and signal is one-sided.

- **Bright-on-dark** (fluorescence): background low, signal adds → a large low-value mode
  (background) plus a thin high-value tail (signal) → **positive skew**; the bright tail
  reaches further from the background level than the dark tail.
- **Dark-on-light** (transmitted-light stain): background high, signal subtracts → large
  high-value mode plus a thin low-value tail → **negative skew**; the dark tail reaches
  further.

So two independent, deterministic signals detect polarity, and their **agreement** is the
confidence:

- **Skew sign** — Fisher skewness `g1 = E[(X−μ)³]/σ³`.
- **Tail asymmetry about the background level** — with background level `m = median(X)`
  (robust; background dominates area), `bright_reach = p99.5 − m`, `dark_reach = m − p0.5`,
  `tail_asym = (bright_reach − dark_reach)/(bright_reach + dark_reach)`. Positive → signal is
  the bright tail.

**Decision rule (conservative — only invert when confident):**

```
strong = |g1| >= SKEW_FLOOR or |tail_asym| >= TAIL_ASYM_FLOOR
if sign(g1) == sign(tail_asym) and strong:
    polarity   = bright_on_dark if g1 > 0 else dark_on_light
    confidence = "ok"
else:
    polarity   = bright_on_dark          # unsure -> the reversible "do nothing" default
    confidence = "low"; reason = "skew/tail-asymmetry disagree or too weak (near-symmetric)"
inverted = (polarity == dark_on_light)   # inversion applied ONLY on a confident dark-on-light
```

Starting constants `SKEW_FLOOR = 0.5`, `TAIL_ASYM_FLOOR = 0.1`, percentiles `p0.5 / p50 /
p99.5` — tuned/confirmed on real data during implementation (the same magic-constant-
validated-on-real-gels style as `lanes.py`). A `low` result **emits an ignorable warning**
(see *Warning surfacing*) naming the reason and the `--invert/--no-invert` override; the code
never blocks and never inverts on `low`. Known low-confidence cases (correctly flagged, not
silently wrong): a heavily-loaded gel where signal ≈ half the area (near-symmetric), or a
scanned stain with a dark frame around a light gel.

## Approach: a canonicalization layer in `core/image_io`

### Contract

```python
canonicalize_image(path, *, channel="auto", invert="auto", page=0)
    -> tuple[np.ndarray, ImageMeta]      # detection path (used at init / image set)

apply_image_meta(path, meta: ImageMeta)
    -> np.ndarray                        # frozen, deterministic replay (used by pipeline)
```

`canonicalize_image` **detects** and returns the canonical image plus the frozen decision;
`apply_image_meta` re-derives the identical canonical image from a stored `ImageMeta` with
**no detection** — so detection happens exactly once (at init), the recipe's `ImageMeta` is
authoritative, and the cached pipeline is fully deterministic. `load_image(path)` is kept as a
thin `canonicalize_image(path)[0]` wrapper for convenience/back-compat callers.

### Pipeline inside canonicalization

1. **Decode + page** — decode; for a multi-page TIFF select `page` (default 0) and record
   `source_pages`. A page count > 1 is **not** a hard error (page 0 is a sane default) but
   **emits an ignorable warning** (see *Warning surfacing*) so the agent/user sees that N pages
   exist and can re-select with `--page`.
2. **dtype → [0,1]** — `skimage.util.img_as_float` maps uint8/uint16/int/float uniformly.
3. **Non-finite guard** — any NaN/Inf → **hard error** with a count. Silently sanitizing a
   scientific image is worse than stopping.
4. **Channel collapse** — drop alpha. If channels are equal within tolerance
   (`p99.9 of per-pixel (max−min) < GRAY_ATOL=0.02`) → `gray_rgb`, collapse by mean
   (lossless). Else → `channel` (default `mean`; overrides `luminance`, `r`/`g`/`b`,
   `dominant`, `max`). Record `channel_strategy` + `channel_index`.
5. **Polarity** — run the detector above unless `invert` is `True`/`False` (manual override:
   `polarity_source="manual"`). Apply `1 − x` iff `inverted`.

Result is guaranteed 2-D float64, finite, in [0,1], signal-bright.

## Persistence, hashing, determinism

- **New `ImageMeta` on the recipe** — *interpretation intent*, editable and precious,
  sibling to `ImageRef` (file identity). Fields: `original_dtype`, `original_shape`,
  `source_channels`, `source_pages`, `page`, `channel_strategy`, `channel_index`,
  `polarity` (`bright_on_dark`|`dark_on_light`), `inverted` (bool),
  `polarity_source` (`auto`|`manual`), `polarity_confidence` (`ok`|`low`), `polarity_reason`.

- **`image_hash` must fold in the interpretation fields.** Today `image_hash(recipe) =
  recipe.source_image.sha256` (`stages.py:33`) — keyed only on raw bytes, so changing
  `--invert`/`--channel` would *not* invalidate the cached `.npy` (a latent correctness bug).
  Fix, mirroring `background_hash`'s validated-params hashing:

  ```python
  def image_hash(recipe):
      return hash_fields(source=recipe.source_image.sha256,
                         channel_strategy=recipe.image.channel_strategy,
                         channel_index=recipe.image.channel_index,
                         inverted=recipe.image.inverted, page=recipe.image.page)
  ```

  Every downstream hash chains off `image_hash`, so this propagates through the whole DAG for
  free — editing the interpretation recomputes exactly what depends on the pixels.

- **Determinism** — the detector is pure/RNG-free; the cached path never detects (it replays
  a frozen `ImageMeta`). Satisfies the content-hash cache's determinism requirement.

## CLI + agent surfacing

- `init … --channel {auto|mean|luminance|r|g|b|dominant|max} --invert/--no-invert/--auto
  --page K`.
- **`image set`** (new, mirrors `lanes set`): re-run detection (`--auto`) or set
  `channel`/`invert`/`page` on an existing run, so an agent corrects a flip cheaply without a
  full re-init. (The recipe is hand-editable too — editing `image.inverted` in `recipe.json`
  invalidates by hash, consistent with the "precious, editable" recipe philosophy — but
  `image set` also offers re-detection and validation.)
- **`init`/`image set` result and `status`** carry the polarity/channel decision +
  `polarity_confidence`, and `schema ImageMeta` makes the contract discoverable. This is the
  agent-first payoff: the deterministic code decides, and the agent intervenes *only* when a
  warning fires.

### Warning surfacing (non-fatal, ignorable)

The deterministic path never blocks on these; it surfaces them and proceeds, mirroring
`set_lanes`' existing overlapping-strips warning (`summary.lanes["warning"]`):

- `RunUpdateSummary` gains `warnings: list[str]` (human/agent-readable lines) and an
  `image: dict[str, Any]` block (the structured polarity/channel decision). `init` and
  `image set` populate both.
- `RunStatus` gains `polarity`, `polarity_confidence`, and `warnings: list[str]`, so a cold
  `status` call re-surfaces an unresolved `low`-confidence flip or a multi-page file without
  recompute.
- Two conditions raise a warning: `polarity_confidence == "low"` (reason + `--invert` hint)
  and `source_pages > 1` (page in use + `--page` hint). The agent or user may ignore either.

## Code changes

- `schemas/image.py` — add `ImageMeta(CanonicalModel)` (fields above). Keep `ImageRef`.
- `schemas/recipe.py` — add `image: ImageMeta`, with a legacy-equivalent `default_factory`
  (`inverted=False`, `polarity=bright_on_dark`, `channel_strategy="mean"`, `page=0`) so
  recipes predating this field remain valid and never auto-flip.
- `schemas/run.py` — `RunUpdateSummary` gains `warnings: list[str]` + `image: dict[str,Any]`;
  `RunStatus` gains `polarity`, `polarity_confidence`, `warnings: list[str]`.
- `core/image_io.py` — the layer: `decode_image(path, *, page)`, `to_float01(arr)`,
  `collapse_channels(arr, strategy, *, gray_atol)`, `detect_polarity(gray01) ->
  PolarityDecision`, `canonicalize_image(...)`, `apply_image_meta(path, meta)`; `load_image`
  becomes the thin wrapper.
- `core/errors.py` — `NonFiniteImageError`, `UnsupportedImageError`; map through
  `to_error_response` with stable codes.
- `runfolder/recipe_io.py` — `build_image_ref` → also build `ImageMeta`
  (`build_image_ref_and_meta(path, *, channel, invert, page)`).
- `runfolder/pipeline.py::ensure_image` — call `apply_image_meta(path, recipe.image)` instead
  of `load_image(path)`; keep the `SourceImageChangedError` check.
- `runfolder/stages.py::image_hash` — fold in the interpretation fields (above).
- `runfolder/commands.py` — `init_run` threads `channel/invert/page`, returns the polarity
  decision + confidence + warnings; `status` includes them; new `set_image` command.
- `schemas/requests.py` — `InitRunRequest` gains `channel/invert/page`; new `SetImageRequest`.
- `cli/app.py` — `init` options above; new `image set` under the `image` sub-app.
- `render/*` — no change; renderers already `imshow(cmap="gray")` and now always get
  signal-bright [0,1].

## Migration / back-compat

- The [0,1] switch **intentionally rescales** absolute intensities for existing runs
  (uint16 → [0,1]); **relative within-image quantitation is unchanged** (a uniform linear
  scale cancels in every ratio/normalization). Adding fields to `image_hash` invalidates old
  image caches, which recompute — the correct outcome, since the pixel representation genuinely
  changed. Old runs stay bright-on-dark (no flip); to adopt auto-detection, re-init or run
  `image set --auto`.

## Testing strategy

- **Synthetic (primary).** Build gels with known polarity, dtype, and channel layout:
  - dtype matrix (uint8, uint16, int16, float, RGB, RGBA, R≈G≈B "gray-RGB") → all canonical to
    float64 [0,1], signal-bright.
  - planted bright-on-dark vs. inverted dark-on-light → detector returns the right polarity +
    `ok`; near-symmetric heavily-loaded gel → `low` (no silent inversion) + a `warnings` entry.
  - NaN/Inf present → `NonFiniteImageError`.
  - colored (pseudo-colored) image → `mean` default; `--channel luminance`/`dominant`/`r`
    overrides are recorded and change the result.
  - multi-page TIFF → page 0 used + a `warnings` entry naming `source_pages`.
- **Hashing.** Flipping `invert` (or `channel`) changes `image_hash` and invalidates
  background/profile/fit; identical inputs → identical hash (determinism).
- **Real (verification).** e2e replay on `examples/data/example_gel_1.tif`
  (bright-on-dark fluorescence): detector → `bright_on_dark`, `ok`, no flip; results
  equivalent to today up to the [0,1] rescale. Spot-check an inverted copy of the same TIFF →
  detector flips it and downstream results match the non-inverted run.
- **Discipline** (per project convention, and per request — do not re-run the whole suite each
  change): unit-test the pure helpers (`to_float01`, `collapse_channels`, `detect_polarity`)
  in isolation as they stabilize; run only the touched layer (`pytest tests/core/test_image_io.py`)
  during the build; run `pytest tests/core tests/runfolder` once at integration; the real-TIFF
  replay is the final proof and runs last.

## Out of scope

- EXIF auto-rotation (rare in scientific TIFFs) — deferred.
- CMYK/indexed/1-bit beyond what the decoder already handles — error clearly, don't special-case.
- Deconvolving a pseudo-color LUT back to the original grayscale (a distinct problem).
- Per-region / spatially-varying polarity (a single global polarity per image).
- A multi-page page-picker UI beyond `--page K`.
