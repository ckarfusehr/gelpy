# Robust Image Input (Canonicalization) Implementation Plan

> **COMPLETED AND MERGED.** Do NOT execute this plan: it describes code that already
> exists, and its unticked checkboxes are an artifact of how it was written, not work
> outstanding. Kept as a record. Current behaviour: `docs/guide/image-input.md`.


> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every gel-image input (8/16-bit, float, RGB(A), inverted, multi-page) load into one canonical representation — 2-D float64 in [0,1], signal-bright, finite — with the polarity/channel decision recorded, confidence-flagged, overridable, and folded into the cache hash.

**Architecture:** A canonicalization layer in `core/image_io.py` decodes → normalizes dtype to [0,1] → collapses channels → auto-detects polarity (flip so signal is bright) and returns `(image, ImageMeta)`. Detection runs once at `init`; the recipe's frozen `ImageMeta` drives a deterministic replay (`apply_image_meta`) in the cached pipeline. `image_hash` folds in the interpretation fields so changing `--invert`/`--channel` recomputes the DAG. Downstream (`background`, `lanes`, `peaks`) is untouched — its "signal is bright, scale is [0,1]" assumption is now always true.

**Tech Stack:** Python 3.11 (`gelpy_dev` conda env), numpy, scipy, scikit-image (`img_as_float`, `rgb2gray`), pydantic v2, typer, pytest. Spec: `docs/decisions/2026-07-12-image-input-canonicalization-design.md`.

## Global Constraints

- Strict one-directional dependencies: `schemas/` → `core/` → (`render/` + `runfolder/`) → `cli/`. Never import upward.
- `schemas/` imports no numpy/scipy/matplotlib; every model subclasses `CanonicalModel` (`extra="forbid"`) — an unknown field is a hard `ValidationError`.
- matplotlib only in `render/` (and `cli/_render.py`). This feature touches neither renderer.
- The cached path must be **deterministic** (no RNG, no `Date.now`-style nondeterminism) — the content-hash cache requires it. Detection may use only pure functions of the pixels.
- All new domain errors subclass `GelpyError` with a stable `code` and are mapped in `core/errors.to_error_response`.
- Canonical image contract, guaranteed by the loader: `ndim == 2`, `dtype == float64`, all-finite, `min >= 0.0`, `max <= 1.0`, signal-bright.
- **Test discipline (per request — do NOT run the whole suite each change):** during a task run only that task's test (`pytest <file>::<test> -v`); run the touched layer (`pytest tests/core` or `pytest tests/runfolder`) only at the end of a task; run `pytest tests/core tests/runfolder tests/cli` once at integration (Task 10); the e2e replay runs last.
- Work on branch `feat/robust-image-input`. Commit per task.

---

### Task 1: Schema contracts — `ImageMeta`, recipe field, summary/status warnings

**Files:**
- Modify: `src/gelpy/schemas/image.py` (add `ImageMeta`)
- Modify: `src/gelpy/schemas/recipe.py` (add `Recipe.image`)
- Modify: `src/gelpy/schemas/run.py` (`RunUpdateSummary`, `RunStatus`)
- Test: `tests/schemas/test_image_meta.py` (create)

**Interfaces:**
- Produces: `ImageMeta(CanonicalModel)` with fields `original_dtype:str`, `original_shape:tuple[int,...]`, `source_channels:int`, `source_pages:int`, `page:int`, `channel_strategy:str`, `channel_index:int|None`, `polarity:Literal["bright_on_dark","dark_on_light"]`, `inverted:bool`, `polarity_source:Literal["auto","manual"]`, `polarity_confidence:Literal["ok","low"]`, `polarity_reason:str`. All have defaults (legacy-safe: `bright_on_dark`, not inverted, `channel_strategy="mean"`). `Recipe.image: ImageMeta`. `RunUpdateSummary.warnings:list[str]` + `.image:dict[str,Any]`. `RunStatus.polarity`, `.polarity_confidence`, `.warnings`.

- [ ] **Step 1: Write the failing test**

Create `tests/schemas/test_image_meta.py`:

```python
import pytest
from pydantic import ValidationError

from gelpy.schemas.image import ImageMeta
from gelpy.schemas.recipe import Recipe
from gelpy.schemas.run import RunStatus, RunUpdateSummary


def test_image_meta_defaults_are_legacy_safe():
    m = ImageMeta()
    assert m.polarity == "bright_on_dark"
    assert m.inverted is False
    assert m.channel_strategy == "mean"
    assert m.page == 0


def test_image_meta_forbids_unknown_field():
    with pytest.raises(ValidationError):
        ImageMeta(polarty="bright_on_dark")  # typo -> extra=forbid


def test_recipe_has_default_image_meta():
    # A recipe dict predating this field still validates (default_factory supplies it).
    data = {
        "source_image": {
            "path": "x.tif", "sha256": "0" * 64, "size": 1, "mtime": 0.0,
            "height": 2, "width": 2, "dtype": "float64",
        },
        "lanes": {"lanes": [], "profile_width_px": 10, "label_rotation": 45.0},
    }
    r = Recipe.model_validate(data)
    assert r.image.polarity == "bright_on_dark"


def test_summary_and_status_carry_warnings():
    s = RunUpdateSummary(warnings=["w"], image={"polarity": "dark_on_light"})
    assert s.warnings == ["w"] and s.image["polarity"] == "dark_on_light"
    st = RunStatus(
        run_dir="r", source_image="x.tif", image_width=2, image_height=2,
        polarity="dark_on_light", polarity_confidence="low", warnings=["w"],
    )
    assert st.polarity_confidence == "low"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/schemas/test_image_meta.py -v`
Expected: FAIL — `ImportError`/`AttributeError` (`ImageMeta` not defined; `Recipe.image`, `RunUpdateSummary.warnings` missing).

- [ ] **Step 3: Add `ImageMeta` to `src/gelpy/schemas/image.py`**

Append (keep existing `ImageRef`):

```python
from typing import Literal


class ImageMeta(CanonicalModel):
    """How the source image is interpreted to produce the canonical analysis array.

    Interpretation intent (channel + polarity + page) — editable and precious, a sibling
    to ``ImageRef`` (file identity). Detected once at init; frozen here to make the cached
    pipeline deterministic. These fields feed ``image_hash`` (``runfolder/stages.py``), so a
    change here recomputes exactly what depends on the pixels.
    """

    original_dtype: str = Field(default="", description="numpy dtype of the decoded source.")
    original_shape: tuple[int, ...] = Field(default=(), description="Decoded source shape.")
    source_channels: int = Field(default=1, ge=1, description="Source channel count (1=grayscale).")
    source_pages: int = Field(default=1, ge=1, description="Pages in the source (TIFF stacks).")
    page: int = Field(default=0, ge=0, description="Page index used from a multi-page source.")
    channel_strategy: str = Field(
        default="mean",
        description="How channels collapse to grayscale: gray|gray_rgb|mean|luminance|r|g|b|dominant|max.",
    )
    channel_index: int | None = Field(
        default=None, description="Chosen channel index for r/g/b/dominant strategies; else null."
    )
    polarity: Literal["bright_on_dark", "dark_on_light"] = Field(
        default="bright_on_dark", description="Signal polarity of the source image."
    )
    inverted: bool = Field(
        default=False, description="Whether 1-x was applied to make signal bright (canonical)."
    )
    polarity_source: Literal["auto", "manual"] = Field(
        default="auto", description="Whether polarity was auto-detected or set explicitly."
    )
    polarity_confidence: Literal["ok", "low"] = Field(
        default="ok", description="Detector confidence; 'low' raises an ignorable warning."
    )
    polarity_reason: str = Field(default="", description="Human reason when confidence is 'low'.")
```

- [ ] **Step 4: Wire `Recipe.image` in `src/gelpy/schemas/recipe.py`**

Add the import and field:

```python
from .image import ImageMeta, ImageRef  # replace the existing `from .image import ImageRef`
```

Inside `class Recipe`, after `source_image: ImageRef`:

```python
    image: ImageMeta = Field(
        default_factory=ImageMeta,
        description="Interpretation of the source image (channel/polarity/page). Feeds image_hash.",
    )
```

- [ ] **Step 5: Add warning/polarity fields in `src/gelpy/schemas/run.py`**

In `RunUpdateSummary`, after `files`:

```python
    warnings: list[str] = Field(
        default_factory=list, description="Non-fatal, ignorable messages for the agent/user."
    )
    image: dict[str, Any] = Field(
        default_factory=dict, description="Structured image interpretation decision (polarity/channel)."
    )
```

In `RunStatus`, after `background`:

```python
    polarity: Literal["bright_on_dark", "dark_on_light"] | None = Field(
        default=None, description="Recorded source-image polarity."
    )
    polarity_confidence: Literal["ok", "low"] | None = Field(
        default=None, description="Polarity-detection confidence."
    )
    warnings: list[str] = Field(
        default_factory=list, description="Unresolved non-fatal image warnings."
    )
```

(`Any` and `Literal` are already imported in `run.py`.)

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/schemas/test_image_meta.py -v`
Expected: PASS (4 tests).

- [ ] **Step 7: Commit**

```bash
git add src/gelpy/schemas/image.py src/gelpy/schemas/recipe.py src/gelpy/schemas/run.py tests/schemas/test_image_meta.py
git commit -m "feat(schemas): ImageMeta interpretation contract + summary/status warnings"
```

---

### Task 2: Pixel normalization — `to_float01` + channel collapse

**Files:**
- Modify: `src/gelpy/core/image_io.py`
- Test: `tests/core/test_image_io.py` (create)

**Interfaces:**
- Produces: `to_float01(arr) -> np.ndarray` (float64, global min-max preserving inter-channel ratios; unsigned int via `img_as_float`); `_collapse_with(arr, strategy, index) -> np.ndarray` (deterministic replay core); `collapse_channels(arr, strategy="auto", *, gray_atol=0.02) -> tuple[np.ndarray, str, int|None]` (detection: returns grayscale, resolved strategy, channel index).
- Consumes: `UnsupportedImageError` — added in Task 4; for this task raise `ValueError` in the unknown-strategy branch and swap to `UnsupportedImageError` in Task 4.

- [ ] **Step 1: Write the failing test**

Create `tests/core/test_image_io.py`:

```python
import numpy as np
import pytest

from gelpy.core.image_io import collapse_channels, to_float01


def test_to_float01_uint8_scales_by_255():
    arr = np.array([[0, 255], [128, 64]], dtype=np.uint8)
    out = to_float01(arr)
    assert out.dtype == np.float64
    assert out.min() == 0.0 and out.max() == 1.0


def test_to_float01_uint16_scales_by_65535():
    arr = np.array([[0, 65535]], dtype=np.uint16)
    assert np.allclose(to_float01(arr), [[0.0, 1.0]])


def test_to_float01_float_out_of_range_minmax_rescales():
    arr = np.array([[10.0, 20.0, 30.0]], dtype=np.float64)
    assert np.allclose(to_float01(arr), [[0.0, 0.5, 1.0]])


def test_to_float01_preserves_channel_ratios_global_scale():
    # A 2x1 RGB image; global (not per-channel) scaling keeps color ratios intact.
    arr = np.zeros((1, 1, 3), dtype=np.uint8)
    arr[0, 0] = [100, 200, 50]
    out = to_float01(arr)
    assert np.allclose(out[0, 0], [100 / 255, 200 / 255, 50 / 255])


def test_collapse_gray_rgb_is_lossless_mean():
    g = np.tile(np.linspace(0, 1, 10)[None, :, None], (4, 1, 3))  # R==G==B
    out, strategy, idx = collapse_channels(g)
    assert strategy == "gray_rgb" and idx is None
    assert np.allclose(out, g[:, :, 0])


def test_collapse_colored_defaults_to_mean():
    c = np.zeros((2, 2, 3))
    c[..., 0] = 0.9  # strong red, differs across channels
    out, strategy, idx = collapse_channels(c)
    assert strategy == "mean" and idx is None
    assert np.allclose(out, 0.3)


def test_collapse_channel_override_records_index():
    c = np.zeros((2, 2, 3))
    c[..., 2] = 0.8
    out, strategy, idx = collapse_channels(c, "b")
    assert strategy == "b" and idx == 2 and np.allclose(out, 0.8)


def test_collapse_2d_passthrough():
    g = np.linspace(0, 1, 12).reshape(3, 4)
    out, strategy, idx = collapse_channels(g)
    assert strategy == "gray" and idx is None and np.allclose(out, g)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_image_io.py -v`
Expected: FAIL — `ImportError: cannot import name 'collapse_channels'`.

- [ ] **Step 3: Implement in `src/gelpy/core/image_io.py`**

Replace the file's body above `load_image` (keep the module docstring) with these helpers; `load_image` is rebuilt in Task 4:

```python
from __future__ import annotations

import numpy as np
from skimage import color, io
from skimage.util import img_as_float

_CHANNEL_INDEX = {"r": 0, "g": 1, "b": 2}


def to_float01(arr: np.ndarray) -> np.ndarray:
    """Canonical float64 in [0,1]. Integer inputs scale by their dtype max; float inputs
    are min-max rescaled only if they fall outside [0,1]. Scaling is GLOBAL (one factor for
    the whole array), so an RGB image keeps its inter-channel ratios."""
    a = np.asarray(arr)
    if a.dtype == bool:
        return a.astype(np.float64)
    if np.issubdtype(a.dtype, np.floating):
        f = a.astype(np.float64, copy=False)
    else:
        f = np.asarray(img_as_float(a), dtype=np.float64)  # unsigned->[0,1], signed->[-1,1]
    lo, hi = float(f.min()), float(f.max())
    if lo < 0.0 or hi > 1.0:
        return (f - lo) / (hi - lo) if hi > lo else np.zeros_like(f)
    return f


def _collapse_with(arr: np.ndarray, strategy: str, index: int | None) -> np.ndarray:
    """Deterministic channel->grayscale collapse for a resolved (strategy, index)."""
    if arr.ndim == 2:
        return arr
    rgb = arr[:, :, :3] if arr.shape[2] == 4 else arr
    if strategy in ("gray_rgb", "mean"):
        return rgb.mean(axis=2)
    if strategy == "luminance":
        return color.rgb2gray(rgb)
    if strategy == "max":
        return rgb.max(axis=2)
    if index is not None:
        return rgb[:, :, index]
    raise ValueError(f"unknown channel strategy {strategy!r}")  # -> UnsupportedImageError in Task 4


def collapse_channels(
    arr: np.ndarray, strategy: str = "auto", *, gray_atol: float = 0.02
) -> tuple[np.ndarray, str, int | None]:
    """Collapse to 2-D grayscale, returning (gray, resolved_strategy, channel_index).

    A 2-D input is passthrough ('gray'). An RGB whose channels are equal within tolerance is
    'gray_rgb' (lossless mean). Otherwise 'auto' resolves to 'mean' (the least-biased default);
    'r'/'g'/'b'/'dominant' pick a single channel (index recorded); 'luminance'/'max' collapse.
    """
    if arr.ndim == 2:
        return arr, "gray", None
    rgb = arr[:, :, :3] if arr.shape[2] == 4 else arr
    spread = rgb.max(axis=2) - rgb.min(axis=2)
    if float(np.percentile(spread, 99.9)) < gray_atol:
        return rgb.mean(axis=2), "gray_rgb", None
    resolved = "mean" if strategy == "auto" else strategy
    index: int | None = None
    if resolved == "dominant":
        spreads = [np.percentile(rgb[:, :, i], 99.5) - np.percentile(rgb[:, :, i], 0.5)
                   for i in range(rgb.shape[2])]
        index = int(np.argmax(spreads))
    elif resolved in _CHANNEL_INDEX:
        index = _CHANNEL_INDEX[resolved]
    return _collapse_with(rgb, resolved, index), resolved, index
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/core/test_image_io.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add src/gelpy/core/image_io.py tests/core/test_image_io.py
git commit -m "feat(core): dtype-aware float [0,1] conversion and channel collapse"
```

---

### Task 3: Polarity detector

**Files:**
- Modify: `src/gelpy/core/image_io.py`
- Test: `tests/core/test_image_io.py` (extend)

**Interfaces:**
- Produces: `SKEW_FLOOR=0.5`, `TAIL_ASYM_FLOOR=0.1`; `PolarityDecision` dataclass (`polarity:str`, `inverted:bool`, `confidence:str`, `reason:str`, `skew:float`, `tail_asym:float`); `detect_polarity(gray01) -> PolarityDecision`.

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_image_io.py`:

```python
from gelpy.core.image_io import detect_polarity


def _bright_on_dark(width=400, height=200, sd=6.0):
    img = np.full((height, width), 0.05)
    x = np.arange(width)
    for c in (80, 200, 320):
        img += np.exp(-((x - c) ** 2) / (2 * sd**2))[None, :]
    return np.clip(img, 0, 1)


def test_detect_bright_on_dark():
    d = detect_polarity(_bright_on_dark())
    assert d.polarity == "bright_on_dark" and d.inverted is False and d.confidence == "ok"


def test_detect_dark_on_light_is_flagged_for_inversion():
    d = detect_polarity(1.0 - _bright_on_dark())
    assert d.polarity == "dark_on_light" and d.inverted is True and d.confidence == "ok"


def test_detect_near_symmetric_is_low_confidence_and_not_inverted():
    rng = np.random.default_rng(0)
    noise = np.clip(0.5 + 0.05 * rng.standard_normal((200, 400)), 0, 1)
    d = detect_polarity(noise)
    assert d.confidence == "low" and d.inverted is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_image_io.py -k polarity -v`
Expected: FAIL — `cannot import name 'detect_polarity'`.

- [ ] **Step 3: Implement in `src/gelpy/core/image_io.py`**

Add near the top (after imports) and a function:

```python
from dataclasses import dataclass

SKEW_FLOOR = 0.5
TAIL_ASYM_FLOOR = 0.1


@dataclass
class PolarityDecision:
    polarity: str          # "bright_on_dark" | "dark_on_light"
    inverted: bool
    confidence: str        # "ok" | "low"
    reason: str
    skew: float
    tail_asym: float


def detect_polarity(gray01: np.ndarray) -> PolarityDecision:
    """Detect signal polarity from two independent, deterministic cues; agreement = confidence.

    Background occupies most of the image area, so the LONG tail of the histogram is signal.
    Cue 1: sign of Fisher skewness. Cue 2: which tail reaches further from the median. Both
    positive -> bright-on-dark. They must agree AND be non-trivial to earn 'ok'; otherwise we
    do NOT invert (the reversible default) and flag 'low'. Inversion is applied only on a
    confident dark-on-light.
    """
    x = np.asarray(gray01, dtype=np.float64).ravel()
    mu, sd = float(x.mean()), float(x.std())
    g1 = float(((x - mu) ** 3).mean() / (sd**3)) if sd > 0 else 0.0
    p_lo, m, p_hi = (float(v) for v in np.percentile(x, [0.5, 50, 99.5]))
    bright_reach, dark_reach = p_hi - m, m - p_lo
    denom = bright_reach + dark_reach
    tail_asym = (bright_reach - dark_reach) / denom if denom > 0 else 0.0
    strong = abs(g1) >= SKEW_FLOOR or abs(tail_asym) >= TAIL_ASYM_FLOOR
    agree = (g1 > 0) == (tail_asym > 0)
    if agree and strong:
        if g1 > 0:
            return PolarityDecision("bright_on_dark", False, "ok", "", g1, tail_asym)
        return PolarityDecision("dark_on_light", True, "ok", "", g1, tail_asym)
    reason = (
        "polarity cues (histogram skew and tail asymmetry) disagree or are too weak "
        "(near-symmetric intensity); assuming bright-on-dark. Set --invert / --no-invert if wrong."
    )
    return PolarityDecision("bright_on_dark", False, "low", reason, g1, tail_asym)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/core/test_image_io.py -k polarity -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/gelpy/core/image_io.py tests/core/test_image_io.py
git commit -m "feat(core): deterministic polarity detector (skew + tail-asymmetry)"
```

---

### Task 4: Public loader — decode, finite guard, `canonicalize_image`, `apply_image_meta`, errors

**Files:**
- Modify: `src/gelpy/core/image_io.py`
- Modify: `src/gelpy/core/errors.py`
- Test: `tests/core/test_image_io.py` (extend)
- Test: `tests/core/test_errors.py` (extend)

**Interfaces:**
- Produces: `decode_image(path, *, page=0) -> tuple[np.ndarray, int]` (array, source_pages); `canonicalize_image(path, *, channel="auto", invert="auto", page=0) -> tuple[np.ndarray, ImageMeta]`; `apply_image_meta(path, meta) -> np.ndarray`; `load_image(path) -> np.ndarray` (thin wrapper = `canonicalize_image(path)[0]`). `NonFiniteImageError`, `UnsupportedImageError` (both `GelpyError`).
- Consumes: `ImageMeta` (Task 1); `to_float01`/`collapse_channels`/`_collapse_with`/`detect_polarity` (Tasks 2-3).

- [ ] **Step 1: Write the failing tests**

Add errors first — append to `tests/core/test_errors.py`:

```python
from gelpy.core.errors import NonFiniteImageError, UnsupportedImageError, to_error_response


def test_nonfinite_and_unsupported_map_to_codes():
    assert to_error_response(NonFiniteImageError("bad")).code == "nonfinite_image"
    assert to_error_response(UnsupportedImageError("bad")).code == "unsupported_image"
```

Append to `tests/core/test_image_io.py`:

```python
from skimage import io as _skio

from gelpy.core.errors import NonFiniteImageError
from gelpy.core.image_io import apply_image_meta, canonicalize_image, load_image


def _save(tmp_path, name, arr):
    p = tmp_path / name
    _skio.imsave(str(p), arr, check_contrast=False)
    return str(p)


def test_canonicalize_uint16_bright_on_dark(tmp_path):
    gel = (_bright_on_dark() * 65535).astype(np.uint16)
    img, meta = canonicalize_image(_save(tmp_path, "g.tif", gel))
    assert img.dtype == np.float64 and img.min() >= 0.0 and img.max() <= 1.0
    assert meta.polarity == "bright_on_dark" and meta.inverted is False
    assert meta.original_dtype == "uint16" and meta.channel_strategy == "gray"


def test_canonicalize_inverts_dark_on_light(tmp_path):
    gel = ((1.0 - _bright_on_dark()) * 255).astype(np.uint8)
    img, meta = canonicalize_image(_save(tmp_path, "d.png", gel))
    assert meta.inverted is True and meta.polarity == "dark_on_light"
    # after canonicalization the band columns are the bright ones again
    assert img[:, 80].mean() > img[:, 5].mean()


def test_manual_invert_override_sets_source_manual(tmp_path):
    gel = (_bright_on_dark() * 255).astype(np.uint8)
    _, meta = canonicalize_image(_save(tmp_path, "g.png", gel), invert=True)
    assert meta.inverted is True and meta.polarity_source == "manual"


def test_apply_image_meta_reproduces_canonical(tmp_path):
    gel = ((1.0 - _bright_on_dark()) * 255).astype(np.uint8)
    p = _save(tmp_path, "d.png", gel)
    img, meta = canonicalize_image(p)
    assert np.allclose(apply_image_meta(p, meta), img)


def test_nonfinite_raises(tmp_path):
    bad = _bright_on_dark().astype(np.float32)
    bad[0, 0] = np.nan
    with pytest.raises(NonFiniteImageError):
        canonicalize_image(_save(tmp_path, "nan.tif", bad))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/core/test_image_io.py -k "canonicalize or apply or nonfinite" tests/core/test_errors.py -v`
Expected: FAIL — `cannot import name 'canonicalize_image'` / `NonFiniteImageError`.

- [ ] **Step 3: Add errors in `src/gelpy/core/errors.py`**

After `SourceImageChangedError`:

```python
class NonFiniteImageError(GelpyError):
    """The decoded image contains NaN/Inf pixels (would poison percentiles and fits)."""

    code = "nonfinite_image"


class UnsupportedImageError(GelpyError):
    """The image could not be interpreted (unexpected ndim/shape or unknown channel strategy)."""

    code = "unsupported_image"
```

(No change needed in `to_error_response` — both subclass `GelpyError`, whose branch already emits `exc.code`.)

- [ ] **Step 4: Implement the loader in `src/gelpy/core/image_io.py`**

Add imports at the top: `from .errors import NonFiniteImageError, UnsupportedImageError` and `from ..schemas.image import ImageMeta`. Then append:

```python
def decode_image(path: str, *, page: int = 0) -> tuple[np.ndarray, int]:
    """Decode to a single 2-D/3-D(HxWxC) frame, returning (array, source_pages).

    A multi-page/stacked source (ndim 3 whose last axis is not a channel count, or ndim 4)
    is treated as pages along axis 0; ``page`` selects one (clamped). The (P,H,W) vs (H,W,3)
    ambiguity is resolved in favor of channels when the last axis is in {2,3,4}."""
    a = np.asarray(io.imread(path))
    if a.ndim <= 2:
        return a, 1
    if a.ndim == 3 and a.shape[-1] in (2, 3, 4):
        return a, 1
    if a.ndim in (3, 4):
        n = int(a.shape[0])
        return a[min(page, n - 1)], n
    raise UnsupportedImageError(f"unsupported image ndim={a.ndim}, shape={a.shape}")


def _check_finite(arr: np.ndarray) -> None:
    a = np.asarray(arr)
    if not np.isfinite(a).all():
        n = int((~np.isfinite(a)).sum())
        raise NonFiniteImageError(
            f"image contains {n} non-finite (NaN/Inf) pixels; clean the source before analysis."
        )


def canonicalize_image(
    path: str, *, channel: str = "auto", invert: bool | str = "auto", page: int = 0
) -> tuple[np.ndarray, ImageMeta]:
    """Decode + canonicalize to a signal-bright float64 [0,1] 2-D array, plus the frozen
    ``ImageMeta`` describing the interpretation. Detection runs here (init path)."""
    raw, source_pages = decode_image(path, page=page)
    _check_finite(raw)
    original_dtype = str(np.asarray(raw).dtype)
    original_shape = tuple(int(s) for s in np.asarray(raw).shape)
    source_channels = 1 if raw.ndim == 2 else int(raw.shape[-1])
    gray, strategy, index = collapse_channels(to_float01(raw), channel)

    if invert is True:
        polarity, inverted, source, confidence, reason = "dark_on_light", True, "manual", "ok", ""
    elif invert is False:
        polarity, inverted, source, confidence, reason = "bright_on_dark", False, "manual", "ok", ""
    else:
        d = detect_polarity(gray)
        polarity, inverted, source = d.polarity, d.inverted, "auto"
        confidence, reason = d.confidence, d.reason

    canonical = 1.0 - gray if inverted else gray
    meta = ImageMeta(
        original_dtype=original_dtype, original_shape=original_shape,
        source_channels=source_channels, source_pages=source_pages, page=page,
        channel_strategy=strategy, channel_index=index, polarity=polarity, inverted=inverted,
        polarity_source=source, polarity_confidence=confidence, polarity_reason=reason,
    )
    return np.ascontiguousarray(canonical, dtype=np.float64), meta


def apply_image_meta(path: str, meta: ImageMeta) -> np.ndarray:
    """Deterministic replay of a frozen ``ImageMeta`` — NO detection. Used by the cached
    pipeline so the same recipe always yields the same canonical pixels."""
    raw, _ = decode_image(path, page=meta.page)
    _check_finite(raw)
    gray = _collapse_with(to_float01(raw), meta.channel_strategy, meta.channel_index)
    canonical = 1.0 - gray if meta.inverted else gray
    return np.ascontiguousarray(canonical, dtype=np.float64)


def load_image(path: str) -> np.ndarray:
    """Backward-compatible convenience: auto-canonicalized array only (no metadata)."""
    return canonicalize_image(path)[0]
```

Then update `_collapse_with`'s final branch to raise the domain error instead of `ValueError`:

```python
    raise UnsupportedImageError(f"unknown channel strategy {strategy!r}")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/core/test_image_io.py tests/core/test_errors.py -v`
Expected: PASS (all image_io + errors tests).

- [ ] **Step 6: Commit**

```bash
git add src/gelpy/core/image_io.py src/gelpy/core/errors.py tests/core/test_image_io.py tests/core/test_errors.py
git commit -m "feat(core): canonicalize_image/apply_image_meta loader + non-finite guard"
```

---

### Task 5: Fold interpretation into `image_hash`

**Files:**
- Modify: `src/gelpy/runfolder/stages.py:31-33`
- Test: `tests/runfolder/test_hashing.py` (extend)

**Interfaces:**
- Consumes: `Recipe.image` (Task 1). Modifies `image_hash(recipe)` so that `channel_strategy`, `channel_index`, `inverted`, `page` participate; every downstream hash already chains off it.

- [ ] **Step 1: Write the failing test**

Append to `tests/runfolder/test_hashing.py`:

```python
from gelpy.runfolder.stages import background_hash, image_hash
from gelpy.schemas.image import ImageMeta, ImageRef
from gelpy.schemas.lanes import LaneLayout
from gelpy.schemas.recipe import Recipe


def _recipe(**meta_kw):
    ref = ImageRef(path="x.tif", sha256="a" * 64, size=1, mtime=0.0, height=2, width=2, dtype="f8")
    return Recipe(source_image=ref, lanes=LaneLayout(lanes=[], profile_width_px=10), image=ImageMeta(**meta_kw))


def test_image_hash_changes_when_inverted_flips():
    assert image_hash(_recipe(inverted=False)) != image_hash(_recipe(inverted=True))


def test_image_hash_stable_for_equal_meta():
    assert image_hash(_recipe(channel_strategy="mean")) == image_hash(_recipe(channel_strategy="mean"))


def test_invert_flip_propagates_to_background_hash():
    assert background_hash(_recipe(inverted=False)) != background_hash(_recipe(inverted=True))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/runfolder/test_hashing.py -k image -v`
Expected: FAIL — `image_hash` ignores `inverted` (both sides equal).

- [ ] **Step 3: Rewrite `image_hash` in `src/gelpy/runfolder/stages.py`**

Replace:

```python
def image_hash(recipe: Recipe) -> str:
    """Identity of the decoded image = the source file's content hash."""
    return recipe.source_image.sha256
```

with:

```python
def image_hash(recipe: Recipe) -> str:
    """Identity of the CANONICAL image = source bytes + how they are interpreted.

    Folds the ``ImageMeta`` interpretation fields (channel + invert + page) into the hash so
    that changing ``--invert``/``--channel``/``--page`` invalidates the cached array (and, by
    chaining, every downstream stage). Mirrors ``background_hash`` hashing validated params.
    """
    m = recipe.image
    return hash_fields(
        source=recipe.source_image.sha256,
        channel_strategy=m.channel_strategy,
        channel_index=m.channel_index,
        inverted=m.inverted,
        page=m.page,
    )
```

(`hash_fields` is already imported in `stages.py`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/runfolder/test_hashing.py -v`
Expected: PASS (existing + 3 new).

- [ ] **Step 5: Commit**

```bash
git add src/gelpy/runfolder/stages.py tests/runfolder/test_hashing.py
git commit -m "fix(runfolder): fold image interpretation into image_hash"
```

---

### Task 6: Pipeline replays the frozen `ImageMeta`

**Files:**
- Modify: `src/gelpy/runfolder/pipeline.py:95` (inside `ensure_image`)
- Test: `tests/runfolder/test_incremental_recompute.py` (extend) OR new `tests/runfolder/test_image_pipeline.py`

**Interfaces:**
- Consumes: `apply_image_meta` (Task 4), `Recipe.image` (Task 1). `ensure_image` now returns a canonical float64 [0,1] array and its cache is keyed by the Task-5 hash.

- [ ] **Step 1: Write the failing test**

Create `tests/runfolder/test_image_pipeline.py`:

```python
import numpy as np
from skimage import io

from gelpy.runfolder.layout import RunPaths
from gelpy.runfolder.manifest import RunManifest
from gelpy.runfolder.pipeline import Pipeline
from gelpy.runfolder.recipe_io import build_image_ref
from gelpy.schemas.image import ImageMeta
from gelpy.schemas.lanes import LaneLayout
from gelpy.schemas.recipe import Recipe


def _dark_on_light_png(tmp_path):
    x = np.arange(400)
    band = np.zeros(400)
    for c in (80, 200, 320):
        band += np.exp(-((x - c) ** 2) / (2 * 6.0**2))
    img = np.clip(0.9 - np.tile(band, (200, 1)), 0, 1)  # dark bands on light bg
    p = tmp_path / "gel.png"
    io.imsave(str(p), (img * 255).astype("uint8"), check_contrast=False)
    return str(p)


def test_ensure_image_returns_canonical_signal_bright(tmp_path):
    path = _dark_on_light_png(tmp_path)
    ref = build_image_ref(path)
    recipe = Recipe(
        source_image=ref, lanes=LaneLayout(lanes=[], profile_width_px=10),
        image=ImageMeta(inverted=True, polarity="dark_on_light", channel_strategy="gray"),
    )
    paths = RunPaths(str(tmp_path / "run")); paths.ensure_dirs()
    img = Pipeline(paths, recipe, RunManifest(recipe_hash="x")).ensure_image()
    assert img.dtype == np.float64 and 0.0 <= img.min() and img.max() <= 1.0
    assert img[:, 80].mean() > img[:, 5].mean()  # bands are bright after inversion
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/runfolder/test_image_pipeline.py -v`
Expected: FAIL — `ensure_image` still calls `load_image` (no inversion) → band columns not brighter.

- [ ] **Step 3: Update `ensure_image` in `src/gelpy/runfolder/pipeline.py`**

Change the import at the top:

```python
from ..core.image_io import apply_image_meta
```

(remove the `load_image` import if unused elsewhere in the file). Then in `ensure_image`, replace the compute thunk:

```python
        arr = get_or_compute_npy(path, lambda: load_image(self.recipe.source_image.path))
```

with:

```python
        arr = get_or_compute_npy(
            path, lambda: apply_image_meta(self.recipe.source_image.path, self.recipe.image)
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/runfolder/test_image_pipeline.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/gelpy/runfolder/pipeline.py tests/runfolder/test_image_pipeline.py
git commit -m "feat(runfolder): pipeline replays frozen ImageMeta (canonical, signal-bright)"
```

---

### Task 7: init builds `ImageMeta` + surfaces warnings

**Files:**
- Modify: `src/gelpy/runfolder/recipe_io.py` (add `build_image_ref_and_meta`)
- Modify: `src/gelpy/runfolder/commands.py` (`init_run`, add `image_warnings` helper)
- Modify: `src/gelpy/schemas/requests.py` (`InitRunRequest` fields)
- Test: `tests/runfolder/test_coherence.py` (extend) OR new `tests/runfolder/test_image_commands.py`

**Interfaces:**
- Produces: `build_image_ref_and_meta(path, *, channel="auto", invert="auto", page=0) -> tuple[ImageRef, ImageMeta]`; `image_warnings(meta) -> list[str]`. `InitRunRequest.channel:str="auto"`, `.invert:bool|None=None`, `.page:int=0`.
- Consumes: `canonicalize_image` (Task 4). `init_run` writes `recipe.image` and populates `RunUpdateSummary.warnings` + `.image`.

- [ ] **Step 1: Write the failing test**

Create `tests/runfolder/test_image_commands.py`:

```python
import numpy as np
from skimage import io

from gelpy.runfolder.commands import init_run
from gelpy.runfolder.recipe_io import load_recipe
from gelpy.runfolder.layout import RunPaths
from gelpy.schemas.requests import InitRunRequest


def _png(tmp_path, arr):
    p = tmp_path / "gel.png"
    io.imsave(str(p), arr, check_contrast=False)
    return str(p)


def _bands(width=400):
    x = np.arange(width); b = np.zeros(width)
    for c in (80, 200, 320):
        b += np.exp(-((x - c) ** 2) / (2 * 6.0**2))
    return np.tile(b, (200, 1))


def test_init_detects_and_records_polarity(tmp_path):
    dark = np.clip(0.9 - _bands(), 0, 1)
    run = str(tmp_path / "run")
    init_run(InitRunRequest(run_dir=run, image_path=_png(tmp_path, (dark * 255).astype("uint8")), n_lanes=3))
    recipe = load_recipe(RunPaths(run))
    assert recipe.image.polarity == "dark_on_light" and recipe.image.inverted is True


def test_init_manual_invert_and_summary_image_block(tmp_path):
    bright = np.clip(_bands(), 0, 1)
    run = str(tmp_path / "run2")
    summary = init_run(InitRunRequest(
        run_dir=run, image_path=_png(tmp_path, (bright * 255).astype("uint8")), n_lanes=3, invert=True))
    assert summary.image["inverted"] is True
    assert load_recipe(RunPaths(run)).image.polarity_source == "manual"


def test_init_low_confidence_emits_warning(tmp_path):
    rng = np.random.default_rng(0)
    flat = np.clip(0.5 + 0.05 * rng.standard_normal((200, 400)), 0, 1)
    run = str(tmp_path / "run3")
    summary = init_run(InitRunRequest(run_dir=run, image_path=_png(tmp_path, (flat * 255).astype("uint8")), n_lanes=3))
    assert any("low-confidence" in w or "polarity" in w for w in summary.warnings)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/runfolder/test_image_commands.py -v`
Expected: FAIL — `InitRunRequest` has no `invert`; `recipe.image` still default.

- [ ] **Step 3: Add request fields in `src/gelpy/schemas/requests.py`**

In `InitRunRequest`, after `label_rotation`:

```python
    channel: str = Field(default="auto", description="Channel->grayscale strategy: auto|mean|luminance|r|g|b|dominant|max.")
    invert: bool | None = Field(default=None, description="Force polarity: True=dark-on-light, False=bright-on-dark, None=auto-detect.")
    page: int = Field(default=0, ge=0, description="Page index for a multi-page source.")
```

- [ ] **Step 4: Add `build_image_ref_and_meta` in `src/gelpy/runfolder/recipe_io.py`**

Add the import `from ..core.image_io import canonicalize_image` and `from ..schemas.image import ImageMeta, ImageRef` (extend the existing ImageRef import), then:

```python
def build_image_ref_and_meta(
    path: str | Path, *, channel: str = "auto", invert: bool | str = "auto", page: int = 0
) -> tuple[ImageRef, ImageMeta]:
    """Build the file-identity ref AND the interpretation meta in one decode."""
    path = Path(path)
    stat = path.stat()
    image, meta = canonicalize_image(str(path), channel=channel, invert=invert, page=page)
    ref = ImageRef(
        path=str(path), sha256=file_hash(path), size=stat.st_size, mtime=stat.st_mtime,
        height=int(image.shape[0]), width=int(image.shape[1]), dtype=str(image.dtype),
    )
    return ref, meta
```

(Leave `build_image_ref` as-is; it stays valid via `load_image`.)

- [ ] **Step 5: Update `init_run` and add `image_warnings` in `src/gelpy/runfolder/commands.py`**

Add near the top-level helpers:

```python
def image_warnings(meta) -> list[str]:
    """Non-fatal, ignorable image messages for the agent/user."""
    warnings: list[str] = []
    if meta.polarity_confidence == "low":
        warnings.append(
            f"image polarity is low-confidence ({meta.polarity_reason}); treating as "
            f"{meta.polarity}. Override with `image set --invert` / `--no-invert` if wrong."
        )
    if meta.source_pages > 1:
        warnings.append(
            f"source image has {meta.source_pages} pages; using page {meta.page}. "
            f"Select another with --page (init or `image set`)."
        )
    return warnings
```

In `init_run`, replace `ref = build_image_ref(request.image_path)` with:

```python
    ref, image_meta = build_image_ref_and_meta(
        request.image_path,
        channel=request.channel,
        invert="auto" if request.invert is None else request.invert,
        page=request.page,
    )
```

Change the recipe construction to include the meta:

```python
    recipe = Recipe(source_image=ref, lanes=layout, image=image_meta)
```

Change the return to carry warnings + the decision block:

```python
    return RunUpdateSummary(
        files={"recipe": _rel(paths, paths.recipe)},
        warnings=image_warnings(image_meta),
        image=image_meta.model_dump(mode="json"),
    )
```

Update the imports in `commands.py`: replace `from .recipe_io import build_image_ref, ...` to include `build_image_ref_and_meta`.

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/runfolder/test_image_commands.py -v`
Expected: PASS (3 tests).

- [ ] **Step 7: Commit**

```bash
git add src/gelpy/runfolder/recipe_io.py src/gelpy/runfolder/commands.py src/gelpy/schemas/requests.py tests/runfolder/test_image_commands.py
git commit -m "feat(runfolder): init auto-detects polarity, records ImageMeta, surfaces warnings"
```

---

### Task 8: `set_image` command + status surfacing

**Files:**
- Modify: `src/gelpy/schemas/requests.py` (add `SetImageRequest`)
- Modify: `src/gelpy/runfolder/commands.py` (add `set_image`, extend `status`)
- Test: `tests/runfolder/test_image_commands.py` (extend)

**Interfaces:**
- Produces: `SetImageRequest(run_dir, channel=None, invert=None, page=None, auto=False)`; `set_image(request) -> RunUpdateSummary`. `status` now sets `RunStatus.polarity`, `.polarity_confidence`, `.warnings`.
- Consumes: `build_image_ref_and_meta`, `image_warnings`, `load_recipe`/`save_recipe`.

- [ ] **Step 1: Write the failing test**

Append to `tests/runfolder/test_image_commands.py`:

```python
from gelpy.runfolder.commands import set_image, status
from gelpy.schemas.requests import SetImageRequest, StatusRequest


def test_set_image_forces_inversion_and_status_reflects_it(tmp_path):
    bright = np.clip(_bands(), 0, 1)
    run = str(tmp_path / "run4")
    init_run(InitRunRequest(run_dir=run, image_path=_png(tmp_path, (bright * 255).astype("uint8")), n_lanes=3))
    set_image(SetImageRequest(run_dir=run, invert=True))
    recipe = load_recipe(RunPaths(run))
    assert recipe.image.inverted is True and recipe.image.polarity_source == "manual"
    st = status(StatusRequest(run_dir=run))
    assert st.polarity == "dark_on_light"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/runfolder/test_image_commands.py -k set_image -v`
Expected: FAIL — `cannot import name 'set_image'` / `SetImageRequest`.

- [ ] **Step 3: Add `SetImageRequest` in `src/gelpy/schemas/requests.py`**

```python
class SetImageRequest(CanonicalModel):
    """Re-interpret the source image of an existing run (polarity/channel/page)."""

    run_dir: str = Field(description="Existing run folder.")
    channel: str | None = Field(default=None, description="New channel strategy; None keeps current.")
    invert: bool | None = Field(default=None, description="Force polarity; None keeps current unless --auto.")
    page: int | None = Field(default=None, ge=0, description="New page index; None keeps current.")
    auto: bool = Field(default=False, description="Re-run auto-detection for polarity.")
```

- [ ] **Step 4: Add `set_image` and extend `status` in `src/gelpy/runfolder/commands.py`**

```python
def set_image(request: SetImageRequest) -> RunUpdateSummary:
    """Re-interpret the source image; changing channel/invert/page invalidates caches by hash."""
    paths, recipe, manifest = _load(request.run_dir)
    current = recipe.image
    channel = request.channel or current.channel_strategy
    page = current.page if request.page is None else request.page
    if request.auto:
        invert: bool | str = "auto"
    elif request.invert is not None:
        invert = request.invert
    else:
        invert = current.inverted  # keep current polarity decision
    _, image_meta = build_image_ref_and_meta(
        recipe.source_image.path, channel=channel, invert=invert, page=page
    )
    recipe.image = image_meta
    _persist(paths, recipe, manifest)
    _append_history(paths, "set_image", {"channel": channel, "invert": invert, "page": page})
    return RunUpdateSummary(
        files={"recipe": _rel(paths, paths.recipe)},
        warnings=image_warnings(image_meta),
        image=image_meta.model_dump(mode="json"),
    )
```

In `status`, where `RunStatus(...)` is constructed, add the three fields from `recipe.image`:

```python
        polarity=recipe.image.polarity,
        polarity_confidence=recipe.image.polarity_confidence,
        warnings=image_warnings(recipe.image),
```

Add `SetImageRequest` to the `from ..schemas.requests import (...)` block.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/runfolder/test_image_commands.py -v`
Expected: PASS (all).

- [ ] **Step 6: Commit**

```bash
git add src/gelpy/runfolder/commands.py src/gelpy/schemas/requests.py tests/runfolder/test_image_commands.py
git commit -m "feat(runfolder): image set command + polarity/warnings in status"
```

---

### Task 9: CLI wiring — `init` flags + `image set`

**Files:**
- Modify: `src/gelpy/cli/app.py` (`init` options; `image set` command)
- Test: `tests/cli/test_cli.py` (extend)

**Interfaces:**
- Consumes: `commands.init_run`/`commands.set_image`, `InitRunRequest`/`SetImageRequest`. `init` gains `--channel`, `--invert/--no-invert` (omit = auto), `--page`. New `image set` command on the existing `image` sub-app.

- [ ] **Step 1: Write the failing test**

Append to `tests/cli/test_cli.py`:

```python
def test_init_invert_flag_and_image_set(gel_image, run_dir):
    _run("init", run_dir, gel_image, "--n-lanes", "4", "--no-invert")
    out = _run("status", run_dir)
    assert json.loads(out.output)["polarity"] == "bright_on_dark"
    _run("image", "set", run_dir, "--invert")
    out = _run("status", run_dir)
    assert json.loads(out.output)["polarity"] == "dark_on_light"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/cli/test_cli.py::test_init_invert_flag_and_image_set -v`
Expected: FAIL — `--no-invert`/`image set` unknown option/command.

- [ ] **Step 3: Extend `init` in `src/gelpy/cli/app.py`**

Add options to the `init` signature (after `labels_file`):

```python
    channel: str = typer.Option("auto", help="Channel->grayscale: auto|mean|luminance|r|g|b|dominant|max."),
    invert: Optional[bool] = typer.Option(None, "--invert/--no-invert", help="Force polarity; omit to auto-detect."),
    page: int = typer.Option(0, help="Page index for a multi-page source."),
```

and pass them into `InitRunRequest(...)`:

```python
            channel=channel,
            invert=invert,
            page=page,
```

- [ ] **Step 4: Add `image set` in `src/gelpy/cli/app.py`**

Find the `image` sub-app (the one carrying `image show`; e.g. `image_app`) and add:

```python
@image_app.command("set")
@_io.handle_errors
def image_set(
    run_dir: str = typer.Argument(...),
    channel: Optional[str] = typer.Option(None, help="New channel strategy."),
    invert: Optional[bool] = typer.Option(None, "--invert/--no-invert", help="Force polarity."),
    page: Optional[int] = typer.Option(None, help="New page index."),
    auto: bool = typer.Option(False, "--auto", help="Re-run polarity auto-detection."),
    fmt: str = typer.Option("json", "--format"),
):
    """Re-interpret the source image (polarity/channel/page) of an existing run."""
    _io.emit(
        commands.set_image(
            SetImageRequest(run_dir=run_dir, channel=channel, invert=invert, page=page, auto=auto)
        ),
        fmt,
    )
```

Add `SetImageRequest` to the request imports at the top of `app.py`.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/cli/test_cli.py::test_init_invert_flag_and_image_set -v`
Expected: PASS.

- [ ] **Step 6: Update the README image-input section**

In `README.md`, under "How to use gelpy", add after the `init` line:

```markdown
gelpy handles 8/16-bit, float, RGB(A), and multi-page images uniformly (canonical float
[0,1], signal-bright). Polarity (bright-band-on-dark vs dark-band-on-light) is auto-detected
and recorded; override with `gelpy init ... --invert/--no-invert` or fix later with
`gelpy image set <run> --invert`. Pick a channel for colored images with `--channel`.
```

- [ ] **Step 7: Commit**

```bash
git add src/gelpy/cli/app.py tests/cli/test_cli.py README.md
git commit -m "feat(cli): init polarity/channel/page flags + image set command"
```

---

### Task 10: Integration, docs, and real-TIFF verification

**Files:**
- Modify: `CLAUDE.md` (architecture note)
- Test: run existing suites; e2e replay

**Interfaces:** none new — this task verifies the whole feature end-to-end.

- [ ] **Step 1: Run the touched layers together**

Run: `pytest tests/schemas tests/core tests/runfolder tests/cli -v`
Expected: PASS. If a pre-existing test asserted native-dtype image pixels (e.g. a uint16 assertion), update it to the canonical [0,1] contract and note it in the commit.

- [ ] **Step 2: Real-TIFF replay (primary verification)**

Run: `pytest tests/e2e -v`
Expected: PASS if `examples/data/folding_screen.tif` is present (else skipped). The fluorescence TIFF must detect as `bright_on_dark`, `ok`, not inverted.

- [ ] **Step 3: Manual inverted-copy spot check**

```bash
python - <<'PY'
import numpy as np
from skimage import io
from gelpy.core.image_io import canonicalize_image
img = io.imread("examples/data/folding_screen.tif")
inv = (img.max() - img).astype(img.dtype)
io.imsave("/tmp/folding_inverted.tif", inv, check_contrast=False)
_, m0 = canonicalize_image("examples/data/folding_screen.tif")
c1, m1 = canonicalize_image("/tmp/folding_inverted.tif")
c0, _ = canonicalize_image("examples/data/folding_screen.tif")
print("orig:", m0.polarity, m0.polarity_confidence, "| inverted-file:", m1.polarity, m1.inverted)
print("canonical match after re-inversion:", np.allclose(c0, c1, atol=1e-6))
PY
```

Expected: `orig: bright_on_dark ok | inverted-file: dark_on_light True` and `canonical match ... True` (both routes converge to the same signal-bright canonical image).

- [ ] **Step 4: Update `CLAUDE.md`**

In the `core/` bullet, after the `image_io` mention, add:

```markdown
  `image_io.canonicalize_image` is the single input funnel: any dtype/RGB(A)/multi-page source
  -> a canonical **float64 [0,1], signal-bright, finite** 2-D array + a recorded `ImageMeta`
  (channel + polarity + page). Polarity is auto-detected (skew + tail-asymmetry), confidence-
  flagged, and overridable (`--invert`/`image set`); detection runs once at init, and the
  frozen `ImageMeta` (folded into `image_hash`) drives a deterministic `apply_image_meta`
  replay in the cached pipeline. Downstream stays polarity-agnostic — signal is always bright.
```

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md tests/
git commit -m "docs+test: image-input canonicalization integration + CLAUDE.md note"
```

---

## Self-Review

**Spec coverage:** polarity auto-detect+flag+override (Tasks 3,4,7,8,9) · canonical float [0,1] (Tasks 2,4) · mean default for colored (Task 2) · non-finite guard (Task 4) · multi-page page-select + warning (Tasks 4,7) · ImageMeta persistence (Tasks 1,7) · image_hash fold-in (Task 5) · deterministic replay (Tasks 4,6) · warnings via summary/status (Tasks 1,7,8) · CLI + image set (Task 9) · migration/back-compat (legacy default_factory, Task 1; verified Task 10) · README/CLAUDE.md docs (Tasks 9,10). No gaps.

**Placeholder scan:** none — every code/test step carries full code and exact commands.

**Type consistency:** `canonicalize_image(path, *, channel, invert, page) -> (ndarray, ImageMeta)`, `apply_image_meta(path, meta) -> ndarray`, `collapse_channels(...) -> (ndarray, str, int|None)`, `detect_polarity -> PolarityDecision`, `image_warnings(meta) -> list[str]`, `build_image_ref_and_meta -> (ImageRef, ImageMeta)` are used identically wherever they recur. `ImageMeta`/`RunUpdateSummary`/`RunStatus`/`InitRunRequest`/`SetImageRequest` field names match across tasks.
