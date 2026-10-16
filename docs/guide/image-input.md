---
title: What gelpy does to your image first
audience: both
summary: How any TIFF or PNG becomes the canonical signal-bright float array every later stage measures, and how to check the two decisions -- channel and polarity -- that can silently invert your numbers.
subject:
  - src/gelpy/core/image_io.py
  - src/gelpy/core/contrast.py
  - src/gelpy/schemas/image.py
  - src/gelpy/schemas/contrast.py
subject_digest: d64d493a554e3f29
---

## One funnel, one contract

`core/image_io` is the only way pixels enter gelpy: `canonicalize_image` on the `init` /
`image set` path, `apply_image_meta` on every cached replay, both over the same
`decode_image` -- no other module opens an image file. What it hands the next stage is
`float64`, 2-D, values in `[0, 1]`, every pixel finite, and **signal bright**. The `[0, 1]`
half of that contract is a property of canonicalization, not of every later stage: background
removal defaults to `clip_negative = False` and passes on a *signed* array, so line profiles
and fits routinely see negative values (see [background.md](background.md)).

Before step 1, `decode_image` reads the file header and refuses a file over the
[size limits](#size-limits) without decoding a pixel. Order of operations:

1. decode the file; select one page if it is a stack
2. reject non-finite pixels, before any arithmetic
3. scale to float `[0, 1]`
4. collapse color channels to 2-D
5. decide polarity: bright-band-on-dark, or dark-band-on-light
6. apply `1 - x` if and only if the source is dark-on-light

Steps 1, 4 and 5 are decisions, each recorded as a categorical outcome in `ImageMeta` in
`recipe.json`, beside the file identity (`ImageRef`). Later runs never re-detect: replay
consumes the recorded outcome and contains no detection code. That is what makes a run folder
replay bit-exact, and why `ImageMeta` is editable intent rather than a log line.

Six inputs feed `image_hash` (`runfolder/stages.py`): the source `sha256`, `channel_strategy`,
`channel_index`, `inverted`, `page` and the gelpy version. Change one and every downstream
stage recomputes. The rest of `ImageMeta` -- `polarity_confidence` included -- is a record, not
a cache key. Full field list: `gelpy schema ImageMeta`.

## First thing to do on a new image

Set the two variables the blocks below use. This block is skipped by the doc test, which sets
both itself and points them at `examples/data/example_gel_1.tif`.

```bash
# doc-test: skip
export RUN=runs/my-gel
export IMAGE=examples/data/example_gel_1.tif
```

`init` prints the whole interpretation it just froze; `status` reprints the fields that decide
whether your numbers have the right sign.

```bash
gelpy init "$RUN" "$IMAGE" --n-lanes 20
gelpy status "$RUN" | grep -E '"polarity|"warnings"'
```

```console
    "source_channels": 1,
    "source_pages": 1,
    "page": 0,
    "channel_strategy": "gray",
    ...
  "polarity": "bright_on_dark",
  "polarity_confidence": "ok",
  "warnings": [],
```

Five fields are worth a glance. `source_pages` (did it see your whole stack?) and
`source_channels` with `channel_strategy` (which channel got measured?) come only from
`init` / `image set` and `recipe.json`. `polarity`, `polarity_confidence` and `warnings` are
the only image fields `RunStatus` carries, so `status` re-checks the sign but not the channel.

## dtype scaling: integers keep an absolute scale, floats may not

- **Integer sources** are divided by the dtype maximum. A dim uint16 scan stays dim: 1000
  counts becomes `1000/65535 = 0.01526`, not `1.0`, so two uint16 TIFFs of the same gel at
  different exposures stay on a comparable scale. (Signed integers land in `[-1, 1]` first;
  they are min-max stretched only if a pixel is actually negative.)
- **Float sources** are passed through untouched inside `[0, 1]` and min-max stretched only
  if a pixel falls outside it. Where that stretch fires it is *relative*: two such TIFFs at
  different exposures come out identically scaled and are no longer comparable. Two floats
  already normalized into `[0, 1]` are not stretched and stay comparable.
- Scaling is global -- one factor for the whole array, though the out-of-range branch is
  `(f - lo) / (hi - lo)`, which shifts as well as scales -- so a channel collapse is never
  distorted per pixel.

The example gel is uint16 with a canonical median of `0.0026`, 99.5th percentile `0.114`,
maximum `0.659`. Nothing rescales that into a full `[0, 1]` swing, which is why the untouched
view of it looks black (see [display contrast](#display-contrast-is-display-only)).

## Color channels

`--channel` accepts `auto`, `mean`, `luminance`, `r`, `g`, `b`, `dominant`, `max`. What gets
*recorded* can additionally be `gray` (the source was already 2-D) or `gray_rgb` (a
multi-channel source whose channels agree within tolerance across at least 99.9% of pixels,
so the mean loses nothing that matters -- the other 0.1% are unconstrained and may differ by
any amount).

- `auto` on a genuinely colored image resolves to `mean`: the intensity-to-ground-truth
  mapping of a color gel photograph is unknowable from the image alone, so the least-biased
  collapse wins.
- `dominant` picks the channel with the widest robust dynamic range (99.5th minus 0.5th
  percentile) and freezes the index in `channel_index`, so replay never re-runs the argmax.
- A trailing alpha channel is dropped and is never a mask or a weight -- for a 2-channel
  gray+alpha source as well as for 4-channel RGBA. Alpha is opacity, not signal: adding an
  opaque *or* a varying alpha channel to a gel changes no canonical pixel
  (`tests/core/test_image_io.py::test_alpha_is_never_measured_as_signal`). Verified on a
  uint8 `(64, 64, 2)` TIFF with gray 50 and alpha 255: canonical value `50/255 = 0.196`,
  not the gray/alpha mean `0.598`.

One trap: the grayness test runs **before** your request is honored. If the 99.9th percentile
of the per-pixel channel spread is below `0.02` the collapse is `gray_rgb` whatever you asked
for, and a 2-D source is a passthrough recorded as `gray`. So asking for a channel that does
not exist is silently ignored rather than refused:

```bash
gelpy init "${RUN}_lum" "$IMAGE" --channel luminance --n-lanes 20 \
  | grep -E '"source_channels"|"channel_strategy"'
```

```console
    "source_channels": 1,
    "channel_strategy": "gray",
```

That is correct -- there is nothing to weight on a single-channel TIFF -- but it means
`channel_strategy` in the output, not your flag, records what happened.

## Multi-page sources

A stack is pages along axis 0 (a length-1 axis, as in `(4, 1, Y, X)` or `(Y, X, 1)`, is dropped first) and `--page K` selects one; a stack of more than 64 pages is
refused before it is decoded (see [size limits](#size-limits)). `source_pages` reports how many
the decoder found, and a stack raises a non-fatal warning naming the page in use:

```console
  "warnings": [
    "source image has 5 pages; using page 1. Select another with --page (init or `image set`)."
  ],
```

- **`page` records what you asked for, not what was used.** An out-of-range page is clamped
  silently (`--page 9` on a 5-page stack reads page 4) but stored as `9`. Replay clamps the
  same way, so pixels stay deterministic -- but `--page 9` and `--page 4` are then two cache
  keys for one set of pixels.
- **The finite check is per selected frame, not per file.** A NaN on page 1 does not stop page
  0 from loading.
- **A file tifffile could only partly read is reported, not repaired.** tifffile logs a damaged
  TIFF and carries on: a six-page stack whose page list breaks after page three comes back as the
  three pages it could reach, and only the log says so. gelpy keeps what tifffile logged during
  the read (warnings and errors: at most five different messages, without the folders of the
  file, in `reader_warnings` of the image record) and `init`, `image set` and `status` carry it
  in `warnings`: `tifffile reported while reading the image: <tifffile.TiffPages @8> invalid page
  offset 29264. The file may be damaged or truncated: check source_pages, and export it again if
  pages are missing.` A file that is cut off in its pixel data fails instead (`decode_error`).

### Caveat: 3- and 4-page grayscale stacks are read as color

The `(pages, H, W)` versus `(H, W, channels)` ambiguity is resolved in favor of channels when
the last axis is 2, 3 or 4. With `scikit-image` 0.26.0 a genuine 3- or 4-page grayscale TIFF
arrives already transposed to `(H, W, 3)` / `(H, W, 4)`, so it is collapsed as color, your
pages get averaged into one frame, and `--page` does nothing. 2-page and 5-plus-page stacks
decode as pages. The tell is `source_pages: 1` with `source_channels: 3`:

```bash
STACKS="$(mktemp -d)"
python - "$STACKS" <<'PY'
import sys, numpy as np, tifffile
for n in (3, 5):
    pages = np.stack([np.full((120, 300), 1000 * (i + 1), "uint16") for i in range(n)])
    tifffile.imwrite(f"{sys.argv[1]}/stack{n}.tif", pages, photometric="minisblack")
PY
for n in 3 5; do
  echo "== ${n}-page stack =="
  gelpy init "${RUN}_p${n}" "$STACKS/stack${n}.tif" --n-lanes 3 --page 1 \
    | grep -E '"source_pages"|"source_channels"|"channel_strategy"'
done
```

```console
== 3-page stack ==
    "source_channels": 3,
    "source_pages": 1,
    ...
== 5-page stack ==
    "source_channels": 1,
    "source_pages": 5,
    ...
```

Split a 3- or 4-page stack outside gelpy and point a run folder at one page.

## Non-finite pixels are refused, not repaired

One NaN or Inf aborts the load rather than propagating: `np.percentile` returns NaN for the
whole array and a least-squares fit does not recover, so a single bad pixel would otherwise
take polarity detection, the display rescale and every fit with it.

```bash
# doc-test: expect-failure
BAD="$(mktemp -d)"
python -c "import numpy as np, tifffile; a = np.zeros((64, 64), 'float32'); a[7, 9] = np.nan; tifffile.imwrite('$BAD/nan.tif', a)"
gelpy init "$BAD/run" "$BAD/nan.tif" --n-lanes 4
```

```console
{
  "error": "image contains 1 non-finite (NaN/Inf) pixels; clean the source before analysis.",
  "code": "nonfinite_image",
  "detail": "Replace the NaN/Inf pixels (export the image again) and run `gelpy init` on the clean file.",
  "available": null
}
```

## Size limits

A file can declare far more pixels than any gel has: a header that says 40000 x 40000 in a few
bytes, or a stack of thousands of pages. Decoded, it would exhaust the machine's memory before
gelpy could object. So gelpy reads the file header first and refuses, with
`code: "image_too_large"`, a file that would decode to more than 100 million pixels, counted as
width x height summed over its pages, or that has more than 64 pages. Nothing is decoded before
that check. The count is of what the decoder will hold: every 2-D plane of a stack is a page,
whatever its axes are called (an ImageJ or OME stack of time points and z slices included), and a
color channel adds no pixels. A length-1 axis is counted as the file stores it: a `(4, 1, Y, X)`
stack is four pages. An axis of 3 or 4 directly before or after Y and X is read as color
channels, as the decoder reads it (the caveat under Multi-page sources), so a stack of three or
four pages counts as one page of Y x X pixels, up to four times fewer than the values held, the
same factor as for a four-channel image.
`GELPY_MAX_PIXELS` and `GELPY_MAX_PAGES` change the two limits for one process.
<!-- claim: image-size-limits -->
Each takes a positive whole number of at most 18 digits, written with the digits 0-9 (an empty
value counts as unset); any other value, `1_000` and `+5` included, is an `invalid_value` error.
The limits are read at
every decode and a run folder does not record them, so a later command that decodes the image
again needs the same variables. A file whose header no reader understands -- a text file with a
`.tif` name, a TIFF that is only a header -- is a `decode_error` instead, and so is a GIF with
more than one frame, because a GIF header does not bound the size of its later frames; export
the frame you want as PNG or TIFF
(`tests/core/test_image_limits.py::test_any_gif_with_more_than_one_frame_is_refused`). The
example gel is 6.4 million pixels on one page, so a limit of one million refuses it:

```bash
# doc-test: expect-failure
GELPY_MAX_PIXELS=1000000 gelpy init "${RUN}_capped" "$IMAGE" --n-lanes 20
```

```console
{
  "error": "'example_gel_1.tif' would decode to 6,394,432 pixels; the limit is 1,000,000.",
  "code": "image_too_large",
  "detail": "Crop or downsample the image, or raise the limit with GELPY_MAX_PIXELS.",
  "available": null
}
```

The lane count is the other size a caller supplies, and it is capped too.
<!-- claim: lane-count-cap -->
A request, a `recipe.json` or a `fit_config.json` holds at most 1000 lanes: `n_lanes`, the count
of `lanes set --linspace`, `--x-center`, the labels and the terms of a lane selector. A larger
request is refused before any work is done, with a `validation_error` that names the field and
the limit (`n_lanes: Input should be less than or equal to 1000`), and a recipe edited to hold
more lanes does not load. This limit has no environment variable.

## Polarity: the one decision that can silently invert every number

A fluorescence scan has bright bands on a dark gel. A stained gel photographed in transmission,
or a scanner that writes an inverted TIFF, has dark bands on a light gel. Get this backwards
and gelpy measures the *gaps between* your bands.

Nothing re-checks the decision: no module in `core/lanes`, `core/background`,
`core/line_profiles` or `core/peaks` reads `polarity` or `inverted` at all. What protects you
is not a check but a collapse. Measured on the example gel, `image set --invert` then
`fit run` returns `status: "empty"`, `confidence: "low"` and zero bands on **all 20 lanes**,
against `status: "ok"` and 3 to 8 bands per lane at the correct polarity: the baseline
follows the now-high background and the residual *dips* never seed a peak. Whether some other
gel could yield a believable wrong fit instead of an empty one is unverified.

Detection uses two cues on the flattened intensity histogram, on the premise that background
occupies most of the frame, so the long tail is signal:

- **Fisher skewness** `g1 = mean((x - mu)^3) / sd^3`. Its sign says which tail is long.
- **Tail asymmetry** from the 0.5th, 50th and 99.5th percentiles:
  `(p99.5 + p0.5 - 2*median) / (p99.5 - p0.5)`, in `[-1, +1]`. It says which tail reaches
  further from the median, and being percentile-based it is far less sensitive than a cubed
  moment to a handful of saturated specks.

The two must **agree in sign**, and at least one must clear its floor: `|g1| >= 0.5` **or**
`|tail_asym| >= 0.1`. Both positive is bright-on-dark, no inversion. Both negative is
dark-on-light, and `1 - x` is applied. Anything else -- disagreement, or both cues weak --
records `polarity_confidence: "low"`, assumes bright-on-dark, and does **not** invert; the
low-confidence fallback is deliberately the reversible one, and it raises a warning that
`init`, `image set` and `status` all carry.

On `examples/data/example_gel_1.tif` the cues are `g1 = +3.61` and `tail_asym = +0.97`
(measured with `core.image_io.detect_polarity`): unambiguous bright-on-dark.

### Overriding it

`--invert` asserts *the source is dark-on-light*, so gelpy applies `1 - x`. It does not mean
"flip whatever you decided". Either `--invert` or `--no-invert` sets
`polarity_source: "manual"`, which suppresses detection; `image set --auto` re-runs it and is
the way back.

```bash
gelpy image set "$RUN" --invert | grep -E '"polarity"|"inverted"|"polarity_source"'
gelpy image set "$RUN" --auto   | grep -E '"polarity"|"inverted"|"polarity_source"'
```

```console
    "polarity": "dark_on_light",
    "inverted": true,
    "polarity_source": "manual",
    "polarity": "bright_on_dark",
    "inverted": false,
    "polarity_source": "auto",
```

`init` takes `--invert` / `--no-invert` too, and omitting them there *is* the auto path --
there is no `--auto` on `init`.

### Caveat: two ways a wrong polarity survives

- **Signal over more than half the frame.** The detector's premise is a hard requirement.
  Measured with `detect_polarity` on synthetic 200x400 frames of a bright block on a dark
  ground: at bright area fractions 0.05 and 0.2 it reports bright-on-dark, confidence `ok`; at
  0.5 the cues vanish and it flags `low`; at 0.6 it reports `dark_on_light` with confidence
  `ok` and inverts a correctly-oriented image (`g1 = -0.41`, under its floor, but
  `tail_asym = -1.00`, over its own -- either cue alone is enough). A heavily over-exposed gel,
  or a crop that is nearly all band, can be flipped with no warning. Check the `raw` render.
- **`image set` launders a `low` flag.** Given no polarity flag, `image set` passes the current
  `inverted` boolean through, and a boolean means "manual" -- so an unverified auto-detected
  `low` becomes `polarity_source: "manual"`, `polarity_confidence: "ok"`, `warnings: []`, with
  no recompute, since `image_hash` excludes the confidence. Verified on a near-symmetric
  synthetic TIFF: `gelpy image set RUN --channel mean` alone erases the warning. Confirm a
  polarity with `--invert` / `--no-invert`; bring the flag back with `--auto`.

## Display contrast is display only

`ContrastConfig` (a gamma, and a linear intensity range) affects rendered PNGs and nothing
else. No stage hash in `runfolder/stages.py` reads `recipe.contrast`, so `gelpy contrast set`
never invalidates a cache and never moves a fitted number; only `render/images.py` and the
interactive lane picker consume it.

Defaults: `gamma = 0.1` with `gain = 1.0`, and `IntensityRange(low=0.0, high=0.9999,
mode="percentile")` -- the 0th percentile so the dark end is never crushed, to the 99.99th so
at most 0.01% of pixels saturate. A gamma of 0.1 is an aggressive brightening, chosen because
a gel's bands are a small fraction of full scale. `gain` is not clipped, so a gain above 1
pushes a float image past 1.0 and lets matplotlib rescale the whole mapping.

The four `image show --kind` values differ in *which array* and *which contrast*:

| `--kind` | array | contrast applied |
|---|---|---|
| `raw` | canonical source array | **none** (see the autoscale note below) |
| `linear` | canonical source array | the recipe's `intensity_range` |
| `non_linear` | canonical source array | the recipe's gamma |
| `corrected` | background-corrected array | the recipe's gamma |

```bash
gelpy image show "$RUN" --kind raw
gelpy image show "$RUN" --kind non_linear --lanes
```

One thing every kind shares: `render_gel_image` calls `imshow` with no `vmin`/`vmax`, so
matplotlib min-max normalizes the display. `raw` applies no gelpy contrast, but the PNG is
still a linear stretch of the canonical array (min to black, max to white; divisor 0.6586 on
the example gel) -- *relative*, not absolute, canonical intensity. The stretch is monotonic,
which is why `raw` remains the view to check when something looks wrong: it is the only one
that reflects what canonicalization and polarity detection handed the pipeline.

On the example gel it is nearly black. Pushing the canonical array through each `--kind`'s
transform and then that autoscale, x255: `raw` has a median displayed level of 1/255 and a
99th percentile of 40/255, against 66/255 and 180/255 for `non_linear`. Dark is expected for
a fluorescence gel; a `raw` view that is bright everywhere with dark bands means polarity
went the wrong way.

Two smaller things: `corrected` renders the same array as `non_linear` until a background stage
is configured, since background removal is a passthrough when off; and any other `--kind` is
refused (`validation_error`, nothing written), because the kind is also the PNG's file name.

## Where to look next

- `gelpy schema ImageMeta`, or [../reference/contracts.md](../reference/contracts.md)
- [background.md](background.md) -- what happens to the canonical array next
- [troubleshooting.md](troubleshooting.md) -- symptoms first
