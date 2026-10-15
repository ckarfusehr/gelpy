---
title: The run folder and the hash chain
audience: both
summary: What a run folder is, which files are precious, and how the four-stage hash chain decides what recomputes.
subject:
  - src/gelpy/runfolder/stages.py
  - src/gelpy/runfolder/cache.py
  - src/gelpy/runfolder/pipeline.py
  - src/gelpy/runfolder/layout.py
  - src/gelpy/runfolder/manifest.py
  - src/gelpy/runfolder/hashing.py
  - src/gelpy/runfolder/recipe_io.py
  - src/gelpy/runfolder/fit_config_io.py
  - src/gelpy/utils/atomic.py
  - src/gelpy/schemas/recipe.py
  - src/gelpy/schemas/run.py
  - src/gelpy/schemas/common.py
subject_digest: 07460f4cd225a93f
---

Two ideas carry the whole design. A **run folder** is the unit of work and the unit of sharing.
A **hash chain** over four stages decides what a change actually invalidated. Everything else in
gelpy is downstream of those.

## The run folder replaces the object you used to pickle

In 0.x the analysis lived in a `Gel` object and was shared as a `.pkl`. A run folder holds the
same state as plain files: JSON you can read, CSV you can open in anything, PNGs, and a Markdown
report. A collaborator needs no Python and no gelpy to read the output, and re-running any
command reproduces it.

A run folder *is* its `recipe.json`. `RunPaths.exists()` is literally
`self.recipe.exists()`, and a command that cannot find that file fails with the
machine-readable code `not_found`. The source image is never copied in: `recipe.json` records
its path, size, mtime and sha256, so many run folders can analyse one 12.8 MB TIFF with
different parameter sets without duplicating it. The path is recorded relative to the run
folder where possible (absolute when the image shares no directory with the run folder below
the filesystem root), and the cache paths in `manifest.json` are relative to the run folder. So
a run folder opens from any working directory and can be moved together with its image.

For the file-by-file table, read [reference/run-folder.md](../reference/run-folder.md) — it is
generated from the code, so it cannot describe a file that is not there.

## Precious, disposable, derived

Three categories, and knowing which is which tells you what is safe to delete.

- **Precious** — `recipe.json` (source image identity, lane layout, contrast, background choice)
  and `fit_config.json` (the `(profile, fit)` settings last applied to each lane, sparse: only
  lanes actually fit appear). This is *intent*. Nothing regenerates it. Both are meant to be
  edited by hand or by an agent. `log/history.jsonl` is append-only and never rewritten either,
  though nothing in gelpy reads it back yet — `replay` is a deferred feature.
- **Disposable** — `manifest.json` (the cache ledger) and everything under `cache/`. Deleting
  either loses no work, but nothing regenerates on its own: a stage is re-recorded only by a
  command that computes it or reuses it from the cache (a fit cache hit also records the image,
  background and profile it did not load). `status` and `results show` recompute nothing, so
  after a ledger wipe they report zero fits until a fit command re-records them (the block
  below does exactly that).
- **Derived** — `results.csv`, `results.json`, `report.md`, `render/*.png` and the HTML report
  folder `report/`. Assembled from the cached fits without recomputing them.

`.gelpy.lock`, the advisory lock file that every command writing the folder holds, belongs to
none of the three: leave it alone; it is safe to remove only while no gelpy command runs.

gelpy never writes through a link inside a run folder: a link where it writes a file is replaced
by the file, and a link where it keeps a folder (`cache/`, `render/`, `log/` or a folder below
them), or at `log/history.jsonl` or `.gelpy.lock`, is refused with `invalid_value` before
anything is written, until you delete the link. The run folder itself may be a link: you pass it.

The loaders treat the two kinds of JSON deliberately differently. `load_manifest` catches every
exception and returns a blank ledger — a wrong ledger costs CPU. `load_fit_config` lets the
`ValidationError` propagate — a silently rebuilt *intent* costs you a wrong number. Corrupt
intent must never be masked.

Every contract sets `extra="forbid"`, so a misspelled field in a hand-edited `recipe.json` or
`fit_config.json` is a hard validation error rather than a silently ignored line.
<!-- claim: extra-fields-are-rejected -->

## A run folder to experiment on

Every block below uses these two variables:

```bash
# doc-test: skip
export IMAGE=examples/data/folding_screen.tif   # your gel
export RUN=/tmp/gelpy-concepts                  # the run folder to build
```

The default layout `init` writes is a geometric guess spanning 5% to 95% of the image width and
is meant to be replaced (see [lane-placement.md](lane-placement.md)); it is good enough to
demonstrate caching, which is all it does here. Two lanes, because the point needs no more.

```bash
gelpy init "$RUN" "$IMAGE" --n-lanes 20
gelpy background remove "$RUN" --model robust_plane
gelpy fit run "$RUN" --lane 0 --lane 1 | head -8
```

The fit reports everything as recomputed — nothing is cached yet:

```console
{ "recomputed": [ "background", "fit:lane=0", "fit:lane=1", "image",
                  "profile:lane=0", "profile:lane=1" ],
  "reused": [], ... }
```

## The hash chain

```
image -> background -> profile[lane] -> fit[lane]
```

Four fixed stages, hand-rolled — there is no stage registry. Each stage has one hash, and each
hash chains off the one upstream, so an edit propagates exactly as far as it should and no
further.

| stage | hash is computed from | cached artifact |
|---|---|---|
| `image` | source `sha256` + `channel_strategy` + `channel_index` + `inverted` + `page` + the gelpy version | `cache/image/<hash>.npy` |
| `background` | `image_hash` + model name + **validated** params | `cache/background/<hash>.npz` |
| `profile[lane]` | `background_hash` + that lane's `x_center_px` + `profile_width_px` | `cache/profiles/lane_NNN__<hash>.json` |
| `fit[lane]` | `profile_hash` + the whole `LineProfileConfig` + the whole `FitConfig` | `cache/fits/lane_NNN__<hash>.json` |

The four interpretation fields are exactly the inputs of `apply_image_meta`, the deterministic
replay that turns file bytes into the canonical float64 array; the `sha256` pins *which* bytes.
Nothing else in `ImageMeta` touches a pixel.

Because every other hash chains off `image`, a new gelpy version recomputes every stage on first
use and never reuses a cache an older version wrote.

With no background stage configured, `background_hash` returns `image_hash` unchanged: a
passthrough, so no background cache entry is ever created or consulted. `profiles extract`,
`fit run` and `fit refit-lane` do not run in that state unless `--allow-raw` says the raw image is
meant.

The `FitConfig` folded into `fit_hash` includes the RNG `seed`, which is what makes even the
stochastic `bayesian` tier content-addressed and cacheable.

### What is deliberately left out, and why

Every exclusion below is a decision, not an oversight.

- **Slice bounds and normalization are not in `profile_hash`; they feed `fit_hash`.**
  <!-- claim: profile-hash-excludes-slice-and-normalization -->
  The cached profile is the full, unsliced, un-normalized column mean. Slicing it is a view;
  normalizing it is a divide. Re-extracting it means averaging `profile_width_px` columns over
  the full image height again. So one cached profile serves every slice and normalization
  choice for that lane, and re-slicing costs only a refit.
- **Cosmetics are in no hash at all.** A lane's `label`, `lanes.label_rotation` and the entire
  `contrast` block. Verified directly: changing `contrast.gamma.gamma` from 0.1 to 0.4 moves
  none of `image_hash`, `background_hash` or `profile_hash`. Renaming a lane must not cost you
  a fit, so a cached fit keeps the label it was fitted under; the results are assembled with the
  label the recipe has now.
- **Image provenance is not in `image_hash`.** `polarity`, `polarity_source`,
  `polarity_confidence`, `polarity_reason`, `original_dtype`, `original_shape`,
  `source_channels`, `source_pages` all describe *how the decision was reached*. `inverted` is
  the operative field; the rest are its human-readable twins. Re-running detection and landing
  on the same `inverted` with a different `reason` must not invalidate a 51 MB image cache and
  everything downstream of it.
- **The source image's path, size and mtime are not in `image_hash`** — only its `sha256`. The
  same bytes at a different path are the same image.

Two properties follow from the table, and both are why this is cheap:

- **No hash ever reads pixel data.** <!-- claim: hashes-never-read-the-image-pixels -->
  Every stage hash is a sha256 over the canonical JSON of the recipe and a few small config
  objects. For a 20-lane run, `recipe.json` is 3.3 KB. Deciding what to recompute is a handful
  of sha256-over-small-JSON operations whether the source is a 12.8 MB TIFF or a 2 GB one.
- **Hashes run on *validated* models, not on the JSON text.** Defaults are resolved and types
  coerced first, so a hand-written `"params": {}` and the fully-defaulted form hash identically
  (verified). Hand-editing `recipe.json` still hits the cache; it does not fall off it.

## Cache filenames carry the content hash

A changed setting writes a *new* file rather than invalidating the old one. So changing your
mind back is a cache hit, not a recompute.

```bash
gelpy fit refit-lane "$RUN" --lane 0 --slice 200 1200 | head -8
ls "$RUN/cache/profiles" "$RUN/cache/fits"
```

The refit reports `recomputed: ["fit:lane=0"]` with `image`, `background` and `profile:lane=0`
all reused, and `cache/fits/` now holds two files for lane 0 against one profile file — exactly
the asymmetry the `profile_hash` exclusion buys.

```bash
gelpy fit refit-lane "$RUN" --lane 0 --poly-order 0 | head -4
gelpy fit refit-lane "$RUN" --lane 0 --poly-order 2 | head -4
```

The first recomputes. The second reports `"recomputed": []` — the earlier fit file was still
there under its own hash.

## Why per-lane fit intent is a separate file

`fit_config.json` is not part of the manifest, so deleting the disposable ledger cannot destroy
what you asked for. <!-- claim: fit-config-is-not-on-the-manifest -->
`Recipe` deliberately excludes fit config too: one recipe backs several fit passes (broad,
refit, tight), so a fit is a request, not a recipe field.

```bash
rm "$RUN/manifest.json"
gelpy status "$RUN" | grep '"n_fit"'
gelpy fit refit-lane "$RUN" --lane 0 | head -4
gelpy fit refit-lane "$RUN" --lane 1 | head -4
gelpy status "$RUN" | grep '"n_fit"'
```

Measured on this run: `n_fit` reads 0 immediately after the ledger is deleted, then 2 again
after the two refits, with `"recomputed": []` on both — full recovery at zero compute cost.
`refit-lane` with no other options re-applies each lane's stored settings, which is only
possible because the store outlived the manifest.

## Atomic writes and corrupt-file recovery

Briefly, because it is the part you should never have to think about:

- Every file gelpy writes in a run folder -- the cache, the precious JSON, `results.csv`,
  `report.md`, the figures, the HTML report -- goes through one writer, `utils/atomic.py`, except
  the two it opens in place, `log/history.jsonl` (appended to) and `.gelpy.lock`, which are opened
  without following a link. The writer writes to a temp file in the same directory,
  named `.<name>.<random>.tmp` and created exclusively with the mode `0o666` (so a link planted at
  a name gelpy would have used is never followed, and the kernel takes the umask off the mode: the
  file gets the mode `open` would have given it), then `os.replace`. Same filesystem, so the rename
  is atomic; a link at the destination is replaced and never written through; a failed compute
  leaves the old file, no partial file and no orphan temp. A process that is killed in the middle
  of a write (`kill -9`, a power cut) can leave its temp file: the report folder's are removed by
  the next `report html` and by `init --force`, and one in `cache/` or `render/` does no harm,
  because nothing reads a dot-named file there. (`np.save`/`np.savez` append their own extension
  to a path and would mangle the temp name, so those two are handed the open file object instead,
  and so is matplotlib's `savefig`.)
- An unreadable cache file is unlinked and recomputed rather than being fatal. The fit stage
  goes further: a structurally valid JSON that no longer validates as a `LaneFitResult` — after
  a schema change, say — also counts as corrupt.
- Two damage modes escape that recovery and raise instead: a truncated `.npz`
  (`zipfile.BadZipFile`, which is not an `OSError`) and a cache JSON holding non-UTF-8 bytes
  (`UnicodeDecodeError`, caught by neither of the two exceptions the JSON path handles). Both
  verified directly. gelpy's own atomic writes cannot produce either, so it takes external
  damage — an interrupted copy, a full disk. Delete the offending file and re-run.

One caveat on the provenance fields: `recomputed`/`reused` are decided by whether the cache file
existed when the stage started. Overwrite a lane's profile cache with garbage and the profile is
silently and correctly re-extracted, but the summary still lists it under `reused` (verified).
The data is right; the label is not.

## What the caching actually buys

Measured on `examples/data/folding_screen.tif` (4688 x 1364 px, 12.8 MB TIFF), 20-lane layout,
`robust_plane` background, `grow_select` fit of lanes 0 and 1, one laptop, three fresh run
folders, timed through the `gelpy` entry point:

| command | wall time, 3 reps |
|---|---|
| `fit run --lane 0 --lane 1`, cold | 5.8 / 6.1 / 6.2 s |
| the identical command again | 1.8 / 2.0 / 1.8 s |
| `python -c "import gelpy.cli.app"` alone | 1.7 / 1.8 / 1.6 s |

Read the ratio, not the seconds. The absolute numbers move with machine load and with how you
invoke the CLI (`python -m gelpy` came out 0.5-1.2 s slower cold in the same session), so your
run will not reproduce these. What holds is the last two rows: paired against a bare interpreter
start measured in the same session, the warm re-run costs 0.13-0.30 s of actual work over six
pairs — a couple of hundred milliseconds. The fit stage hits its cache and reads two small JSON
files, and because each stage checks its own cache *before* pulling on its upstream, the TIFF is
never decoded and the background surface is never re-fitted. The disk price is real, though: the
canonical image is 51 MB of float64 and the corrected image another 51 MB, so this warmed run
folder is 102 MB for a 12.8 MB source.

The same chain also catches upstream edits. Shift lane 0's `x_center_px` by 7 px in
`recipe.json` and its expected `fit_hash` no longer matches the recorded one: `status` moves
that lane from `n_fit` to `n_stale` and names the lane's `profile` and `fit` stages under
`stale_stages`, and `results show` drops its rows rather than reporting a fit that no longer
reflects intent. A fit is current when the fit the recipe calls for is in the cache, not when the
ledger's recorded hash says so: the cache keeps every fit it computed, so after you change your
mind back `stale_stages` is empty again and so is `n_stale`, and `results show` has the fits back,
with no fit command. A deleted cache file is judged the same way, and stays in `stale_stages` until
a command that needs it runs: a fit that is cached never reads what is upstream of it, so `fit run`
leaves a missing image, background or profile listed, and a render or `profiles extract` computes it
again.

## A typo is an error, not a silent no-op

Hand-editing is expected, so hand-editing has to fail loudly. This block is *meant* to fail:

```bash
# doc-test: expect-failure
mkdir -p "$RUN.typo"
cp -R "$RUN/." "$RUN.typo/"
sed 's/"poly_order"/"polly_order"/' "$RUN/fit_config.json" > "$RUN.typo/fit_config.json"
gelpy status "$RUN.typo"
```

```console
{
  "error": "FitConfigStore: lanes.0.fit.baseline.polly_order: Extra inputs are not permitted. `gelpy schema FitConfigStore` prints the contract.",
  "code": "validation_error",
  "detail": "[{\"type\": \"extra_forbidden\", \"loc\": [\"lanes\", \"0\", \"fit\", \"baseline\", \"polly_order\"], \"msg\": \"Extra inputs are not permitted\", \"input\": 2}]",
  "available": null
}
```

It exits non-zero and writes that to stderr, naming the exact path of the bad key.

## Known limitations

All four were reproduced against the current code while writing this page.

1. **A run folder written by 1.0 is not portable.** 1.0 recorded the image path in
   `recipe.json` and every `cache_path` in `manifest.json` exactly as the command line gave
   them, so a relative path meant relative to the working directory. Later versions open such a
   folder from the directory it was made in, and nowhere else: a command that has to re-read
   the image fails with `code: "not_found"`, and the ledger's paths point at the old location.
   Measured on a 1.0 run moved together with its image and opened from a third directory:
   `fit refit-lane` failed with `not_found` for the image. After `source_image.path` in
   `recipe.json` was set to the image's path relative to the run folder, the same command
   worked and re-recorded that lane's stages relative to the run folder. A stage that no
   command has re-recorded keeps its 1.0 record. Run folders created by 1.1 or later have no
   such limit.
2. **`gelpy init --force` on an existing run folder replaces its precious files.** Without the
   flag, `init` refuses a folder that already holds a run (`code: "run_exists"`). With it,
   measured: it reset `background` from `robust_plane` to `null` in `recipe.json`, emptied
   `fit_config.json` (2 lane entries to 0), cleared the manifest (2 stage records to 0), and
   left 7 of the folder's 8 cache files unreachable. Only the canonical image `.npy` is still
   addressed by the rewritten recipe: `background: null` makes `background_hash` equal
   `image_hash`, which re-keys both profiles and all four fits. It also removes what the old run
   derived (`results.csv`, `results.json`, `report.md`, the files the marker of `report/` lists
   and the temp files of killed writes there, and `render/`); the cache and `log/history.jsonl`
   stay, and so does a `report/` that no marker claims. A `report` that is a link is unlinked, and
   what it points to is not touched. `init` a new directory instead to keep the old run.
3. **A broad `fit run` overwrites per-lane refit intent.** The store is last-write-wins and
   `fit run` seeds every lane in its selection. Measured: after a `refit-lane --lane 0
   --slice 200 1200`, a plain `fit run --lane 0 --lane 1` reset lane 0's stored slice back to
   the default. Refit stragglers *after* the broad pass, not before.
4. **Superseded cache files are never collected.** There is no `clean` command. Two knob changes
   on one lane left three fit files for it; a `--force` re-`init` left 7 of the folder's 8 cache
   files unreachable (limitation 2). A new gelpy version supersedes every stage at once: the first fit
   under it wrote its own image, profile and fit files next to the old ones. Deleting `cache/`
   (and `manifest.json` with it) is the only cleanup, and it is safe by definition.
