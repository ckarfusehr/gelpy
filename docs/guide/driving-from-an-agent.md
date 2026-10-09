---
title: Driving gelpy from an agent
audience: agent
summary: The machine-facing contract -- JSON on stdout, structured error codes on stderr, schema discovery, and the run folder as durable state between sessions.
subject:
  - src/gelpy/core/errors.py
  - src/gelpy/schemas/requests.py
subject_digest: d7b99387ff917d28
---

gelpy was built to be driven by a program. This is not a convenience layer over a "real" Python
API: `runfolder/commands.py` is `f(request) -> result` over pydantic models, and the CLI is a thin
argv adapter over it. What follows is the part of that contract a driver loop depends on.

Set two variables once; every block below uses them.

```bash
export IMAGE="${IMAGE:-$PWD/examples/data/folding_screen.tif}"   # absolute, deliberately
export RUN="${RUN:-/tmp/gelpy-agent-run}"                        # where the analysis lives
```

Three commands give the rest of this page something to talk to. The analysis workflow itself,
including why the lane overlay must be looked at before fitting, is `guide/analyze-a-gel.md`.

```bash
gelpy init "$RUN" "$IMAGE" --n-lanes 20
gelpy lanes detect "$RUN" --n-lanes 20 --apply
gelpy fit run "$RUN" --lane 3
```

## One JSON object per invocation

`--format json` is the default on every command that has the option -- 15 of the 16 leaf
commands; `schema` has no `--format` and always prints JSON. Three exit codes, and they do not
mean the same kind of thing:

- **0** -- success, one JSON object on **stdout**.
- **1** -- domain failure, the structured error on **stderr** (one exception, in the caveats).
- **2** -- argv/usage error, raised by click before gelpy runs: a usage panel on stderr,
  nothing on stdout, never JSON. A misspelled option or a missing argument lands here.

Branch on the exit code and parse only on 1; a 2 means the call itself was malformed.

`results show` is the exception in kind, not in default: it takes `json|csv|table`.

```bash
gelpy results show "$RUN" --format table
gelpy results show "$RUN" --format csv | head -3
```

Two things about the other formats, both measured:

- `--format text` is *not* a human rendering. For a command returning a pydantic model it is
  the same payload on one line. For the four commands that return a plain dict --
  `lanes detect`, `image show`, `background show`, `models list` -- it is a Python `repr`
  (`{'names': [...]}`, single quotes), which is not JSON and will not parse. Use the default.
- `results show --format csv` prints `results.csv` off disk -- an empty string if the file is
  absent. It does not assemble the csv from the fit cache (though it does pay for the assembly
  and then discard it, so a broken run folder still errors). `json` and `table` *are* assembled
  from the cache, with a staleness check. After a recipe edit these disagree -- see the caveats.

## Discovery, so you never have to read source

Three commands are enough to construct any request without opening a file:

```bash
gelpy --help
gelpy fit refit-lane --help
gelpy schema RefitLaneRequest | head -18
gelpy models list --kind shape | head -8
```

`gelpy schema <Model>` prints the JSON Schema of any gelpy contract, and every contract in the
package is reachable that way -- 47 models, all exported from `gelpy.schemas`.
<!-- claim: every-schema-is-discoverable -->
The request models are the tool arguments. `FitRunRequest` is a full fit pass over a lane
selection; `RefitLaneRequest` is one lane plus all-optional overrides merged onto that lane's
last-used settings. Their `Field` descriptions are written to be read by a driver rather than
skimmed by a human, so the schema is the documentation for what a knob does.

If you do not know a model's name, ask for one that does not exist -- the failure lists them all.

```bash
# doc-test: expect-failure
gelpy schema NoSuchModel
```

`gelpy models list --kind fit|shape|background` does the same job for the three registries:
names and descriptions, plus a `schemas` object. Only background models register a config
model, so only their schemas are populated; for `fit` and `shape` every `schemas` entry is
`{}`. A fit strategy's knobs are not per-strategy -- they live in `FitConfig` and the models it
nests (`gelpy schema FitConfig`), and shapes are chosen by name in `FitConfig.shapes`.

## Errors: branch on `code`, never on the message

Every domain error carries a stable machine-readable `code`, and `core/errors.py` holds the
single `to_error_response` mapping that the CLI and any future MCP wrapper share.
<!-- claim: error-codes-are-stable-and-complete -->
Message text is for a human reading a log; treat it as unstable. The `code` and the `available`
list are what you act on.

A real failure -- a lane selector that matches nothing:

```bash
# doc-test: expect-failure
gelpy profiles extract "$RUN" --lane "re:ladder"
```

```console
{
  "error": "Lane selector ['re:ladder'] matched no lanes.",
  "code": "lane_selection",
  "detail": null,
  "available": ["1", "2", "3", ... "20"]
}
```

Exit 1, on stderr, with `available` carrying the full label list -- enough to pick a valid
selector and retry without a second discovery call. A selector that matches nothing always
raises; it never falls back to all lanes.
<!-- claim: lane-selector-matching-nothing-raises -->
The same shape, with the same `available` affordance, comes back as `code: "registry"` when you
name a strategy, shape or background model that is not registered.

`code: "validation_error"` is the one where you need `detail`: it carries the pydantic field
errors as a JSON *string*, so `loads(response["detail"])` gives you
`[{"loc": ["width"], "msg": "Input should be greater than 0", ...}]` and you know which field to
fix. The full code catalogue is generated: [reference/errors.md](../reference/errors.md).

## `fit_config.json` is intent you may rewrite directly

Per-lane fit intent -- the `(profile, fit)` pair last used for each lane -- lives in
`fit_config.json`, sparse, only for lanes that were actually fit. It is deliberately not on the
manifest, so wiping the cache ledger does not lose it, and it is meant to be edited.
`refit-lane` loads the store from disk, merges your explicit overrides onto it, and writes the
merged result back (last write wins). Editing the file and refitting with no overrides at all is
therefore a supported path, and the one to take when you want to change several knobs at once.

```bash
python - "$RUN/fit_config.json" <<"PY"
import json, sys
path = sys.argv[1]
store = json.load(open(path))
lane = store["lanes"]["3"]
lane["fit"]["shapes"] = ["gaussian"]
lane["fit"]["priors"]["max_components"] = 4
json.dump(store, open(path, "w"), indent=2)
PY
gelpy fit refit-lane "$RUN" --lane 3
```

On lane 3 of the example gel that takes the fit from 7 components to 4. On a cold run folder the
summary reports `"recomputed": ["fit:lane=3"], "reused": ["image", "profile:lane=3"]` -- the
profile is not re-extracted and the image not re-decoded, because neither hash moved (a cache
hit is decided by file existence at the content-hash path, so on a folder that has already seen
this edit the fit is reused too and `recomputed` is empty). Keys under `lanes` are lane
indices as JSON strings; the file is validated against `FitConfigStore` on load and a parse
error is raised rather than repaired, because this is precious input state. Every contract
forbids unknown fields, so a misspelled key is a `validation_error`, not a silent no-op.

## The run folder is the state, not the session

There is no daemon and no in-memory context. Everything needed to resume work from last week is
in the run folder, with one caveat: the recipe stores the source image path exactly as it was
handed to `init`, unresolved. Pass a relative path and the run folder only reopens from the
directory you created it in -- any command that has to re-read pixels then fails with
`code: "not_found"`. Pass an absolute path.

`gelpy status` is the cold-orientation call -- pure assembly over `recipe.json`,
`manifest.json`, `fit_config.json` and each lane's cached fit JSON (for `status` and `n_bands`),
recomputing nothing.

```bash
gelpy status "$RUN"
```

Trimmed to the fields this section is about (`gelpy schema RunStatus` is the full surface):

```console
{"run_dir": "...", "source_image": ".../folding_screen.tif", ...,
 "background": null, "polarity": "bright_on_dark", "polarity_confidence": "ok", ...,
 "n_lanes": 20, "n_fit": 1, "n_stale": 0, "n_failed": 0,
 "lanes": [{"index": 3, "label": "4", "has_fit": true, "fit_current": true,
            "status": "ok", "n_bands": 4}, ...]}
```

Read `n_stale` and per-lane `fit_current` before doing anything else. A fit is stale when the
recipe moved underneath it: its cached `fit_hash` no longer matches what the current recipe
would produce. Changing the profile strip width does exactly that.

```bash
gelpy lanes set "$RUN" --width 140
gelpy status "$RUN"
```

`n_fit` drops to 0, `n_stale` becomes 1, and lane 3's `fit_current` flips to `false`. Nothing
was deleted -- the old cache file still sits under its old hash, so setting the width back is a
cache hit, not a recompute.

Every command that changes run-folder *intent* -- recipe, fit config, results -- appends to an
append-only log. The render commands (`image show`, `background show`), `lanes detect` without
`--apply`, `status` and `results show` do not, even though the first three write files under
`render/` or `cache/`. So the log is a record of decisions, not of file writes. This is how you
reconstruct what a previous session did, and what a human changed by hand in between:

```bash
tail -n 2 "$RUN/log/history.jsonl"
```

```console
{"t": 1789040244.05, "command": "refit_lane", "payload": {"run_dir": "...", "lane": 3, ...}}
{"t": 1789040262.33, "command": "set_lanes", "payload": {"n": 20, "width": 140}}
```

Entries are `{t, command, payload}`, with the complete request payload for `fit_run` and
`refit_lane`. It is a log, not a replay mechanism -- nothing reads it back.

## Installing the skill

The skill that encodes the judgment loop -- detect lanes, confirm the overlay, fit, inspect every
lane, refit the stragglers -- is `plugin/skills/gelpy-analyze/SKILL.md`, packaged as a Claude Code
plugin (`plugin/.claude-plugin/plugin.json`). It lives in this repo so it cannot drift from the
code it drives.

Without a marketplace, copy `plugin/skills/gelpy-analyze/` into `~/.claude/skills/` (every
project) or into `.claude/skills/` of the project you analyse gels in; Claude Code picks it up
from there. A copy does not update itself: after pulling a new version, delete the old copy
first and copy again, so files removed upstream do not linger.

To install it through a Claude Code plugin marketplace instead, add one of the two entries in
`plugin/marketplace-entry.json` to that marketplace's `marketplace.json`.

`referenced` points at this repo's `plugin/` subdirectory: nothing is copied, and every push to
`main` is picked up on the next plugin update. Use it for a marketplace added with
`/plugin marketplace add <owner>/<repo>`.

`vendored` is for a marketplace that lists its plugins by a relative `./` path, so the plugin
has to be copied into the marketplace repository. That is one command plus a CI-usable check.
Both need a marketplace checkout, which is why these are not executed here:

```bash
# doc-test: skip
python tools/sync_plugin.py --to /path/to/marketplace-checkout
python tools/sync_plugin.py --to /path/to/marketplace-checkout --check
```

The copy is idempotent and the `marketplace.json` entry is upserted, never duplicated. Re-run it
after any change under `plugin/`; until then the marketplace serves the previous copy.

## Where an agent can actually run this

gelpy is a local CLI that reads local image files. What it needs is a shell on the machine
holding the gel, with `gelpy` on `PATH`; that is the configuration it was developed in
(**Claude Code** in a terminal). Every runnable command block in `docs/guide` and
`docs/walkthrough` is executed against the real example gel by
`tests/docs/test_examples_execute.py` -- as headless `bash -c` subprocesses rather than an agent
session, and blocks marked `# doc-test: skip` are excluded.

Anything without a local shell on that machine is untested by this project; do not assume it
works.

## What a good prompt looks like

The failure mode is not a wrong flag. It is a plausible number from a lane nobody looked at. So
supply what gelpy cannot derive, and ask for the verification step explicitly:

- **The true lane count.** Ground truth from the plate sheet, never inferred. Detection anchors
  *that many* centers to the image; it does not count lanes for you.
- **The lane labels, in order.** Without them every row is labelled `1..N` and you match samples
  to columns by hand later.
- **What the answer is for.** "Purity of the folded band" and "yield across the MgCl2 series"
  lead to different lane selections and different scrutiny.

Works: *"Analyse gel1.tif -- 20 lanes, labels in plate.csv column B. I want the folded-band area
fraction per lane. Detect the lanes, show me the overlay before fitting, then fit and show me
every lane whose confidence is low."* The overlay checkpoint is the load-bearing part, and
`status: "ok"` on a fit means it converged, not that it is right.

Fails: *"quantify the bands in gel1.tif"* -- no lane count, no labels, no question, so lane
placement is a guess and nothing gets verified.

## Caveats

- **A non-domain exception is not structured.** The CLI's error decorator catches `GelpyError`,
  `FileNotFoundError` and `ValidationError`. Anything else escapes it and prints a formatted
  Python traceback to stderr with exit 1 -- verified by pointing `init` at a text file, which
  raises `OSError` from the image decoder. `to_error_response` would map that to
  `code: "internal_error"`, but the CLI never reaches it for a bare exception. So exit 1 does
  not guarantee JSON on stderr: attempt the parse, and on failure treat stderr as an opaque
  crash.
- **A selector term that matches nothing is an error only if *every* term does.** The emptiness
  check is on the combined match set, so `--lane all --lane 3` resolves to lane 3 alone: the
  literal `all` matched no label and was dropped silently (`all` is special-cased only when it
  is the sole term). Compare the resolved lane list against what you intended.
- **`results show --format csv` can be stale while the other formats are correct.**
  `results.csv` is written only by `fit_run` and `refit_lane`. After the `lanes set --width 140`
  above, the file on disk still holds the four old band rows, while `--format json` returns
  `{"lanes": []}` and `--format table` prints nothing, both having dropped the fit as stale.
  Prefer `--format json`, and check `status` first.
