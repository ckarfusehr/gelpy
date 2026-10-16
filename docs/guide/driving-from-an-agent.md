---
title: Driving gelpy from an agent
audience: agent
summary: The machine-facing contract -- JSON on stdout, structured error codes on stderr, schema discovery, and the run folder as durable state between sessions.
subject:
  - src/gelpy/core/errors.py
  - src/gelpy/schemas/requests.py
  - src/gelpy/cli/tools.py
  - src/gelpy/cli/mcp_server.py
  - src/gelpy/schemas/tools.py
subject_digest: fd8c3b6aaec9160d
---

gelpy was built to be driven by a program. This is not a convenience layer over a "real" Python
API: `runfolder/commands.py` is `f(request) -> result` over pydantic models, `cli/tools.py` pairs
each request with the rendering its CLI step does, and the CLI is a thin argv adapter over those
tools. What follows is the part of that contract a driver loop depends on.

Set two variables once; every block below uses them.

```bash
export IMAGE="${IMAGE:-$PWD/examples/data/example_gel_1.tif}"   # the source gel
export RUN="${RUN:-/tmp/gelpy-agent-run}"                        # where the analysis lives
```

Four commands give the rest of this page something to talk to. The analysis workflow itself,
including why the lane overlay must be looked at before fitting, is `guide/analyze-a-gel.md`.
`fit run` refuses a run that has no background removal (`code: "background_required"`), which is
why `background remove` is the third of them; `--allow-raw` measures the raw image on purpose.

```bash
gelpy init "$RUN" "$IMAGE" --n-lanes 20
gelpy lanes detect "$RUN" --n-lanes 20 --apply
gelpy background remove "$RUN"
gelpy fit run "$RUN" --lane 3
```

## One JSON object per invocation

`--format json` is the default on every command that has the option -- 16 of the 20 leaf
commands. `schema` and `tool call` have no `--format` and always print JSON; `tool list` prints
one line per tool unless you pass `--json`; `mcp` prints no result at all, because its stdout
carries the protocol (see "The same tools over MCP", below). Three exit codes carry a result, and
they do not mean the same kind of thing:

- **0** -- success, one JSON object on **stdout**.
- **1** -- the call failed: one `ErrorResponse` on **stderr**, nothing on stdout.
- **2** -- the command line is malformed (a misspelled option or command, a missing argument):
  one `ErrorResponse` with `code: "usage_error"` on **stderr**, `available` listing the valid
  options or subcommands.

Parse stderr on any non-zero exit but 130; a 2 means fix the call, not the run folder. The one silent
exit 1 is a reader that closes the pipe before gelpy has written everything (`gelpy ... | head`):
a reader that leaves early is not a failure of the command, so gelpy stops with exit 1 and prints
nothing. The other silent exit is **130**, the shell's code for a run ended by Ctrl-C (SIGINT): no
error object, and for `gelpy mcp` it is how the server stops.

```bash
# doc-test: expect-failure
gelpy fit run
```

```console
{
  "error": "Missing argument 'RUN_DIR'.",
  "code": "usage_error",
  "detail": "Run `gelpy fit run --help` to see its arguments.",
  "available": [
    "--allow-raw",
    "--auto-escalate",
    "--baseline",
    "--format",
    "--lane",
    "--method",
    "--no-plot",
    "--normalization",
    "--plot",
    "--poly-order",
    "--seed",
    "--shapes",
    "--slice"
  ]
}
```

`results show` is the exception in kind, not in default: it takes `json|csv|table`.

```bash
gelpy results show "$RUN" --format table
gelpy results show "$RUN" --format csv | head -3
```

Two things about the other formats, both measured:

- `--format text` is *not* a human rendering. For a command returning a pydantic model it is
  the same payload on one line of JSON, `lanes detect` and `image show` included. Only
  `background show` and `models list`, the two commands that return a plain dict, print a Python
  `repr` with it (`{'names': [...]}`, single quotes), which is not JSON and will not parse. Use
  the default.
- All three `results show` formats render the same assembly from the fit cache, with a
  staleness check, so they agree with each other: after a recipe edit that makes a fit stale,
  `json` returns an empty `lanes` list, `table` prints nothing and `csv` prints the header row
  alone. Only the **file** `results.csv` can be out of date -- see the caveats.

## Discovery, so you never have to read source

Three commands are enough to construct any request without opening a file:

```bash
gelpy --help
gelpy fit refit-lane --help
gelpy schema RefitLaneRequest | head -18
gelpy models list --kind shape | head -8
```

`gelpy schema <Model>` prints the JSON Schema of any gelpy contract, and every contract in the
package is reachable that way -- every model exported from `gelpy.schemas`.
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

## Tools: one JSON object in, one out

Every workflow step is also a tool. `gelpy tool list` names them; `gelpy tool list --json`
describes each one -- its input and output JSON Schema (registry names as enums), MCP-style
annotations (`read_only_hint`, `destructive_hint`, `idempotent_hint`) and which arguments are
paths (`run`, `input`, `output`). `gelpy tool call NAME` reads one JSON object on stdin (or
`--input FILE`; UTF-8, a byte order mark is ignored) and writes one: the object the matching
command prints. The commands run the
same handlers, so the two cannot disagree, and a failure is the same `ErrorResponse`. Tool names
are public API and change only with a major version.

```bash
gelpy tool list
echo "{\"run_dir\": \"$RUN\"}" | gelpy tool call status | head -5
```

```console
init               Create a run folder
status             Read a run folder's state
image_set          Re-interpret the source image
contrast_set       Set the display contrast
lanes_detect       Detect lane centres
lanes_set          Set lane positions, labels or width
background_remove  Configure background removal
image_render       Render a gel view
profiles_extract   Extract lane profiles
fit_run            Fit bands in a lane selection
fit_refit_lane     Refit one lane
results_get        Read the results
report_html        Write the HTML report
{
  "run_dir": "...",
  "source_image": ".../example_gel_1.tif",
  "image_width": 4688,
  "image_height": 1364,
```

`report_html` is the one tool with a free destination: `out_dir` can be any folder, so it writes only
into a folder that is new, empty or holds an earlier report (it carries `.gelpy-report.json`), and
refuses any other non-empty one with `invalid_value`; the run folder, the folders inside it (but
its own `report/`) and the folders around it are refused whatever they carry. It replaces only
the files that earlier report wrote, which is why it is not marked destructive. Its `index` is relative to the run folder
when the report is inside it and absolute otherwise, so it resolves from any working directory.

Not every command is a tool. `lanes pick` needs a display (without one it answers `display_required`). `background show` has no tool of its
own: `background_remove` with `plot` renders the same figure. `schema` and `models list` have
none either: the tool schemas carry the registry names, while a model's parameters and
descriptions come from `gelpy models list` and `gelpy schema`.

## The same tools over MCP

`gelpy mcp` serves the tool layer over the Model Context Protocol on stdin/stdout, for an MCP
client on the same machine. It needs the optional dependency:

```bash
# doc-test: skip
pip install "gelpy[mcp]"
claude mcp add gelpy -- gelpy mcp
```

The documentation tests skip that block because it changes a client's configuration. Without
the extra, `gelpy mcp` fails with `code: "missing_dependency"` and names the extra to install.

The server offers exactly the tools of `gelpy tool list`, with the same schemas and annotations.
A call answers with the JSON the CLI prints, as text and as structured content; a failure is
the same `ErrorResponse`, flagged as an error result and carried as its text only.

Within one server, calls run one at a time, first come first served, and that includes the
read-only tools: a `status` sent during a `fit_run` is answered when the fit has finished. A
call waits for its turn in the event loop, so the protocol keeps answering however many calls
are waiting, and a call that is cancelled while it waits never runs. One that is cancelled while
it runs is not stopped, because a running tool cannot be interrupted: it finishes, and the next
call starts after it. Two calls of a server never interleave their writes or their renders.
That order holds for one server only. Another process is not queued: `gelpy status` in a shell
answers during a call, because a reader never waits for the run lock, and a process that writes
the same run folder still meets that lock (`run_locked`, below).

The server keeps nothing between calls. It resolves a relative path against the directory it was
started in, which is the client's choice, so pass absolute paths for `run_dir` and `image_path`.
Most of the tests drive the server through the SDK's in-memory transport; the ones that start
it as a process talk to it over stdio. None of them runs inside an agent.

## Errors: branch on `code`, never on the message

Every failure is one `ErrorResponse`. Every domain error carries a stable machine-readable
`code`, and `core/errors.py` holds the single `to_error_response` mapping that the CLI,
`gelpy tool call` and the MCP server share. Exceptions that are not domain errors -- a bare
`ValueError`, a `PermissionError`, an unexpected exception -- map to codes of their own
(`invalid_value`, `os_error`, `internal_error`), so nothing leaves as a traceback.
<!-- claim: error-codes-are-stable-and-complete -->
Message text is for a human reading a log; treat it as unstable. The `code` and the `available`
list are what you act on, and `detail` names the next step -- what to run or change -- for every
code except `validation_error`.

A real failure -- a lane selector that matches nothing:

```bash
# doc-test: expect-failure
gelpy profiles extract "$RUN" --lane "re:ladder"
```

```console
{
  "error": "Lane selector ['re:ladder'] matched no lanes.",
  "code": "lane_selection",
  "detail": "Select lanes by a label from `available`, a 0-based index, or `re:<pattern>`; `gelpy status RUN` lists the lanes.",
  "available": [
    "1",
    "2",
    "3",
    ...
    "20"
  ]
}
```

Exit 1, on stderr, with `available` carrying the full label list -- enough to pick a valid
selector and retry without a second discovery call. A selector that matches nothing always
raises; it never falls back to all lanes.
<!-- claim: lane-selector-matching-nothing-raises -->
The same shape, with the same `available` affordance, comes back as `code: "registry"` when you
name a strategy, shape or background model that is not registered.

`code: "validation_error"` is the one whose `detail` is data rather than a sentence: it carries
the pydantic field errors as a JSON *string*, so `loads(response["detail"])` gives you
`[{"loc": ["width"], "msg": "Input should be greater than 0", ...}]` and you know which field to
fix; its `error` names the first failing field, or just the model when the JSON itself is invalid
(a truncated `recipe.json`). When that first field takes one of a fixed set of values -- an image
kind, a normalization -- `available` lists them, so a bad value is as recoverable as an unknown
registry name:

```bash
# doc-test: expect-failure
echo "{\"run_dir\": \"$RUN\", \"kind\": \"sepia\"}" | gelpy tool call image_render
```

```console
{
  "error": "RenderImageRequest: kind: Input should be 'raw', 'linear', 'non_linear' or 'corrected'. `gelpy schema RenderImageRequest` prints the contract.",
  "code": "validation_error",
  "detail": "[{\"type\": \"literal_error\", \"loc\": [\"kind\"], \"msg\": \"Input should be 'raw', 'linear', 'non_linear' or 'corrected'\", \"input\": \"sepia\", \"ctx\": {\"expected\": \"'raw', 'linear', 'non_linear' or 'corrected'\"}}]",
  "available": [
    "raw",
    "linear",
    "non_linear",
    "corrected"
  ]
}
```

No request takes a number that is not finite. `1e999`, `NaN` and `-Infinity` are refused as
`validation_error` with `type: "finite_number"` before anything is written: JSON cannot hold them,
so stored in `recipe.json` they would read back as `null` and the run folder would not load.

The field list echoes each error's `input`, cut at 200 characters: a megabyte sent where a name
belongs, or a whole damaged file, does not come back. A cut input ends in
`... [cut: 1000000 characters in all]`, and every other part of the error list stays as it was.
The same cut applies to a field name in `loc` (a megabyte-long unknown key) and to the run
folder, tool name, registry name, lane selector or file name that an error message quotes.
The full code catalogue, each code with its next step, is generated:
[reference/errors.md](../reference/errors.md).

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
summary reports `"recomputed": ["fit:lane=3"]` and
`"reused": ["background", "image", "profile:lane=3"]` -- the profile is not re-extracted, the
background not re-fitted and the image not re-decoded, because none of their hashes moved (a
cache hit is decided by file existence at the content-hash path, so on a folder that has already
seen this edit the fit is reused too and `recomputed` is empty). Keys under `lanes` are lane
indices as JSON strings; the file is validated against `FitConfigStore` on load and a parse
error is raised rather than repaired, because this is precious input state. Every contract
forbids unknown fields, so a misspelled key is a `validation_error`, not a silent no-op.

## The run folder is the state, not the session

There is no in-memory context: `gelpy mcp` runs as long as its client does, but every call starts
from the run folder. Everything needed to resume work from last week is in the run folder, and it
does not depend on the directory a command runs from. Paths are recorded relative to the run
folder where possible -- the source image in `recipe.json` (`source_image.path`) and every
`cache_path` in `manifest.json` -- and resolved against it. So an agent may change directory
between calls, and a run folder can be moved together with its image. A relative record is kept
only if it opens the image from the run folder as the operating
system follows it, so a run folder below a symlinked directory works too. When the image shares no
directory with the run folder below the filesystem root (a run under `/tmp`, an image under your
home directory), its absolute path is recorded instead. Run folders written by 1.0 recorded both
as given on the command line; they still open from the directory they were made in.

`gelpy status` is the cold-orientation call -- pure assembly over `recipe.json`,
`manifest.json`, `fit_config.json` and the cached fit JSON of each lane whose fit is current (for
`status` and `n_bands`), recomputing nothing. A current fit whose file is truncated or does not
validate does not make it fail: that fit counts as missing, and `results show` leaves it out the
same way.

```bash
gelpy status "$RUN"
```

Trimmed to the fields this section is about (`gelpy schema RunStatus` is the full surface):

```console
{"run_dir": "...", "source_image": ".../example_gel_1.tif", ...,
 "background": "robust_plane", "polarity": "bright_on_dark", "polarity_confidence": "ok", ...,
 "n_lanes": 20, "n_fit": 1, "n_stale": 0, "n_failed": 0, "stale_stages": [],
 "lanes": [{"index": 3, "label": "4", "has_fit": true, "fit_current": true,
            "status": "ok", "n_bands": 4}, ...]}
```

Read `n_stale`, `stale_stages` and per-lane `fit_current` before doing anything else. `report` says
whether the HTML report in the run's own `report/` is out of date: it is `{path, current}`, or null
when there is none, and `current` is false after a relabel, a refit or any other change that the
page does not show yet (see [html-report.md](html-report.md)). A fit is
stale when the recipe or the gelpy version moved underneath it: its cached `fit_hash` no longer
matches what the current recipe would produce under the running version. Changing the profile
strip width does exactly that, and so does installing a new gelpy. `stale_stages` lists what the
next command that needs them would recompute: the stages the run folder has recorded (`image`,
`background`, `profile:lane=3`, `fit:lane=3`, ...), for lanes the recipe still has, whose cache
artifact for the current recipe and the running gelpy version does not exist. So it shows how far
upstream a change reached, and it follows the cache, not the recorded hashes: undo an edit and
the older artifacts are current again, so nothing is listed. `n_stale` is the lane-level count and
follows the same rule: a lane is current when the fit the recipe calls for is in the cache, so after
an undone edit it is current at once, with no fit command. A fit file that is missing or damaged
is counted in `n_stale` and listed in `stale_stages`. A deleted upstream artifact (the image, the
background or a profile) stays listed until a command that needs it runs: a fit that is cached
pulls on nothing upstream, so `fit run` leaves it listed, while a render or `profiles extract`
computes it again. Lane stages come sorted by kind, then by lane number.

```bash
gelpy lanes set "$RUN" --width 140
gelpy status "$RUN"
```

`n_fit` drops to 0, `n_stale` becomes 1, and lane 3's `fit_current` flips to `false`; its
`status` and `n_bands` fall back to `null` and 0, because `status` does not read the cache file
of a fit it no longer trusts. `stale_stages` becomes `["fit:lane=3", "profile:lane=3"]`: the
width is part of the profile hash and the profile hash feeds the fit hash, while `image` and
`background` did not move. Nothing was deleted -- the old cache file still sits under its old
hash, so setting the width back is a cache hit, not a recompute. Run
`gelpy lanes set "$RUN" --width 159` (the width `lanes detect` measured) and `status` reads
`n_fit: 1` and an empty `stale_stages` again.

Every command that changes run-folder *intent* -- recipe, fit config, results -- appends to an
append-only log. The render commands (`image show`, `background show`, `report html`),
`lanes detect` without `--apply`, `status` and `results show` do not, even though the first four
write files under `render/`, `cache/` or `report/`. So the log is a record of decisions, not of
file writes. This is how you reconstruct what a previous session did, and what a human changed by
hand in between:

```bash
tail -n 2 "$RUN/log/history.jsonl"
```

```console
{"t": 1791998251.2242408, "command": "refit_lane", "payload": {"lane": 3, "fit_overrides": {"method": null, "baseline": null, "shapes": null, "priors": null, "robust_loss": null, "seed": null}, "profile_overrides": {"normalization": null, "slice_start": null, "slice_end": null}, "auto_escalate": false, "plot": false, "allow_raw": false}, "gelpy_version": "1.1.0"}
{"t": 1791998253.487885, "command": "set_lanes", "payload": {"n": 20, "width": 140}, "gelpy_version": "1.1.0"}
```

Entries are `{t, command, payload, gelpy_version}`, with the request payload for `fit_run` and
`refit_lane` in full except `run_dir`: the log lives in the run folder, so repeating where that
folder was would only go stale when it moves. `gelpy_version` is the version that wrote the entry,
so a log that spans an upgrade shows where it happened. Files written by gelpy 1.0 have no
`gelpy_version` -- not their history lines, not `manifest.json`, not `results.json` -- and their
`fit_run` and `refit_lane` payloads still carry `run_dir`. Read both fields with `.get`. It is a
log, not a replay mechanism -- nothing reads it back.

One writer at a time. Every command that writes a run folder holds an advisory lock on it
(`.gelpy.lock`) from its first write to its last, its report and plots included: `fit run`
takes one lock for the fit, the report and the fit plots. A second command that writes the same
folder -- a retried slow call, two calls in flight -- does not wait: it fails at once with
`code: "run_locked"`, and the first command is unaffected. Repeat the call when the first has
finished. `status` and `results show` only read, so they never wait for the lock and never fail
on it. Both statements are about separate processes: the MCP server queues the calls it receives
and runs them one at a time, reads included, so two calls in flight through one server never
meet the lock, and a `status` through it waits for the call before it. The operating system
releases the lock when its holder exits, even after a crash, so
there is no stale lock file to delete. `init` is a writer too, and it refuses a folder that
already holds a run (`code: "run_exists"`) unless you pass `--force`, which replaces the recipe
and the fit configuration and removes the results, the reports and the renders; the cache and the
history log are kept. Of an HTML report only the files its marker lists go, and the `warnings` of
the summary say what was removed and what was left.

## Installing the skill

The skill that encodes the judgment loop -- detect lanes, confirm the overlay, fit, inspect every
lane, refit the stragglers -- is `plugin/skills/gelpy-analyze/`: `judgment.md` holds the checks
and `SKILL.md` the commands and tool names. It is a Claude Code plugin
(`plugin/.claude-plugin/plugin.json`) in this repo, so it cannot drift from the code it drives.

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

- **A selector term that matches nothing is an error only if *every* term does.** The emptiness
  check is on the combined match set, so `--lane all --lane 3` resolves to lane 3 alone: the
  literal `all` matched no label and was dropped silently (`all` is special-cased only when it
  is the sole term). Compare the resolved lane list against what you intended.
- **The file `results.csv` can be stale; every `results show` format is assembled live.**
  `results.csv` is written only by `fit_run` and `refit_lane`. After the `lanes set --width 140`
  above, the file on disk still holds the four old band rows, while `results show` has dropped
  the fit as stale in every format: `--format json` returns an empty `lanes` list,
  `--format table` prints nothing and `--format csv` prints the header row alone. Read results
  through `results show`, and check `status` first.
