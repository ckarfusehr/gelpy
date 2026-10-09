# gelpy documentation

Start here. This page is a router: find the row that matches what you want, read that page.

## If you want to...

| ...then read | what it covers | who it's for |
|---|---|---|
| **analyse a gel, start to finish** | [guide/analyze-a-gel.md](guide/analyze-a-gel.md) | everyone |
| **see it done on a real gel, with the actual numbers** | [walkthrough/](walkthrough/) | everyone |
| **install it and check the install works** | [guide/install.md](guide/install.md) | everyone |
| **drive gelpy from Claude Code (or any agent)** | [guide/driving-from-an-agent.md](guide/driving-from-an-agent.md) | agents, and whoever sets them up |
| **understand the run folder and why re-running is cheap** | [guide/concepts.md](guide/concepts.md) | everyone |
| **read a figure gelpy produced** | [guide/reading-the-figures.md](guide/reading-the-figures.md) | everyone |
| **know what a number in `results.csv` means** | [guide/results.md](guide/results.md) | everyone |
| **understand how bands are fitted, and how much to trust a fit** | [guide/fit-engine.md](guide/fit-engine.md) | anyone reporting numbers |
| **understand how lanes get placed** | [guide/lane-placement.md](guide/lane-placement.md) | anyone whose overlay looks wrong |
| **understand background removal** | [guide/background.md](guide/background.md) | anyone comparing lanes |
| **know what gelpy did to your image before measuring it** | [guide/image-input.md](guide/image-input.md) | anyone with an unusual image |
| **understand line profiles and normalization** | [guide/line-profiles.md](guide/line-profiles.md) | everyone |
| **fix something that went wrong** | [guide/troubleshooting.md](guide/troubleshooting.md) | everyone |
| **know what gelpy is bad at** | [guide/known-limitations.md](guide/known-limitations.md) | read this before publishing numbers |
| **add a peak shape, fit strategy, or background model** | [guide/extending.md](guide/extending.md) | developers |
| **understand how the codebase is laid out** | [guide/architecture.md](guide/architecture.md) | developers |
| **port an analysis from gelpy 0.x** | [guide/migrating-from-0.x.md](guide/migrating-from-0.x.md) | anyone with old notebooks |
| **look up a flag, a contract, or an error code** | [reference/](reference/) | everyone |

## Reference (generated -- do not edit)

These are produced from the code by `python tools/docs.py build`, and a test regenerates and
diffs them, so they cannot describe a flag or field that does not exist:

- [reference/cli.md](reference/cli.md) -- every command and every option
- [reference/contracts.md](reference/contracts.md) -- every request, result and config object
- [reference/registries.md](reference/registries.md) -- the four extension points and their entries
- [reference/errors.md](reference/errors.md) -- the stable error-code catalogue
- [reference/run-folder.md](reference/run-folder.md) -- the on-disk format

At a terminal you can get the same information live, which is always preferable:

```bash
gelpy --help                       # the command tree
gelpy fit run --help               # one command's options
gelpy schema FitRunRequest         # any contract, as JSON Schema
gelpy models list --kind shape     # what's registered, with parameter schemas
```

## How this documentation is kept honest

gelpy's previous documentation was wrong in 38 verified places, including a workflow script
that aborted a third of the way through and a skill that told agents to type two commands
that did not exist. None of it was noticed for two months, because nothing checked. So the
documentation now has to earn its trust:

- **The reference is generated**, not written. `tests/docs/test_reference_is_current.py`
  regenerates it and fails on any difference.
- **Every command block here is executed** against the real example gel, in order, by
  `tests/docs/test_examples_execute.py`. A documented command that stops parsing fails a test.
- **Load-bearing facts are asserted.** Where prose states something specific it cites a claim
  id -- `<!-- claim: default-background-model -->` -- and `tests/docs/test_claims.py` asserts
  it against the code. A bijection test forbids an uncited claim or an untested citation.
- **Narrative pages declare what they describe.** Each carries a digest of the public surface
  of its subject files, so changing a default or a signature fails a test naming the page for
  review. See [guide/extending.md](guide/extending.md#keeping-the-documentation-honest) for
  what to do when that happens.
- **Historical documents say so.** Design specs under [decisions/](decisions/) record what was
  intended at a point in time; several were implemented differently. They are never a
  description of current behaviour.

If you find a statement here that is false, that is a bug of the same severity as a broken
command -- please report it that way.

## Machine-readable index

[docs-index.json](docs-index.json) lists every page with its audience, summary, subject files
and claim ids. An agent can read that one small file to decide what is worth opening, instead
of grepping the whole set.
