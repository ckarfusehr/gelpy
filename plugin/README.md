# The gelpy plugin for Claude Code

This directory is a self-contained Claude Code plugin with the `gelpy-analyze` skill. It holds
the only `SKILL.md` in this repository; `tests/test_packaging.py` fails on a second one,
because two copies of a skill drift apart.

```
plugin/
  .claude-plugin/plugin.json      the manifest
  skills/gelpy-analyze/SKILL.md   the skill
  skills/gelpy-analyze/examples/  scripts the skill adapts (component fit, PDF report)
  marketplace-entry.json          the snippet to paste into a marketplace (not delivered)
```

## What the skill does

It encodes the judgment loop that decides whether gelpy's numbers mean anything: place lanes
from the image, **verify the overlay**, remove the background, fit, inspect every lane, refit
the stragglers. Most of it is about what not to trust: that lane positions never transfer
between gels, why `status: "ok"` is not a quality signal, and when a low-confidence lane needs
a human. Without the skill, an agent has to guess at these.

## Installing it without a marketplace

Claude Code also loads skills from a skills folder, so a copy of the skill directory is
enough:

```bash
mkdir -p ~/.claude/skills
rm -rf ~/.claude/skills/gelpy-analyze    # remove an older copy first
cp -R plugin/skills/gelpy-analyze ~/.claude/skills/
```

`~/.claude/skills/` makes it available in every project; `.claude/skills/` inside the project
you analyse gels in limits it to that project (not inside this repository: a second
`SKILL.md` here fails `tests/test_packaging.py`). A copy does not update itself. After pulling
a new version, remove the old copy and copy again, as above: `cp -R` over an existing copy
keeps files that the new version no longer has.

## Installing it through a marketplace

There are two arrangements, depending on how the marketplace lists its plugins. gelpy
supports both.

### Vendored (for a marketplace that lists its plugins by a relative path)

A marketplace that lists its plugins by a relative `./` path needs the plugin inside its own
repository. Copy it there:

```bash
python tools/sync_plugin.py --to /path/to/your-marketplace-repo
```

This copies `plugin/` to `<marketplace>/plugins/gelpy/` and inserts or updates the
`marketplace.json` entry. Then commit and push in the marketplace repository. Re-run it after
any change to `plugin/`, otherwise the marketplace's users keep the copy they already have. To
check whether a re-sync is needed (usable as a CI gate in either repository):

```bash
python tools/sync_plugin.py --to /path/to/your-marketplace-repo --check
```

It exits 1 and prints what differs when the vendored copy is stale.

### Referenced (if users add the marketplace themselves)

If users run `/plugin marketplace add <your-marketplace>` themselves, nothing needs copying.
Add the `referenced` entry from `marketplace-entry.json` to the marketplace once, and every
push to this repository's default branch reaches everyone. The entry has no `sha` on purpose:
pinning one freezes the plugin at that commit.

## Why `plugin.json` has no `version`

A plugin with a `version` is re-delivered to users only when that field is bumped. Without
one, the plugin resolves to the commit it was fetched from and tracks the branch, so a fix to
`SKILL.md` reaches every user on the next push, with no release step. Bumping a version for
every skill edit would be extra work for no benefit: the version people need to know is that
of the gelpy *package*, which `pip` reports.

The cost is that `/plugin` shows a commit rather than a semantic version.

## The skill needs the CLI

The plugin gives an agent the *knowledge*. The agent still needs `gelpy` on PATH in whatever
environment it runs commands in; see [../docs/guide/install.md](../docs/guide/install.md).

The verified setup is Claude Code (terminal or desktop app) working in a directory with gelpy
installed and the gel images on disk. I have not tested other agent environments, so do not
assume one works because the skill installed.

## Validating a change

After editing the skill:

```bash
claude plugin validate ./plugin      # if the Claude Code CLI is available
pytest tests/docs/test_plugin.py -q  # layout, manifest, and sync-tool checks
```
