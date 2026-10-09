#!/usr/bin/env python3
"""Vendor gelpy's Claude Code plugin into a marketplace repository.

Why this exists
---------------
gelpy's skill lives in this repo (``plugin/``) so it can never drift from the code it
drives. A marketplace that lists its plugins by a relative ``./`` path needs the plugin
*copied into* that marketplace's repository. This script does that copy and keeps the
``marketplace.json`` entry in step, so the whole delivery step is one command instead of a
manual, forgettable ritual.

If the marketplace's users add it themselves (``/plugin marketplace add``), you do not need
this script at all: use the ``referenced`` entry from ``plugin/marketplace-entry.json`` once
and every push reaches everyone automatically.

Usage
-----
    python tools/sync_plugin.py --to /path/to/marketplace-repo
    python tools/sync_plugin.py --to /path/to/marketplace-repo --check   # no writes
    python tools/sync_plugin.py --to /path/to/marketplace-repo --slug gelpy-analyze

``--check`` exits 1 if the marketplace copy is missing or stale, which makes it usable as a
CI gate in either repository.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PLUGIN_DIR = REPO / "plugin"
MANIFEST = PLUGIN_DIR / ".claude-plugin" / "plugin.json"
ENTRY_SNIPPET = PLUGIN_DIR / "marketplace-entry.json"


def _fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(2)


def _load_manifest() -> dict:
    if not MANIFEST.exists():
        _fail(f"{MANIFEST.relative_to(REPO)} not found -- is this a gelpy checkout?")
    return json.loads(MANIFEST.read_text())


#: Never delivered to the marketplace. ``marketplace-entry.json`` is a gelpy-side aid (the
#: paste-me snippet), not part of the plugin -- and it must be excluded from the *comparison*
#: as well as the copy, or ``--check`` reports drift forever on a correctly synced target.
NOT_DELIVERED = ("__pycache__", ".DS_Store", ".pytest_cache", "marketplace-entry.json")


def _tracked_files(root: Path) -> list[Path]:
    """Every delivered file under ``root``. Sorted for stable comparison."""
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and not any(part in NOT_DELIVERED for part in p.relative_to(root).parts)
    )


def _differences(source: Path, target: Path) -> list[str]:
    """Human-readable list of what a sync would change."""
    if not target.exists():
        return [f"missing entirely: {target}"]
    src_rel = {p.relative_to(source) for p in _tracked_files(source)}
    dst_rel = {p.relative_to(target) for p in _tracked_files(target)}
    diffs = [f"only in gelpy: {r}" for r in sorted(src_rel - dst_rel)]
    diffs += [f"only in marketplace (stale): {r}" for r in sorted(dst_rel - src_rel)]
    diffs += [
        f"differs: {r}"
        for r in sorted(src_rel & dst_rel)
        if not filecmp.cmp(source / r, target / r, shallow=False)
    ]
    return diffs


def _marketplace_json(marketplace: Path) -> Path:
    path = marketplace / ".claude-plugin" / "marketplace.json"
    if not path.exists():
        _fail(
            f"{path} not found. Point --to at the root of the marketplace repository "
            "(the directory containing .claude-plugin/marketplace.json)."
        )
    return path


def _upsert_entry(marketplace_json: Path, slug: str, write: bool) -> str:
    """Insert or update this plugin's entry. Returns a description of what changed."""
    data = json.loads(marketplace_json.read_text())
    plugins = data.setdefault("plugins", [])
    manifest = _load_manifest()

    entry = json.loads(ENTRY_SNIPPET.read_text())["vendored"]
    entry = {k: v for k, v in entry.items() if not k.startswith("_")}
    entry["name"] = manifest["name"]
    entry["description"] = manifest["description"]
    entry["source"] = f"./plugins/{slug}"

    for index, existing in enumerate(plugins):
        if existing.get("name") == entry["name"]:
            if existing == entry:
                return "marketplace.json entry already current"
            plugins[index] = entry
            action = "updated the existing marketplace.json entry"
            break
    else:
        plugins.append(entry)
        action = "appended a new marketplace.json entry"

    if write:
        marketplace_json.write_text(json.dumps(data, indent=2) + "\n")
    return action


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--to", required=True, type=Path, help="Marketplace repository checkout.")
    parser.add_argument(
        "--slug",
        default="gelpy",
        help="Directory name under the marketplace's plugins/ (default: gelpy).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report drift and exit 1 if a sync is needed; write nothing.",
    )
    args = parser.parse_args()

    manifest = _load_manifest()
    marketplace = args.to.expanduser().resolve()
    marketplace_json = _marketplace_json(marketplace)
    target = marketplace / "plugins" / args.slug

    diffs = _differences(PLUGIN_DIR, target)

    if args.check:
        entry_state = _upsert_entry(marketplace_json, args.slug, write=False)
        stale = diffs or entry_state != "marketplace.json entry already current"
        if stale:
            print(f"plugin '{manifest['name']}' is OUT OF SYNC with {target}:")
            for diff in diffs:
                print(f"  - {diff}")
            if entry_state != "marketplace.json entry already current":
                print(f"  - marketplace.json: would be {entry_state}")
            print("\nrun without --check to sync.")
            return 1
        print(f"plugin '{manifest['name']}' is in sync with {target}")
        return 0

    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(PLUGIN_DIR, target, ignore=shutil.ignore_patterns(*NOT_DELIVERED))
    entry_state = _upsert_entry(marketplace_json, args.slug, write=True)

    print(f"synced plugin '{manifest['name']}' -> {target}")
    print(f"  {entry_state}")
    for diff in diffs or ["(no file changes)"]:
        print(f"  {diff}")
    print("\nNow commit and push in the marketplace repository.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
