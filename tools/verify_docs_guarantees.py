#!/usr/bin/env python3
"""Prove the documentation guarantees actually fail when they should.

A verification mechanism nobody has watched fail is a claim, not a mechanism. gelpy's
documentation makes four structural promises -- generated reference cannot drift, documented
commands must run, stated facts are asserted, narrative is flagged when its subject moves --
and each is only worth as much as its failure mode.

So this script breaks each promise on purpose, checks that the named test goes red, and puts
everything back. Run it after changing anything under ``tools/`` or ``tests/docs/``:

    python tools/verify_docs_guarantees.py
    python tools/verify_docs_guarantees.py --keep-going   # report all, don't stop at the first

It edits tracked files and restores them from their pre-run contents in a ``finally`` block.
It refuses to start if the working tree is dirty, so a crash can never be confused with your
own uncommitted work.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


@dataclass
class Probe:
    """One deliberate breakage and the test that must notice it."""

    name: str
    why: str
    file: Path
    find: str
    replace: str
    expect_failures: list[str]


PROBES = [
    Probe(
        name="changed default",
        why="A changed default is the most-quoted kind of fact; the claim test must catch it.",
        file=REPO / "src/gelpy/schemas/baseline.py",
        find='    poly_order: int = Field(\n        default=2,',
        replace='    poly_order: int = Field(\n        default=3,',
        expect_failures=[
            "tests/docs/test_claims.py::test_claim_default_baseline_is_poly_order_2",
            "tests/docs/test_freshness.py::test_page_is_current_with_the_code_it_describes[fit-engine.md]",
        ],
    ),
    Probe(
        name="changed Field description",
        why=(
            "Field descriptions are prose the runtime prints, so the guide may depend on them. "
            "The freshness digest must move even though no signature or default changed."
        ),
        file=REPO / "src/gelpy/schemas/baseline.py",
        find='description="Baseline estimator."',
        replace='description="Baseline estimator (pick one)."',
        expect_failures=[
            "tests/docs/test_freshness.py::test_page_is_current_with_the_code_it_describes[fit-engine.md]",
            "tests/docs/test_reference_is_current.py::test_generated_file_is_current[contracts.md]",
        ],
    ),
    Probe(
        name="removed CLI flag",
        why=(
            "This is the failure that rotted the old documentation: a flag deleted from the CLI "
            "while the docs kept using it. The generated reference must go stale immediately."
        ),
        file=REPO / "src/gelpy/cli/app.py",
        find='    auto_escalate: bool = typer.Option(False, "--auto-escalate", help="Climb tiers on low confidence."),\n    plot: bool = typer.Option(False),\n    allow_raw: bool = typer.Option(\n        False, "--allow-raw",\n        help="Measure the raw image although no background removal is configured "\n        "(refused otherwise: background_required).",\n    ),\n    fmt: str = typer.Option("json", "--format", help="Output format: json|text."),\n):\n    """Run a full fit pass over a lane selection (no manual peak-detection tuning needed)."""',
        replace='    plot: bool = typer.Option(False),\n    allow_raw: bool = typer.Option(\n        False, "--allow-raw",\n        help="Measure the raw image although no background removal is configured "\n        "(refused otherwise: background_required).",\n    ),\n    fmt: str = typer.Option("json", "--format", help="Output format: json|text."),\n):\n    """Run a full fit pass over a lane selection (no manual peak-detection tuning needed)."""\n    auto_escalate = False',
        expect_failures=[
            "tests/docs/test_reference_is_current.py::test_generated_file_is_current[cli.md]",
        ],
    ),
    Probe(
        name="hand-edited generated file",
        why="Generated reference is not maintained, it is regenerated. An edit must be rejected.",
        file=REPO / "docs/reference/cli.md",
        find="# CLI reference",
        replace="# CLI reference\n\nSomeone edited this by hand.",
        expect_failures=[
            "tests/docs/test_reference_is_current.py::test_generated_file_is_current[cli.md]",
        ],
    ),
    Probe(
        name="uncited claim",
        why="A claim test that no page cites is dead weight; the bijection must notice.",
        file=REPO / "tests/docs/test_claims.py",
        find="def test_claim_default_background_model():",
        replace=(
            "def test_claim_a_fact_nobody_states():\n"
            '    """A claim with no citation anywhere in the docs."""\n'
            "    assert True\n\n\n"
            "def test_claim_default_background_model():"
        ),
        expect_failures=[
            "tests/docs/test_claims.py::test_every_claim_is_cited_and_every_citation_is_tested",
        ],
    ),
    Probe(
        name="undocumented module",
        why=(
            "A whole new module must not slip in without anyone deciding where it is explained."
        ),
        file=REPO / "src/gelpy/core/brand_new_subsystem.py",
        find="",  # empty find => create the file
        replace='"""A subsystem nobody documented."""\n\n\ndef do_something() -> None: ...\n',
        expect_failures=[
            "tests/docs/test_freshness.py::test_every_source_module_is_described_by_some_page",
        ],
    ),
]


def run_test(node_id: str) -> tuple[bool, str]:
    """Run one pytest node. Returns (passed, tail of output)."""
    result = subprocess.run(
        [PYTHON, "-m", "pytest", node_id, "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=REPO,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=1800,
    )
    return result.returncode == 0, (result.stdout + result.stderr)[-600:]


def check_clean_tree() -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True, encoding="utf-8"
    )
    if result.stdout.strip():
        print(
            "refusing to run with a dirty working tree -- this script edits tracked files and\n"
            "restores them afterwards, and a crash must not be confusable with your own work.\n\n"
            + result.stdout,
            file=sys.stderr,
        )
        raise SystemExit(2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--keep-going", action="store_true", help="Report every probe.")
    args = parser.parse_args()

    check_clean_tree()

    print(f"Verifying {len(PROBES)} documentation guarantees by breaking each one.\n")
    failures: list[str] = []

    for probe in PROBES:
        print(f"[{probe.name}]")
        print(f"  {probe.why}")
        created = not probe.file.exists()
        original = None if created else probe.file.read_text(encoding="utf-8")
        try:
            if created:
                probe.file.write_text(probe.replace, encoding="utf-8")
            else:
                if probe.find not in original:
                    failures.append(
                        f"{probe.name}: anchor text not found in "
                        f"{probe.file.relative_to(REPO)} -- this script needs updating"
                    )
                    print("  SKIPPED: anchor text not found (the code moved; update the probe)\n")
                    continue
                probe.file.write_text(original.replace(probe.find, probe.replace, 1), encoding="utf-8")

            for node_id in probe.expect_failures:
                passed, tail = run_test(node_id)
                if passed:
                    failures.append(
                        f"{probe.name}: {node_id} PASSED while the guarantee was broken -- "
                        "the mechanism does not work"
                    )
                    print(f"  BROKEN GUARANTEE UNDETECTED: {node_id} still passes")
                else:
                    print(f"  caught by {node_id}")
        finally:
            if created:
                probe.file.unlink(missing_ok=True)
            elif original is not None:
                probe.file.write_text(original, encoding="utf-8")

        # The reference is regenerated from the code, so a code probe leaves it stale.
        subprocess.run([PYTHON, "tools/docs.py", "build"], cwd=REPO, capture_output=True)
        print()

        if failures and not args.keep_going:
            break

    check = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True, encoding="utf-8"
    )
    if check.stdout.strip():
        failures.append(
            "the working tree is dirty after restoring; inspect it:\n" + check.stdout
        )

    if failures:
        print("PROBLEMS:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print(f"All {len(PROBES)} guarantees fail when broken, and the tree is clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
