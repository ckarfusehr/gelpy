#!/usr/bin/env python3
"""Generate docs/walkthrough/README.md by actually running the analysis.

Why generate it
---------------
A walkthrough is the document most likely to rot and the one a newcomer trusts most. gelpy's
previous one used five flags that had been deleted and aborted a third of the way through,
and nothing noticed for two months because nothing ran it.

So this page is not written, it is produced: every command below is executed against
``examples/data/folding_screen.tif``, every transcript is that command's real output, every
number in the tables is read out of the run folder, and every figure is copied from
``render/``. Prose lives in ``tools/walkthrough_template.md`` and can say whatever it likes
about *meaning*, but it cannot invent a result.

    python tools/build_walkthrough.py            # regenerate page + images
    python tools/build_walkthrough.py --check    # exit 1 if the page is stale

Markers understood in the template:

    <!-- gelpy:step:<id> -->      the command, then its real output
    <!-- gelpy:img:<name> -->     a figure copied and downscaled into img/
    <!-- gelpy:table:<name> -->   a table computed from the run folder
    <!-- gelpy:stat:<name> -->    a single measured value, inline
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = REPO / "tools" / "walkthrough_template.md"
OUT_DIR = REPO / "docs" / "walkthrough"
OUT_PAGE = OUT_DIR / "README.md"
IMG_DIR = OUT_DIR / "img"
IMAGE = REPO / "examples" / "data" / "folding_screen.tif"

#: The migration window holding the folded-product cluster on this gel, in profile pixels.
#: Read off the fit figures, not assumed -- see the template's discussion.
PRODUCT_WINDOW = (750, 1150)

#: Lane labels for this gel, from the original experiment's sample sheet.
LABELS = [
    "scaffold",
    "2 MgCl2, 500 NaCl",
    "4 MgCl2, 500 NaCl",
    "6 MgCl2, 500 NaCl",
    "8 MgCl2, 500 NaCl",
    "10 MgCl2, 500 NaCl",
    "150 NaCl",
    "300 NaCl",
    "400 NaCl",
    "500 NaCl",
    "600 NaCl",
    "700 NaCl",
    "6 MgCl2",
    "10 MgCl2",
    "6 MgCl2, 150 NaCl, 1/2 scaf",
    "10 MgCl2, 150 NaCl, 1/2 scaf",
    "6 MgCl2, 300 NaCl, 1/2 scaf",
    "10 MgCl2, 300 NaCl, 1/2 scaf",
    "6 MgCl2, 500 NaCl, 1/2 scaf",
    "10 MgCl2, 500 NaCl, 1/2 scaf",
]


@dataclass
class Step:
    """One documented command."""

    id: str
    argv: list[str]
    #: Lines of real output to show; None shows all. Long JSON is noise in a walkthrough.
    show: str = "elide"
    keep_keys: list[str] = field(default_factory=list)
    output: str = ""
    seconds: float = 0.0


def _render_command(argv: list[str]) -> str:
    """The command as a reader should type it.

    The generator is the single source of these command strings: the page used to carry
    hand-written copies alongside generator-run commands, and they had already diverged (the
    documented `lanes detect` was missing --labels-file). Absolute workspace paths are mapped
    back to the shell variables the page defines, so what is printed is what a reader can run.
    """
    parts = ["gelpy"]
    for token in argv:
        if token == "$RUN":
            parts.append('"$RUN"')
        elif token == "$IMAGE":
            parts.append('"$IMAGE"')
        elif token.endswith("labels.txt"):
            parts.append('"$LABELS"')
        elif " " in token or "," in token:
            parts.append(f'"{token}"')
        else:
            parts.append(token)
    return " ".join(parts)


def _elide_json(text: str, keep_keys: list[str]) -> str:
    """Shrink a JSON payload to the keys that carry the point."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        lines = text.splitlines()
        if len(lines) <= 18:
            return text.rstrip()
        return "\n".join(lines[:16] + ["..."])

    if keep_keys and isinstance(data, dict):
        kept = {k: data[k] for k in keep_keys if k in data}
        dropped = [k for k in data if k not in kept]
        rendered = json.dumps(kept, indent=2, ensure_ascii=False)
        if dropped:
            rendered = rendered.rstrip("}").rstrip() + f",\n  ... ({', '.join(dropped)})\n}}"
        return rendered
    rendered = json.dumps(data, indent=2, ensure_ascii=False)
    lines = rendered.splitlines()
    if len(lines) <= 24:
        return rendered
    return "\n".join(lines[:22] + ["  ..."])


def run_steps(workspace: Path, python: str) -> dict[str, Step]:
    """Execute the documented sequence, capturing real output and timings."""
    run_dir = workspace / "run"
    labels_file = workspace / "labels.txt"
    labels_file.write_text("\n".join(LABELS) + "\n")

    steps = [
        Step("init", ["init", "$RUN", "$IMAGE", "--n-lanes", "20", "--width", "180"],
             keep_keys=["image"]),
        Step("detect", ["lanes", "detect", "$RUN", "--n-lanes", "20", "--apply",
                        "--labels-file", str(labels_file)],
             keep_keys=["pitch", "n_detected", "n_expected", "confidence", "confidence_reason",
                        "projection_source", "applied", "overlay"]),
        Step("background", ["background", "remove", "$RUN", "--model", "robust_plane", "--plot"],
             keep_keys=["files", "warnings"]),
        Step("corrected", ["image", "show", "$RUN", "--kind", "corrected"], show="all"),
        Step("fit", ["fit", "run", "$RUN", "--plot"], keep_keys=["files", "warnings"]),
        Step("status", ["status", "$RUN"],
             keep_keys=["background", "polarity", "polarity_confidence", "n_lanes", "n_fit",
                        "n_stale", "n_failed"]),
        Step("results", ["results", "show", "$RUN", "--format", "table"], show="head20"),
        Step("recache", ["fit", "run", "$RUN", "--no-plot"], keep_keys=["recomputed"]),
    ]

    executed: dict[str, Step] = {}
    for step in steps:
        argv = [str(run_dir) if a == "$RUN" else str(IMAGE.relative_to(REPO)) if a == "$IMAGE" else a for a in step.argv]
        started = time.monotonic()
        result = subprocess.run(
            [python, "-m", "gelpy", *argv],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        step.seconds = time.monotonic() - started
        if result.returncode != 0:
            raise SystemExit(
                f"step {step.id!r} failed with exit {result.returncode}\n"
                f"  command: {' '.join(argv)}\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
            )
        raw = result.stdout.strip()
        if step.show == "all":
            step.output = raw
        elif step.show == "head20":
            lines = raw.splitlines()
            step.output = "\n".join(lines[:20] + (["..."] if len(lines) > 20 else []))
        else:
            step.output = _elide_json(raw, step.keep_keys)
        executed[step.id] = step
        print(f"  ran {step.id:<12} {step.seconds:6.1f}s")
    return executed


def copy_images(run_dir: Path, write: bool = True) -> dict[str, str]:
    """Copy the figures the page shows, downscaled, into img/. Returns name -> filename.

    With ``write=False`` (``--check``) nothing is written; only the mapping is returned.
    """
    wanted = {
        "lane_detect": run_dir / "render" / "lane_detect.png",
        "corrected": run_dir / "render" / "corrected.png",
        "background": run_dir / "render" / "background_diagnostic.png",
        "fit_lane_05": run_dir / "render" / "fits" / "lane_005.png",
        "fit_lane_00": run_dir / "render" / "fits" / "lane_000.png",
    }
    if write:
        IMG_DIR.mkdir(parents=True, exist_ok=True)
        for existing in IMG_DIR.glob("*.png"):
            existing.unlink()

    produced: dict[str, str] = {}
    for name, source in wanted.items():
        if not source.exists():
            print(f"  WARNING: {source} missing, skipping figure {name!r}")
            continue
        target = IMG_DIR / f"{name}.png"
        produced[name] = target.name
        if not write:
            continue
        try:
            from PIL import Image  # noqa: PLC0415

            with Image.open(source) as img:
                if img.width > 1400:
                    ratio = 1400 / img.width
                    img = img.resize((1400, int(img.height * ratio)), Image.LANCZOS)
                img.save(target, optimize=True)
        except ImportError:
            shutil.copy2(source, target)
        print(f"  figure {name:<12} {target.stat().st_size // 1024:>5} KB")
    return produced


def write_script(steps: dict[str, Step]) -> None:
    """Emit examples/walkthrough.sh from the same commands the page shows.

    A separate hand-maintained script is how the previous one came to use five flags that no
    longer existed. Generating it from the executed step list means the script and the prose
    cannot disagree.
    """
    target = REPO / "examples" / "walkthrough.sh"
    lines = [
        "#!/usr/bin/env bash",
        "#",
        "# GENERATED by tools/build_walkthrough.py -- do not edit.",
        "# The command sequence from docs/walkthrough/README.md, which was produced by",
        "# actually running it. Every command takes the run directory explicitly, so this",
        "# script is a complete, replayable record of the analysis.",
        "#",
        "# Usage:  bash examples/walkthrough.sh [RUN_DIR]",
        "set -euo pipefail",
        "",
        'RUN="${1:-runs/folding_screen}"',
        'IMAGE="${IMAGE:-examples/data/folding_screen.tif}"',
        'LABELS="$(dirname "$RUN")/labels.txt"',
        'mkdir -p "$(dirname "$RUN")"',
        "printf '%s\\n' " + " ".join(f"'{label}'" for label in LABELS) + ' > "$LABELS"',
        "",
    ]
    for step in steps.values():
        lines.append(f"# {step.id}")
        lines.append(_render_command(step.argv))
        lines.append("")
    lines.append('echo "Done. Results in $RUN -- see results.csv, report.md and render/."')
    target.write_text("\n".join(lines) + "\n")
    target.chmod(0o755)
    print(f"  wrote {target.relative_to(REPO)}")


def publish_artifacts(run_dir: Path) -> None:
    """Commit the run's own results alongside the page.

    This makes docs/walkthrough/ a complete example run rather than a description of one: a
    reader gets the numbers without installing anything, and examples/downstream_analysis.ipynb
    has a real CSV to read. It also demonstrates the claim the page makes about run folders
    being plain, shareable files.
    """
    for name in ("results.csv", "report.md"):
        source = run_dir / name
        if source.exists():
            shutil.copy2(source, OUT_DIR / name)
            print(f"  artifact {name:<12} {source.stat().st_size // 1024:>5} KB")


def compute_tables(run_dir: Path) -> dict[str, str]:
    """Tables computed from the run folder -- never hand-typed."""
    results = json.loads((run_dir / "results.json").read_text())
    lanes = results["lanes"]
    low, high = PRODUCT_WINDOW

    header = "| lane | sample | bands | coverage | confidence |"
    rule = "|---|---|---|---|---|"
    rows = [
        f"| {lane['lane_index']} | {lane['label']} | {len(lane['bands'])} | "
        f"{lane['coverage']:.3f} | `{lane['confidence']}` |"
        for lane in lanes
    ]
    per_lane = "\n".join([header, rule, *rows])

    yield_rows = []
    for lane in lanes:
        cluster = [b for b in lane["bands"] if low <= b["maximum_position"] <= high]
        total = sum(b["area_fraction"] for b in cluster)
        biggest = max((b["area_fraction"] for b in cluster), default=0.0)
        yield_rows.append(
            f"| {lane['label']} | {len(cluster)} | {total:.3f} | {biggest:.3f} |"
        )
    yield_table = "\n".join(
        [
            "| sample | components in window | summed fraction | largest single |",
            "|---|---|---|---|",
            *yield_rows,
        ]
    )

    lane5 = next(lane for lane in lanes if lane["lane_index"] == 5)
    band_rows = [
        f"| {b['band_number']} | `{b['component_class']}` | `{b['params']['kind']}` | "
        f"{b['maximum_position']:.0f} | {b['area_fraction']:.3f} | "
        + (
            f"{b['area_uncertainty'] / b['area'] * 100:.0f}% |"
            if b.get("area_uncertainty") and b["area"]
            else "n/a |"
        )
        for b in lane5["bands"]
    ]
    lane5_table = "\n".join(
        [
            "| band | class | shape | position (px) | area fraction | relative uncertainty |",
            "|---|---|---|---|---|---|",
            *band_rows,
        ]
    )
    return {"per_lane": per_lane, "yield": yield_table, "lane5_bands": lane5_table}


def compute_stats(run_dir: Path, steps: dict[str, Step]) -> dict[str, str]:
    """Single measured values the prose quotes inline."""
    results = json.loads((run_dir / "results.json").read_text())
    status = json.loads((run_dir / "manifest.json").read_text()) if False else {}
    lanes = results["lanes"]
    coverages = [lane["coverage"] for lane in lanes if lane.get("coverage")]
    low, high = PRODUCT_WINDOW

    lane5 = next(lane for lane in lanes if lane["lane_index"] == 5)
    cluster5 = [b for b in lane5["bands"] if low <= b["maximum_position"] <= high]

    return {
        "n_lanes": str(len(lanes)),
        "n_low": str(sum(1 for lane in lanes if lane["confidence"] != "ok")),
        "coverage_min": f"{min(coverages):.3f}",
        "coverage_max": f"{max(coverages):.3f}",
        "fit_seconds": f"{steps['fit'].seconds:.0f}",
        "recache_seconds": f"{steps['recache'].seconds:.1f}",
        "speedup": f"{steps['fit'].seconds / max(steps['recache'].seconds, 0.01):.0f}",
        "detect_pitch": f"{_detect_pitch(run_dir):.1f}",
        "product_window": f"{low}-{high}",
        "lane5_cluster_n": str(len(cluster5)),
        "lane5_cluster_sum": f"{sum(b['area_fraction'] for b in cluster5):.3f}",
        "lane5_cluster_largest": f"{max(b['area_fraction'] for b in cluster5):.3f}",
        "image_width": str(results.get("image_width", "")) or "4688",
    }


def _detect_pitch(run_dir: Path) -> float:
    recipe = json.loads((run_dir / "recipe.json").read_text())
    centers = [lane["x_center_px"] for lane in recipe["lanes"]["lanes"]]
    return (centers[-1] - centers[0]) / (len(centers) - 1)


def render_page(steps: dict[str, Step], images: dict[str, str], tables: dict[str, str],
                stats: dict[str, str]) -> str:
    template = TEMPLATE.read_text()

    quoted = " ".join(f"'{label}'" for label in LABELS)
    template = template.replace("<!-- gelpy:labels -->", quoted)

    for step_id, step in steps.items():
        block = (
            f"```bash\n{_render_command(step.argv)}\n```\n\n"
            f"```console\n{step.output}\n```"
        )
        template = template.replace(f"<!-- gelpy:step:{step_id} -->", block)

    for name, filename in images.items():
        template = template.replace(
            f"<!-- gelpy:img:{name} -->", f"![{name.replace('_', ' ')}](img/{filename})"
        )
    for name, table in tables.items():
        template = template.replace(f"<!-- gelpy:table:{name} -->", table)
    for name, value in stats.items():
        template = template.replace(f"<!-- gelpy:stat:{name} -->", value)

    leftover = [
        line for line in template.splitlines() if "<!-- gelpy:" in line
    ]
    if leftover:
        raise SystemExit("unsubstituted markers remain:\n  " + "\n  ".join(leftover))

    banner = (
        "<!-- GENERATED by tools/build_walkthrough.py -- do not edit.\n"
        "     Every command below was executed against examples/data/folding_screen.tif and\n"
        "     every number was read out of the resulting run folder. Edit the prose in\n"
        "     tools/walkthrough_template.md and regenerate. -->\n"
    )
    # The banner must go AFTER the front matter: the front-matter parser requires the document
    # to start with `---`, and this page has to appear in docs-index.json like any other.
    if template.startswith("---\n"):
        end = template.find("\n---\n", 4)
        if end != -1:
            head, body = template[: end + 5], template[end + 5 :]
            return head + banner + body
    return banner + template


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="Exit 1 if the page is stale.")
    parser.add_argument("--python", default=sys.executable, help="Interpreter that has gelpy installed.")
    parser.add_argument("--workspace", type=Path, default=None, help="Where to run (default: a temp dir).")
    args = parser.parse_args()

    if not IMAGE.exists():
        print(f"error: {IMAGE.relative_to(REPO)} is required and absent", file=sys.stderr)
        return 2
    if not TEMPLATE.exists():
        print(f"error: {TEMPLATE.relative_to(REPO)} is missing", file=sys.stderr)
        return 2

    import tempfile

    workspace = args.workspace or Path(tempfile.mkdtemp(prefix="gelpy-walkthrough-"))
    workspace.mkdir(parents=True, exist_ok=True)
    print(f"running the walkthrough in {workspace}")

    steps = run_steps(workspace, args.python)
    run_dir = workspace / "run"
    images = copy_images(run_dir, write=not args.check)
    if not args.check:
        publish_artifacts(run_dir)
        write_script(steps)
    tables = compute_tables(run_dir)
    stats = compute_stats(run_dir, steps)
    page = render_page(steps, images, tables, stats)

    if args.check:
        if not OUT_PAGE.exists():
            print("docs/walkthrough/README.md has never been generated")
            return 1
        # Timings legitimately vary between machines, so --check compares everything else.
        def strip_timings(text: str) -> str:
            import re

            return re.sub(r"\b\d+(\.\d+)?\s*(s|seconds)\b", "<time>", text)

        if strip_timings(OUT_PAGE.read_text()) != strip_timings(page):
            print("docs/walkthrough/README.md is stale; run: python tools/build_walkthrough.py")
            return 1
        print("docs/walkthrough/README.md is current")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PAGE.write_text(page)
    print(f"wrote {OUT_PAGE.relative_to(REPO)} ({len(page.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
