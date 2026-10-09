# EXAMPLE (gelpy-analyze skill): summary PDF for one gelpy run folder.
# Title, extra figures, regions and observations differ per analysis -- adapt, do not run as-is.
"""Build a dense, factual summary PDF for one gelpy run folder (one gel image or row).

Content, in order: title; image file name, size, dtype, saturation check and the lane table,
all read from the run folder; processing; gelpy's standard renders; line profiles; extra
figures; with --fits the figures, results table and per-lane plots written by
constrained_component_fit.py; with --region a profile-metrics table; observations.

Reads:  RUN_DIR/recipe.json and RUN_DIR/render/*.png
        --figures DIR       extra PNGs, captioned with their file name
        --fits DIR          the --out folder of constrained_component_fit.py
        --observations FILE one observation per line: numbers only, no interpretation
Writes: --out (the PDF)

Needs reportlab (``pip install reportlab``). Lanes are gelpy's 0-based indices.

Usage:
  python report_pdf.py RUN_DIR --title TITLE --out report.pdf
      [--figures DIR] [--fits DIR] [--observations FILE]
      [--signal-range Y0 Y1 --region NAME=Y0-Y1 ...]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
from xml.sax.saxutils import escape

import numpy as np

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                    Spacer, Table, TableStyle)
except ImportError:
    sys.exit("report_pdf.py needs reportlab: pip install reportlab")

W = A4[0] - 30 * mm
ss = getSampleStyleSheet()
H1 = ss["Title"]
H2 = ParagraphStyle("h2", parent=ss["Heading2"], spaceBefore=6, spaceAfter=3)
P = ParagraphStyle("p", parent=ss["Normal"], fontSize=8.5, leading=11)
CAP = ParagraphStyle("cap", parent=P, fontSize=8, leading=10, textColor=colors.HexColor("#333333"))
B = ParagraphStyle("b", parent=P, leftIndent=10, bulletIndent=2)
CELL = ParagraphStyle("cell", parent=P, fontSize=7, leading=8.5)

# gelpy's standard renders (RUN_DIR/render/) and their captions; other figures get their file name.
STANDARD = [
    ("linear.png", "Linear contrast (gelpy image show --kind linear), lane labels on top."),
    ("non_linear.png", "Gamma (non-linear) contrast of the same image; enhances faint features."),
    ("corrected.png", "Background-corrected image, gamma contrast (gelpy image show --kind corrected)."),
    ("lane_detect.png", "Lane-detection QC (gelpy lanes detect): red lines = lane centres; bottom panel = "
                        "column projection of the image with the detected peaks."),
    ("background_diagnostic.png", "Background-removal QC (gelpy background show): fitted background and "
                                  "lane profiles before / after subtraction."),
]
# Written by constrained_component_fit.py into its --out folder.
FIT_FIGURES = ["fit_overview_*.png", "fractions_*.png", "fastest_area.png", "band_migration.png"]

# Observations: numbers only, recomputed from the table; no interpretation.
OBSERVATIONS: list[str] = []


def img(path, width=W, max_h=200 * mm):
    iw, ih = ImageReader(path).getSize()
    w = width
    h = w * ih / iw
    if h > max_h:
        h, w = max_h, max_h * iw / ih
    return Image(path, w, h)


def fig(path, caption, width=W, max_h=200 * mm):
    return KeepTogether([img(path, width, max_h), Paragraph(escape(caption), CAP), Spacer(1, 5 * mm)])


def bullets(items):
    return [Paragraph(t, B, bulletText="-") for t in items]


def table(data, col_w, font=7):
    t = Table(data, colWidths=col_w, repeatRows=1)
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), font),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
        ("ALIGN", (2, 0), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
    ]))
    return t


def cell(text):
    return Paragraph(escape(str(text)), CELL)


def digest(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def span_list(ranges):
    return ", ".join(f"{a}-{b}" for a, b in ranges)


def saturation(rec):
    """Pixels at the integer dtype ceiling of the source image, located by lane."""
    from gelpy.core.image_io import decode_image

    src = rec["source_image"]["path"]
    if not os.path.exists(src):
        return "not checked (source image not found from the current directory)"
    raw, _ = decode_image(src, page=rec["image"].get("page", 0))
    if not np.issubdtype(raw.dtype, np.integer):
        return f"not checked ({raw.dtype} image)"
    top = np.iinfo(raw.dtype).max
    sat = raw >= top if raw.ndim == 2 else (raw >= top).any(axis=-1)
    if not sat.any():
        return f"no pixels at the {raw.dtype} ceiling ({top})"
    ys, xs = np.nonzero(sat)
    lanes = rec["lanes"]["lanes"]
    half = rec["lanes"]["profile_width_px"] / 2
    centers = np.array([lane["x_center_px"] for lane in lanes])
    near = np.abs(xs[:, None] - centers[None, :]).argmin(axis=1)
    inside = np.abs(xs - centers[near]) <= half
    parts = []
    for k in np.unique(near[inside]):
        sel = inside & (near == k)
        parts.append(f"lane {lanes[k]['index']}: {sel.sum()} px at y {ys[sel].min()}-{ys[sel].max()}")
    if (~inside).any():
        parts.append(f"outside the lanes: {(~inside).sum()} px")
    return f"{int(sat.sum())} pixels at the {raw.dtype} ceiling ({top}); " + "; ".join(parts)


def load_profiles(run, lanes):
    """Current full-length profile per lane, from gelpy's profile cache."""
    # Refresh first: manifest.json keeps a stale entry after a recipe edit until the next extract.
    r = subprocess.run([sys.executable, "-m", "gelpy", "profiles", "extract", run, "--lane", "all"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if r.returncode:
        sys.exit(f"`gelpy profiles extract {run} --lane all` failed:\n{r.stderr.strip()}")
    with open(os.path.join(run, "manifest.json")) as f:
        stages = json.load(f)["stages"]
    out = {}
    for i in lanes:
        h = stages[f"profile:lane={i}"]["hash"]
        with open(os.path.join(run, "cache", "profiles", f"lane_{i:03d}__{h}.json")) as f:
            out[i] = np.asarray(json.load(f), dtype=float)
    return out


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="gelpy run folder")
    ap.add_argument("--title", required=True, help="PDF title")
    ap.add_argument("--out", default="report.pdf", help="output PDF (default: report.pdf)")
    ap.add_argument("--figures", metavar="DIR", help="extra PNGs to include (top level of DIR)")
    ap.add_argument("--fits", metavar="DIR", help="the --out folder of constrained_component_fit.py")
    ap.add_argument("--observations", metavar="FILE", help="one observation per line, numbers only")
    ap.add_argument("--signal-range", type=int, nargs=2, metavar=("Y0", "Y1"),
                    help="y-range summed as a lane's total signal (default: the whole profile)")
    ap.add_argument("--region", action="append", default=[], metavar="NAME=Y0-Y1",
                    help="profile-metrics region: share of the total and position of the maximum "
                         "(repeatable; adds the metrics table)")
    return ap.parse_args()


def main():
    args = parse_args()
    run = args.run_dir
    if not os.path.isfile(os.path.join(run, "recipe.json")):
        sys.exit(f"{run}: not a gelpy run folder (no recipe.json)")
    fits_json = os.path.join(args.fits, "results", "component_fits.json") if args.fits else None
    if fits_json and not os.path.isfile(fits_json):
        sys.exit(f"--fits {args.fits}: no results/component_fits.json; write it with "
                 f"constrained_component_fit.py --out {args.fits}")
    if args.figures and not os.path.isdir(args.figures):
        sys.exit(f"--figures {args.figures}: not a folder")
    if args.observations and not os.path.isfile(args.observations):
        sys.exit(f"--observations {args.observations}: no such file")
    regions = []
    for item in args.region:
        try:
            name, rng = item.split("=")
            a, b = rng.split("-")
            regions.append((name, int(a), int(b)))
        except ValueError:
            sys.exit(f"--region {item!r}: expected NAME=Y0-Y1, e.g. fast=600-900")
    with open(os.path.join(run, "recipe.json")) as f:
        rec = json.load(f)
    lanes = rec["lanes"]["lanes"]
    src, meta, bg = rec["source_image"], rec["image"], rec.get("background")
    centers = [lane["x_center_px"] for lane in lanes]
    pitch = float(np.median(np.diff(centers))) if len(centers) > 1 else float("nan")
    fits = None
    if fits_json:
        with open(fits_json) as f:
            fits = json.load(f)
    observations = list(OBSERVATIONS)
    if args.observations:
        with open(args.observations) as f:
            observations += [line.strip() for line in f if line.strip()]

    s = [Paragraph(escape(args.title), H1)]
    page = f", page {meta['page']} of {meta['source_pages']}" if meta.get("source_pages", 1) > 1 else ""
    s.append(Paragraph(
        f"Image: <i>{escape(os.path.basename(src['path']))}</i> ({src['width']} x {src['height']} px, "
        f"{meta.get('original_dtype') or src['dtype']}{page}). Saturation: {escape(saturation(rec))}. "
        f"{len(lanes)} lanes; gelpy run folder <i>{escape(os.path.basename(os.path.normpath(run)))}</i>.", P))
    s.append(Spacer(1, 2 * mm))
    s.append(table([["lane", "label", "x centre px"]]
                   + [[lane["index"], cell(lane["label"]), f"{lane['x_center_px']:.0f}"] for lane in lanes],
                   [12 * mm, 90 * mm, 20 * mm], font=7.5))

    s.append(Paragraph("Processing", H2))
    proc = [f"Lanes: {len(lanes)} lane centres from recipe.json, median pitch {pitch:.1f} px; "
            f"profile width {rec['lanes']['profile_width_px']} px."]
    if bg:
        params = ", ".join(f"{k}={v}" for k, v in sorted(bg.get("params", {}).items()))
        proc.append(f"Background: <i>gelpy background remove --model {escape(bg['model'])}</i>"
                    f"{' (' + escape(params) + ')' if params else ''}, subtracted before any profile.")
    else:
        proc.append("Background: none; profiles come from the raw image.")
    proc.append(f"Profiles: mean over the lane width of the {'background-corrected' if bg else 'raw'} "
                "image, per y pixel.")
    if fits:
        middle = sorted({r["group"] for r in fits if r["middle"]})
        masked = [f"lane {r['lane_index']}: {span_list(r['masked_ranges'])} px" for r in fits if r["masked_ranges"]]
        proc += [
            "Component fit (constrained_component_fit.py, least squares with soft-L1 loss): well = "
            "exponentially modified Gaussian (EMG); smear = generalised Gaussian; "
            + (f"middle band = Gaussian (groups {escape(', '.join(middle))}); " if middle else "")
            + "leading band = EMG with its tail toward the well; fastest peak = bi-Gaussian; "
            "plus a constant offset.",
            f"Fit window per lane: from the zero-crossing after the well-cavity dip "
            f"({min(r['y_start'] for r in fits)}-{max(r['y_start'] for r in fits)} px) to "
            f"{max(r['y_end'] for r in fits)} px.",
            "Masked (dust and hand-confirmed specks): " + ("; ".join(masked) if masked else "none") + ".",
            "Fraction = component area / sum of the lane's component areas; area in corrected-intensity "
            "x px. Leading-band migration = mode of the band component minus the fitted well-peak centre.",
        ]
    if regions:
        sig = args.signal_range
        proc.append(f"Profile metrics: total = summed signal over "
                    f"{'y = %d-%d px' % tuple(sig) if sig else 'the whole profile'}; per region the share of "
                    "the total and the position of the profile maximum.")
    s += bullets(proc)

    shown = set()

    def add(path, caption, **kw):
        h = digest(path)
        if h not in shown:  # skip byte-identical copies (e.g. renders copied into figures/)
            shown.add(h)
            s.append(fig(path, caption, **kw))

    render = os.path.join(run, "render")
    s.append(PageBreak())
    s.append(Paragraph("Gel images", H2))
    for name, caption in STANDARD:
        if os.path.exists(os.path.join(render, name)):
            add(os.path.join(render, name), caption, max_h=118 * mm)
    profiles = sorted(glob.glob(os.path.join(render, "profiles_*.png")))
    if profiles:
        s.append(PageBreak())
        s.append(Paragraph("Line profiles", H2))
        for p in profiles:
            add(p, os.path.basename(p))
    fit_figs = []
    if args.fits:
        for pattern in FIT_FIGURES:
            fit_figs += sorted(glob.glob(os.path.join(args.fits, "figures", pattern)))
    if args.figures:
        skip = shown | {digest(p) for p in fit_figs}
        extra = [p for p in sorted(glob.glob(os.path.join(args.figures, "*.png"))) if digest(p) not in skip]
        if extra:
            s.append(PageBreak())
            s.append(Paragraph("Other figures", H2))
            for p in extra:
                add(p, os.path.basename(p))

    if fits:
        s.append(PageBreak())
        s.append(Paragraph("Component fits", H2))
        for p in fit_figs:
            add(p, os.path.basename(p), width=W * 0.8)
        s.append(PageBreak())
        s.append(Paragraph("Results", H2))
        head = ["lane", "label", "group", "well %", "smear %", "middle %", "band %", "fast %", "fast area",
                "fast px", "band px", "migr. px", "R2", "resid. %"]
        rows = [head]
        for r in fits:
            rows.append([r["lane_index"], cell(r["label"]), cell(r["group"]), f"{r['well_frac'] * 100:.1f}",
                         f"{r['smear_frac'] * 100:.1f}", f"{r['middle_frac'] * 100:.1f}" if r["middle"] else "-",
                         f"{r['band_frac'] * 100:.1f}", f"{r['fastest_frac'] * 100:.1f}",
                         f"{r['fastest_area']:.3g}", f"{r['fastest_center']:.0f}", f"{r['band_center']:.0f}",
                         f"{r['band_migration_px']:.0f}", f"{r['r2']:.3f}", f"{r['residual_frac'] * 100:.1f}"])
        s.append(table(rows, [8 * mm, 30 * mm, 12 * mm] + [11.8 * mm] * 11))
        s.append(Paragraph("fast = fastest peak; px = component position along the lane; resid. = integrated "
                           "|data - model| / integrated signal.", CAP))

    if regions:
        prof = load_profiles(run, [lane["index"] for lane in lanes])
        n = min(len(y) for y in prof.values())
        for flag, (a, b) in [("--signal-range", args.signal_range or (0, n))] + [
                (f"--region {name}", (a, b)) for name, a, b in regions]:
            if not 0 <= a < b <= n:
                sys.exit(f"{flag} {a}-{b}: needs 0 <= Y0 < Y1 <= {n}, the profile length")
        head = ["lane", "label", "total"]
        for name, _, _ in regions:
            head += [f"{name} %", f"{name} px"]
        rows = [head]
        for lane in lanes:
            y = prof[lane["index"]]
            sig = slice(*args.signal_range) if args.signal_range else slice(None)
            tot = float(y[sig].sum())
            row = [lane["index"], cell(lane["label"]), f"{tot:.3g}"]
            for _, a, b in regions:
                row += [f"{100 * y[a:b].sum() / tot:.1f}", f"{a + int(np.argmax(y[a:b]))}"]
            rows.append(row)
        s.append(Paragraph("Profile metrics", H2))
        s.append(table(rows, [10 * mm, 60 * mm, 16 * mm] + [14 * mm] * (2 * len(regions)), font=7.5))

    if observations:
        s.append(Paragraph("Observations", H2))
        s += bullets([escape(o) for o in observations])

    per_lane = (sorted(glob.glob(os.path.join(args.fits, "figures", "fits", "lane_*.png"))) if args.fits
                else sorted(glob.glob(os.path.join(render, "fits", "lane_*.png"))))
    if per_lane:
        s.append(PageBreak())
        s.append(Paragraph("Per-lane fits", H2))
        s.append(Paragraph("Each panel: profile, model and components; the title gives the fit quality.", CAP))
        for p in per_lane:
            s.append(img(p, W * 0.82))

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    doc = SimpleDocTemplate(args.out, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=12 * mm,
                            bottomMargin=12 * mm, title=args.title, author="gelpy")
    doc.build(s)
    print(args.out)


if __name__ == "__main__":
    main()
