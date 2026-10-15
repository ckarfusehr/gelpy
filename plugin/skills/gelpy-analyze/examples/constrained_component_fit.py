# EXAMPLE (gelpy-analyze skill): constrained per-lane component fit on a gelpy run folder.
# Windows, bounds, lane groups and masks differ for every gel -- adapt, do not run as-is.
"""Constrained per-lane component fit of gelpy's background-corrected line profiles.

For when the user prescribes the peak set and the free-form ``gelpy fit run`` does not honor
it. Components (one each, fixed structure):

  well    : EMG (Gaussian + exponential tail toward +y) right after the well-cavity dip
  smear   : generalized Gaussian exp(-(|y-mu|/alpha)^p) between well and band
  middle  : optional Gaussian between well and band (only for the groups given with --middle)
  band    : EMG with its exponential tail toward the well (slow side) = leading band
  fastest : bi-Gaussian = the fastest-migrating species

plus a constant offset bounded to +/-2 sigma noise. Narrow dust spikes are masked (weight 0).
Fit window per lane: from the zero-crossing after the well-cavity dip to --fit-end. The
well-cavity dip is the below-baseline dip where the well cuts into the gel; it is not signal.
judgment.md, section "A well-cavity dip is not signal", shows how to find it on a profile plot.

y is the position along the profile (image row, px). Read every window off this gel's own
profile plots (``gelpy profiles extract RUN_DIR --lane all --plot``); positions never transfer
between gels. Lanes are gelpy's 0-based indices throughout (CLI, tables, file names).

Reads:  RUN_DIR/recipe.json and the cached background-corrected profiles (refreshed first with
        ``gelpy profiles extract RUN_DIR --lane all``). A run without background removal stops the
        script: gelpy refuses its profiles (``background_required``), and the components are fitted
        on background-corrected profiles only. Run ``gelpy background remove RUN_DIR`` and check
        ``gelpy background show RUN_DIR`` first.
Writes: OUT/figures/fits/lane_NNN.png, OUT/figures/fit_overview_<group>.png,
        OUT/figures/fractions_<group>.png, OUT/figures/fastest_area.png,
        OUT/figures/band_migration.png, OUT/results/component_fits.csv/.json

Usage:
  python constrained_component_fit.py RUN_DIR --out OUT
      --dip-window Y0 Y1 --fit-end Y --band-window Y0 Y1
      --fastest-window Y0 Y1 --noise-window Y0 Y1
      [--group A=0-9 --group B=10-19] [--band-window-for B Y0 Y1]
      [--middle A --middle-window Y0 Y1] [--mask LANE:Y0-Y1] [--bound NAME=LO,HI]
      [--x-from-label REGEX --x-label TEXT --x-scale symlog]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import least_squares
from scipy.special import erfc, gamma
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy < 2 only has trapz

COMPS = ["well", "smear", "middle", "band", "fastest"]
COLORS = {"well": "#009E73", "smear": "#E69F00", "middle": "#CC79A7", "band": "#0072B2", "fastest": "#D55E00"}
NAMES = {"well": "well", "smear": "smear", "middle": "middle band", "band": "leading band",
         "fastest": "fastest peak"}
PARAMS = ["well_A", "well_mu", "well_sigma", "well_tau", "smear_A", "smear_mu", "smear_alpha", "smear_p",
          "middle_A", "middle_mu", "middle_sigma", "band_A", "band_mu", "band_sigma", "band_tau",
          "fastest_A", "fastest_mu", "fastest_sigma_l", "fastest_sigma_r", "offset"]
SQRT2 = np.sqrt(2.0)

# ---- CONFIG. All values are px along the lane, tuned on one gel: placeholders to check against ---
# ---- your own profile plots. Positions come from the CLI windows; bounds: --bound NAME=LO,HI. ---
BOUNDS = {
    "well_sigma": (1.0, 6.0),       # Gaussian core of the well peak
    "well_tau": (0.3, 10.0),        # its exponential tail toward +y
    "smear_alpha": (30.0, 250.0),   # generalized-Gaussian half-width
    "smear_p": (1.0, 8.0),          # its exponent: ~1 long-tailed, large = flat-topped
    "middle_sigma": (5.0, 18.0),
    "band_sigma": (2.0, 25.0),
    "band_tau": (0.5, 80.0),        # tail toward the well
    "fastest_sigma": (12.0, 70.0),  # each half of the bi-Gaussian
}
START = {  # start values of the component widths
    "well_sigma": 3.0, "well_tau": 4.0, "middle_sigma": 10.0, "band_sigma": 7.0, "band_tau": 10.0,
    "fastest_sigma": 30.0,
}
AREA_WIDTH = {"well": 8.0, "middle": 25.0}  # start area = start height x this (band, fastest: x 2.5 x width)
WELL_SEARCH = 25         # searched after the window start for the well peak
WELL_SHIFT = 8           # the well center may move this far past its start value
DIP_TOLERANCE = 6        # a lane's dip crossing may differ this much from its group median
SMEAR_AFTER_WELL = 30    # smear center at least this far after the well peak
SMEAR_BEFORE_BAND = 40   # ... and at least this far before the band maximum
PLATEAU_GAP = (20, 50)   # smear start height: median from well peak + 20 to band maximum - 50
BAND_START = 3           # band start center this far toward the well from the band maximum
BAND_SHIFT = (25, 10)    # band center may move 25 toward the well and 10 away from the maximum
MIDDLE_FREEDOM = 12      # the middle-band center may move this far from its start value
FASTEST_INSET = 5        # fastest start center kept this far inside --fastest-window
FASTEST_SMOOTH = 15      # median filter that locates the fastest start center
DUST_SIGMA = 6.0         # spike threshold above a running median, in noise sigmas
DUST_MEDIAN = 13         # running-median window
DUST_MAX_RUN = 7         # only runs this narrow count as dust; mask wider specks by hand (--mask)
DUST_AFTER_WELL = 15     # no dust masking before well peak + 15
BAND_GUARD = 25          # never mask dust this close to the band top
MIDDLE_GUARD = 20        # ... nor to the middle-band top
AREA_MARGIN = (80, 120)  # component areas integrate this far before / after the fit window


def safe(name: str) -> str:
    """A group name usable in a file name."""
    return re.sub(r"\W+", "_", name)


# A spreadsheet evaluates a text cell that starts with one of these as a formula. gelpy's own
# results.csv guards the same prefixes the same way; the helper is inline because an example must
# not import gelpy internals that are not public API.
FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def spreadsheet_safe(value):
    """A CSV cell that a spreadsheet will not evaluate: a leading apostrophe marks text.

    Lane labels and group names are the user's text (``=1+1`` is a legal label); numbers, negative
    ones included, are returned unchanged."""
    return "'" + value if isinstance(value, str) and value.startswith(FORMULA_PREFIXES) else value


def write_csv(path, rows, keys):
    """``rows`` (dicts) as a CSV file, UTF-8, every text cell guarded like gelpy's results.csv.

    Python 3.10's csv module leaves a field with a bare CR unquoted (3.11 and later quote it), and a
    reader that takes a bare CR for a record break would start a record at the text after it,
    possibly a live formula. So when any cell holds a CR, every cell that is not a number is quoted."""
    safe = [{k: spreadsheet_safe(r[k]) for k in keys} for r in rows]
    has_cr = any(isinstance(v, str) and "\r" in v for r in safe for v in r.values())
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=keys, quoting=csv.QUOTE_NONNUMERIC if has_cr else csv.QUOTE_MINIMAL)
        wr.writeheader()
        wr.writerows(safe)


def parse_lanes(spec: str) -> list[int]:
    """'0-9' or '0,2,4-6' -> lane indices."""
    out = []
    for part in spec.split(","):
        a, _, b = part.strip().partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def load_profiles(run: str, lanes: list[int]) -> dict[int, np.ndarray]:
    """Current full-length profile per lane, from gelpy's profile cache."""
    # Refresh first: manifest.json keeps a stale entry after a recipe edit until the next extract.
    # Without --allow-raw on purpose: that flag would measure the raw image, each lane with its own
    # background floor, and the fitted components would absorb it. A run without background removal
    # is refused (gelpy answers background_required); main() stops before for the same reason, and
    # any other refusal is printed with gelpy's own error, never swallowed.
    r = subprocess.run([sys.executable, "-m", "gelpy", "profiles", "extract", run, "--lane", "all"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, encoding="utf-8")
    if r.returncode:
        sys.exit(f"`gelpy profiles extract {run} --lane all` failed:\n{r.stderr.strip()}")
    with open(os.path.join(run, "manifest.json"), encoding="utf-8") as f:
        stages = json.load(f)["stages"]
    out = {}
    for i in lanes:
        h = stages[f"profile:lane={i}"]["hash"]
        with open(os.path.join(run, "cache", "profiles", f"lane_{i:03d}__{h}.json"), encoding="utf-8") as f:
            out[i] = np.asarray(json.load(f), dtype=float)
    return out


# ---------- shapes, each normalized to area A ----------
def emg(x, A, mu, sig, tau):
    z = (x - mu) / sig - sig / tau
    arg = np.clip(sig**2 / (2 * tau**2) - (x - mu) / tau, -700, 700)
    return A / (2 * tau) * np.exp(arg) * erfc(-z / SQRT2)


def gen_gauss(x, A, mu, alpha, p):
    return A * p / (2 * alpha * gamma(1.0 / p)) * np.exp(-(np.abs(x - mu) / alpha) ** p)


def gauss(x, A, mu, s):
    return A / (np.sqrt(2 * np.pi) * s) * np.exp(-0.5 * ((x - mu) / s) ** 2)


def emg_left(x, A, mu, sig, tau):
    """EMG mirrored: Gaussian core with an exponential tail toward -y (slow side)."""
    return emg(-x, A, -mu, sig, tau)


def bigauss(x, A, mu, sl, sr):
    h = A / (np.sqrt(np.pi / 2) * (sl + sr))
    return h * np.exp(-0.5 * ((x - mu) / np.where(x < mu, sl, sr)) ** 2)


# parameter layout: well(4) smear(4) middle(3) band(4) fastest(4) offset(1), names in PARAMS
def components(th, x, use_middle):
    out = {
        "well": emg(x, *th[0:4]),
        "smear": gen_gauss(x, *th[4:8]),
        "middle": gauss(x, *th[8:11]) if use_middle else np.zeros_like(x),
        "band": emg_left(x, *th[11:15]),
        "fastest": bigauss(x, *th[15:19]),
    }
    return out, th[19]


@dataclass
class LaneFit:
    lane_index: int
    label: str
    group: str
    middle: bool
    x_value: float
    y_start: int
    y_end: int
    well_area: float
    smear_area: float
    middle_area: float
    band_area: float
    fastest_area: float
    well_frac: float
    smear_frac: float
    middle_frac: float
    band_frac: float
    fastest_frac: float
    well_center: float
    middle_center: float
    band_center: float
    fastest_center: float
    band_migration_px: float
    offset: float
    r2: float
    residual_frac: float
    n_masked: int
    masked_ranges: list
    nfev: int
    status: int
    theta: list


def window_start(y, dip):
    """Zero-crossing back above baseline after the well-cavity dip found inside ``dip``."""
    m = dip[0] + int(np.argmin(y[dip[0]:dip[1]]))
    return m + int(np.argmax(y[m:] > 0))


def ranges(px):
    """Sorted pixel positions -> [[first, last], ...] of contiguous runs."""
    px = np.asarray(px, dtype=int)
    if not px.size:
        return []
    cuts = np.flatnonzero(np.diff(px) > 1) + 1
    return [[int(r[0]), int(r[-1])] for r in np.split(px, cuts)]


def fit_lane(i: int, yf: np.ndarray, label: str, group: str, start: int, cfg) -> LaneFit:
    use_middle = group in cfg.middle
    b = cfg.bounds
    y0 = window_start(yf, cfg.dip_window)
    if abs(y0 - start) > DIP_TOLERANCE:  # dip detection fooled -> group median
        y0 = start
    y_end = cfg.fit_end
    x = np.arange(y0, y_end, dtype=float)
    y = yf[y0:y_end]
    noise = float(np.std(yf[cfg.noise_window[0]:cfg.noise_window[1]]))

    # initial guesses from the data
    wk = y0 + int(np.argmax(y[:WELL_SEARCH]))
    lo_b, hi_b = cfg.band_windows.get(group, cfg.band_window)
    mub = lo_b + int(np.argmax(yf[lo_b:hi_b]))
    hb = yf[mub]
    f0, f1 = cfg.fastest_window
    muf = f0 + int(np.argmax(median_filter(yf[f0:f1], FASTEST_SMOOTH)))
    hf = max(float(median_filter(yf, FASTEST_SMOOTH)[muf]), 3 * noise)
    gap = yf[wk + PLATEAU_GAP[0] : mub - PLATEAU_GAP[1]]
    plateau = max(float(np.median(gap)) if gap.size else 0.0, 2 * noise)
    if use_middle:
        m0, m1 = cfg.middle_window
        mid0 = m0 + int(np.argmax(yf[m0:m1]))
    else:
        mid0 = 0.5 * (wk + mub)

    # dust: narrow spikes far above a running median, outside the well peak
    w = np.ones_like(y)
    dev = y - median_filter(y, DUST_MEDIAN)
    spike = (dev > DUST_SIGMA * noise) & (x > wk + DUST_AFTER_WELL)
    for a, c in cfg.masks.get(i, []):  # by hand, after confirming the speck in the image
        w[(x >= a) & (x <= c)] = 0.0
    spike &= ~((x > mub - BAND_GUARD) & (x < mub + BAND_GUARD))  # never mask the band top
    if use_middle:
        spike &= ~((x > mid0 - MIDDLE_GUARD) & (x < mid0 + MIDDLE_GUARD))  # ... nor the middle top
    idx = np.flatnonzero(spike)
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1) if idx.size else []
    for r in runs:
        if len(r) <= DUST_MAX_RUN:  # wider bumps are real signal
            w[max(0, r[0] - 4) : r[-1] + 5] = 0.0

    span = mub - wk
    s, aw = START, AREA_WIDTH
    th0 = [
        yf[wk] * aw["well"], wk, s["well_sigma"], s["well_tau"],
        plateau * span * 0.9, wk + 0.45 * span, span / 1.6, 1.3,
        plateau * aw["middle"], mid0, s["middle_sigma"],
        hb * 2.5 * s["band_tau"], mub - BAND_START, s["band_sigma"], s["band_tau"],
        hf * 2.5 * s["fastest_sigma"], np.clip(muf, f0 + FASTEST_INSET, f1 - FASTEST_INSET),
        s["fastest_sigma"], s["fastest_sigma"],
        0.0,
    ]
    lo = np.array([
        0, y0, b["well_sigma"][0], b["well_tau"][0],
        0, wk + SMEAR_AFTER_WELL, b["smear_alpha"][0], b["smear_p"][0],
        0, mid0 - MIDDLE_FREEDOM, b["middle_sigma"][0],
        0, mub - BAND_SHIFT[0], b["band_sigma"][0], b["band_tau"][0],
        0, f0, b["fastest_sigma"][0], b["fastest_sigma"][0],
        -2 * noise,
    ], dtype=float)
    hi = np.array([
        np.inf, wk + WELL_SHIFT, b["well_sigma"][1], b["well_tau"][1],
        np.inf, mub - SMEAR_BEFORE_BAND, b["smear_alpha"][1], b["smear_p"][1],
        np.inf, mid0 + MIDDLE_FREEDOM, b["middle_sigma"][1],
        np.inf, mub + BAND_SHIFT[1], b["band_sigma"][1], b["band_tau"][1],
        np.inf, f1, b["fastest_sigma"][1], b["fastest_sigma"][1],
        2 * noise,
    ], dtype=float)
    # a switched-off middle component is dropped from the parameter vector, not frozen
    free = np.ones(len(th0), bool)
    if not use_middle:
        free[8:11] = False
    bad = [PARAMS[k] for k in np.flatnonzero(free) if not lo[k] < hi[k]]
    if bad:
        raise SystemExit(f"lane {i}: empty bounds for {', '.join(bad)}; check the windows against "
                         "the profile plot")
    th0 = np.clip(np.asarray(th0, dtype=float), lo, hi)

    def full(p):
        th = th0.copy()
        th[free] = p
        return th

    def resid(p):
        c, off = components(full(p), x, use_middle)
        return w * (sum(c.values()) + off - y)

    # two starts for the smear shape (long-tailed p~1.3 vs flat-topped p~3); keep the lower cost
    best = None
    for alpha0, p0 in [(span / 1.6, 1.3), (span / 2.5, 3.0)]:
        t0 = th0.copy()
        t0[6:8] = np.clip([alpha0, p0], lo[6:8], hi[6:8])
        r = least_squares(resid, t0[free], bounds=(lo[free], hi[free]), loss="soft_l1", x_scale="jac",
                          f_scale=max(3 * noise, 1e-9), max_nfev=5000)
        if best is None or r.cost < best.cost:
            best = r
    th = full(best.x)
    c, off = components(th, x, use_middle)
    fit = sum(c.values()) + off
    m = w > 0
    r2 = 1 - np.sum((y[m] - fit[m]) ** 2) / np.sum((y[m] - y[m].mean()) ** 2)

    xw = np.arange(y0 - AREA_MARGIN[0], y_end + AREA_MARGIN[1], dtype=float)
    cw, _ = components(th, xw, use_middle)
    areas = {k: float(trapezoid(cw[k], xw)) for k in COMPS}
    tot = sum(areas.values())
    integral = float(trapezoid((y - off)[m], x[m]))
    resid_frac = float(trapezoid(np.abs(y - fit)[m], x[m]) / max(integral, 1e-12))
    xf = np.arange(y0, y_end, 0.1)
    band_mode = float(xf[np.argmax(emg_left(xf, *th[11:15]))])
    return LaneFit(
        lane_index=i, label=label, group=group, middle=use_middle, x_value=cfg.x_of(i, label),
        y_start=int(y0), y_end=int(y_end),
        **{f"{k}_area": areas[k] for k in COMPS},
        **{f"{k}_frac": areas[k] / tot for k in COMPS},
        well_center=float(th[1]), middle_center=float(th[9]) if use_middle else float("nan"),
        band_center=band_mode, fastest_center=float(th[16]),
        band_migration_px=band_mode - float(th[1]), offset=float(off), r2=float(r2),
        residual_frac=resid_frac, n_masked=int((w == 0).sum()), masked_ranges=ranges(x[w == 0]),
        nfev=int(best.nfev), status=int(best.status), theta=[float(v) for v in th],
    )


def draw(ax, lf: LaneFit, yf: np.ndarray, plot_range, lw=1.1, legend=False):
    x = np.arange(lf.y_start, lf.y_end, dtype=float)
    c, off = components(np.array(lf.theta), x, lf.middle)
    xp = np.arange(*plot_range)
    ax.plot(xp, yf[plot_range[0] : plot_range[1]], color="0.35", lw=lw, label="profile")
    ax.plot(x, sum(c.values()) + off, color="black", lw=lw * 0.8, ls="--", label="sum")
    ax.axvline(lf.y_start, color="0.7", lw=0.7, ls=":")
    for k in COMPS:
        if k == "middle" and not lf.middle:
            continue
        ax.fill_between(x, off, c[k] + off, color=COLORS[k], alpha=0.5, lw=0,
                        label=f"{NAMES[k]} {getattr(lf, k + '_frac') * 100:.1f}%")
    ax.set_xlim(*plot_range)
    ax.tick_params(labelsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if legend:
        ax.legend(fontsize=7, frameon=False)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="gelpy run folder (background removed)")
    ap.add_argument("--out", required=True, help="output folder; figures/ and results/ go here")
    ap.add_argument("--group", action="append", default=[], metavar="NAME=LANES",
                    help="lane group, e.g. A=0-9 or B=10,12-19 (repeatable); default: one group 'all'")
    ap.add_argument("--dip-window", type=int, nargs=2, required=True, metavar=("Y0", "Y1"),
                    help="y-range holding the well-cavity dip; the fit starts at its zero-crossing")
    ap.add_argument("--fit-end", type=int, required=True, metavar="Y", help="end of the fit window")
    ap.add_argument("--band-window", type=int, nargs=2, required=True, metavar=("Y0", "Y1"),
                    help="y-range holding the leading band's maximum")
    ap.add_argument("--band-window-for", nargs=3, action="append", default=[],
                    metavar=("GROUP", "Y0", "Y1"), help="per-group --band-window (repeatable)")
    ap.add_argument("--fastest-window", type=int, nargs=2, required=True, metavar=("Y0", "Y1"),
                    help="y-range allowed for the fastest component's center")
    ap.add_argument("--noise-window", type=int, nargs=2, required=True, metavar=("Y0", "Y1"),
                    help="signal-free y-range for the noise estimate")
    ap.add_argument("--middle", action="append", default=[], metavar="GROUP",
                    help="group that gets the optional middle component (repeatable)")
    ap.add_argument("--middle-window", type=int, nargs=2, metavar=("Y0", "Y1"),
                    help="y-range holding the middle band's maximum (needed with --middle)")
    ap.add_argument("--mask", action="append", default=[], metavar="LANE:Y0-Y1",
                    help="exclude a range by hand, after confirming the speck in the image (repeatable)")
    ap.add_argument("--bound", action="append", default=[], metavar="NAME=LO,HI",
                    help=f"override a shape bound (px); names: {', '.join(BOUNDS)}")
    ap.add_argument("--plot-range", type=int, nargs=2, metavar=("Y0", "Y1"),
                    help="x-range of the plots; default: start of --dip-window to --fit-end")
    ap.add_argument("--x-from-label", metavar="REGEX",
                    help="numeric x for the summary plots: first capture group (or the match) of REGEX "
                         "in each lane label; default: the lane index")
    ap.add_argument("--x-label", help="axis label for that x")
    ap.add_argument("--x-scale", choices=["linear", "symlog", "log"], default="linear")
    return ap.parse_args()


def main():
    cfg = parse_args()
    run = cfg.run_dir
    if not os.path.isfile(os.path.join(run, "recipe.json")):
        sys.exit(f"{run}: not a gelpy run folder (no recipe.json)")
    with open(os.path.join(run, "recipe.json"), encoding="utf-8") as f:
        rec = json.load(f)
    if rec.get("background") is None:
        sys.exit(f"no background removal in {run}: gelpy refuses profiles without it (background_required), "
                 "and the components are fitted on background-corrected profiles only. "
                 f"Run `gelpy background remove {run}`, check `gelpy background show {run}`, then repeat")
    labels = {lane["index"]: lane["label"] for lane in rec["lanes"]["lanes"]}
    groups = {}
    for g in cfg.group:
        name, sep, spec = g.partition("=")
        try:
            if not (name and sep):
                raise ValueError
            groups[name] = parse_lanes(spec)
        except ValueError:
            sys.exit(f"--group {g!r}: expected NAME=LANES, e.g. A=0-9 or B=10,12-19")
    groups = groups or {"all": sorted(labels)}
    unknown = [i for lanes in groups.values() for i in lanes if i not in labels]
    unknown += [g for g in cfg.middle + [o[0] for o in cfg.band_window_for] if g not in groups]
    if unknown:
        sys.exit(f"unknown lanes or groups: {unknown}")
    if cfg.middle and not cfg.middle_window:
        sys.exit("--middle needs --middle-window")
    try:
        cfg.band_windows = {g: (int(a), int(b)) for g, a, b in cfg.band_window_for}
    except ValueError:
        sys.exit("--band-window-for: expected GROUP Y0 Y1, e.g. B 300 420")
    cfg.bounds = dict(BOUNDS)
    for item in cfg.bound:
        name, _, vals = item.partition("=")
        if name not in BOUNDS:
            sys.exit(f"unknown bound {name!r}; names: {', '.join(BOUNDS)}")
        try:
            lo, hi = (float(v) for v in vals.split(","))
        except ValueError:
            sys.exit(f"--bound {item!r}: expected NAME=LO,HI, e.g. well_sigma=1,6")
        cfg.bounds[name] = (lo, hi)
    cfg.masks = {}
    for item in cfg.mask:
        try:
            lane, span = item.split(":")
            a, b = span.split("-")
            cfg.masks.setdefault(int(lane), []).append((int(a), int(b)))
        except ValueError:
            sys.exit(f"--mask {item!r}: expected LANE:Y0-Y1, e.g. 5:600-620")
    if cfg.x_from_label:
        try:
            re.compile(cfg.x_from_label)
        except re.error as e:
            sys.exit(f"--x-from-label {cfg.x_from_label!r}: not a valid regular expression ({e})")

    def x_of(i, label):
        if not cfg.x_from_label:
            return float(i)
        m = re.search(cfg.x_from_label, label)
        return float(m.group(1) if m and m.groups() else m.group(0)) if m else float("nan")

    cfg.x_of = x_of
    plot_range = tuple(cfg.plot_range or (cfg.dip_window[0], cfg.fit_end))
    fig_dir = os.path.join(cfg.out, "figures")
    fit_dir = os.path.join(fig_dir, "fits")
    res_dir = os.path.join(cfg.out, "results")

    prof = load_profiles(run, sorted({i for lanes in groups.values() for i in lanes}))
    n = min(len(y) for y in prof.values())
    windows = {"--dip-window": cfg.dip_window, "--band-window": cfg.band_window,
               "--fastest-window": cfg.fastest_window, "--noise-window": cfg.noise_window,
               "--plot-range": plot_range, "--middle-window": cfg.middle_window,
               **{f"--band-window-for {g}": w for g, w in cfg.band_windows.items()}}
    if not cfg.dip_window[1] < cfg.fit_end <= n:
        sys.exit(f"--fit-end {cfg.fit_end}: needs to lie after --dip-window and at most {n}, "
                 "the profile length")
    for flag, w in windows.items():
        if w is not None and not 0 <= w[0] < w[1] <= n:
            sys.exit(f"{flag} {w[0]} {w[1]}: needs 0 <= Y0 < Y1 <= {n}, the profile length")
    os.makedirs(fit_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    fits = {}
    for g, lanes in groups.items():
        start = int(np.median([window_start(prof[i], cfg.dip_window) for i in lanes]))
        fits[g] = [fit_lane(i, prof[i], labels[i], g, start, cfg) for i in lanes]

    for lfs in fits.values():
        for lf in lfs:
            fig, ax = plt.subplots(figsize=(7, 3.4), dpi=150)
            draw(ax, lf, prof[lf.lane_index], plot_range, legend=True)
            ax.set_title(f"lane {lf.lane_index}: {lf.label}   (R2 {lf.r2:.3f}, "
                         f"residual {lf.residual_frac * 100:.1f}%)", fontsize=9)
            ax.set_xlabel("position along the lane [px]")
            ax.set_ylabel("corrected intensity")
            fig.tight_layout()
            fig.savefig(os.path.join(fit_dir, f"lane_{lf.lane_index:03d}.png"))
            plt.close(fig)
            print(f"{lf.lane_index:3d} {lf.label:24s} y0={lf.y_start} "
                  + " ".join(f"{k}={getattr(lf, k + '_frac') * 100:5.1f}" for k in COMPS)
                  + f"  band@{lf.band_center:.0f} mig={lf.band_migration_px:.0f} "
                  f"fast@{lf.fastest_center:.0f} R2={lf.r2:.3f} res={lf.residual_frac * 100:.1f}% "
                  f"mask={lf.n_masked}")

    # overview per group
    for g, lfs in fits.items():
        ncol = min(5, len(lfs))
        nrow = -(-len(lfs) // ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.2 * nrow + 0.4), dpi=130,
                                 sharex=True, squeeze=False)
        for ax in axes.flat[len(lfs):]:
            ax.set_visible(False)
        for ax, lf in zip(axes.flat, lfs):
            draw(ax, lf, prof[lf.lane_index], plot_range, lw=0.9)
            ax.set_title(lf.label, fontsize=9)
        used = [k for k in COMPS if k != "middle" or g in cfg.middle]
        handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[k], alpha=0.5) for k in used]
        fig.legend(handles, [NAMES[k] for k in used], loc="upper center", ncol=len(used), frameon=False,
                   fontsize=10)
        fig.supxlabel("position along the lane [px]", fontsize=10)
        fig.supylabel("corrected intensity", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / (3.2 * nrow + 0.4)))
        fig.savefig(os.path.join(fig_dir, f"fit_overview_{safe(g)}.png"))
        plt.close(fig)

    # stacked fractions per group
    for g, lfs in fits.items():
        used = [k for k in ["band", "middle", "smear", "well", "fastest"] if k != "middle" or g in cfg.middle]
        fig, ax = plt.subplots(figsize=(max(4.0, 0.4 * len(lfs) + 2), 5.5), dpi=150)
        xpos = np.arange(len(lfs))
        bottom = np.zeros(len(lfs))
        for k in used:
            v = np.array([getattr(lf, k + "_frac") * 100 for lf in lfs])
            ax.bar(xpos, v, 0.75, bottom=bottom, color=COLORS[k], edgecolor="white", linewidth=1.2,
                   label=NAMES[k])
            for xx, vv, b0 in zip(xpos, v, bottom):
                if vv >= 5:
                    ax.text(xx, b0 + vv / 2, f"{vv:.0f}", ha="center", va="center", fontsize=7,
                            color="white" if k in ("band", "fastest", "well") else "0.15")
            bottom += v
        ax.set_xticks(xpos)
        ax.set_xticklabels([lf.label for lf in lfs], rotation=90, fontsize=8)
        ax.set_ylabel("share of fitted lane intensity [%]")
        ax.set_ylim(0, 100)
        ax.set_title(f"{g}: component fractions per lane", fontsize=11)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"fractions_{safe(g)}.png"), bbox_inches="tight")
        plt.close(fig)

    # fastest-peak area and leading-band migration per lane, one line per group
    xlabel = cfg.x_label or ("value from the lane label" if cfg.x_from_label else "lane index")
    for key, ylab, fname, title in [
        ("fastest_area", "fastest-peak area [a.u.]", "fastest_area.png", "Fastest-peak area"),
        ("band_migration_px", "leading band distance from well [px]", "band_migration.png",
         "Leading-band migration"),
    ]:
        fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=150)
        for g, lfs in fits.items():
            pts = sorted((lf.x_value, getattr(lf, key)) for lf in lfs if np.isfinite(lf.x_value))
            if pts:
                ax.plot(*zip(*pts), "o-", label=g, ms=5, lw=1.4)
        if cfg.x_scale != "linear":
            ax.set_xscale(cfg.x_scale)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=11)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.yaxis.grid(True, color="0.9")
        ax.set_axisbelow(True)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, fname))
        plt.close(fig)

    rows = [asdict(lf) for lfs in fits.values() for lf in lfs]
    with open(os.path.join(res_dir, "component_fits.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    keys = [k for k in rows[0] if k not in ("theta", "masked_ranges")]
    write_csv(os.path.join(res_dir, "component_fits.csv"), rows, keys)  # labels are user text


if __name__ == "__main__":
    main()
