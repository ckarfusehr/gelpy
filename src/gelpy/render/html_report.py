"""A static HTML report of a run: one page, a stylesheet and its images; no scripts.

Pure: it takes the run's data and the bytes of its images and writes files, and never opens a run
folder -- ``cli/_render.py`` collects the data and the images, and ``runfolder/report_folder.py``
owns the folder they go into. Every text value that reaches the page (labels, file names,
parameter values) passes through ``html.escape``, and the page carries no script, no inline style
and no reference outside its folder, so it renders the same from a copy on any machine. Control
and bidirectional-formatting characters in a value are shown as ``<U+202E>``, not acted on, and
every element that holds a value is isolated from the text around it.

Styling is CSS custom properties (``--gelpy-*``) with a neutral default, ``DEFAULT_CSS``,
written as ``report.css``. A caller restyles the page with its own stylesheet, written as
``theme.css`` byte for byte and linked after the default, so it can override any variable or rule.

``build_report`` makes every file in memory, so a page that cannot be built fails before anything
is written; ``write_report`` writes them atomically, the page last.
"""

from __future__ import annotations

import html
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from ..schemas.fit_config import FitConfigStore
from ..schemas.recipe import Recipe
from ..schemas.report import REPORT_INDEX
from ..schemas.results import AnalysisResult
from ..schemas.run import RunStatus
from ..utils.atomic import atomic_write

#: The files of a report besides its images.
INDEX_FILE = REPORT_INDEX
STYLESHEET_FILE = "report.css"
THEME_FILE = "theme.css"

DEFAULT_CSS = """\
/* gelpy report: neutral default. A theme stylesheet overrides any --gelpy-* variable. */
:root {
  --gelpy-font: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --gelpy-mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  --gelpy-bg: #ffffff;
  --gelpy-fg: #1f2328;
  --gelpy-muted: #59636e;
  --gelpy-border: #d1d9e0;
  --gelpy-surface: #f6f8fa;
  --gelpy-accent: #0969da;
  --gelpy-ok: #1a7f37;
  --gelpy-low: #9a6700;
  --gelpy-max-width: 72rem;
}
body {
  margin: 0;
  background: var(--gelpy-bg);
  color: var(--gelpy-fg);
  font-family: var(--gelpy-font);
  line-height: 1.45;
}
main { max-width: var(--gelpy-max-width); margin: 0 auto; padding: 1.5rem; }
h1 { font-size: 1.6rem; margin: 0 0 .25rem; }
h2 {
  font-size: 1.2rem;
  margin: 2rem 0 .75rem;
  padding-bottom: .25rem;
  border-bottom: 2px solid var(--gelpy-accent);
}
h3 { font-size: 1rem; margin: 1.25rem 0 .5rem; }
.meta, .note, figcaption { color: var(--gelpy-muted); font-size: .9rem; }
.warnings { color: var(--gelpy-low); font-weight: 600; }
figure { margin: 1rem 0; }
img { max-width: 100%; height: auto; border: 1px solid var(--gelpy-border); background: var(--gelpy-bg); }
table { border-collapse: collapse; width: 100%; font-size: .9rem; margin: .5rem 0; }
th, td {
  border-bottom: 1px solid var(--gelpy-border);
  padding: .3rem .5rem;
  text-align: left;
  vertical-align: top;
  overflow-wrap: anywhere;
}
th { background: var(--gelpy-surface); font-weight: 600; border-bottom: 2px solid var(--gelpy-accent); }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
td.reasons { color: var(--gelpy-muted); font-size: .85rem; }
.confidence-ok { color: var(--gelpy-ok); }
.confidence-low { color: var(--gelpy-low); font-weight: 600; }
dl { display: grid; grid-template-columns: max-content 1fr; gap: .25rem 1rem; }
dt { color: var(--gelpy-muted); }
dd { margin: 0; font-family: var(--gelpy-mono); font-size: .85rem; word-break: break-word; }
h1, td, th, dd, figcaption, .user { unicode-bidi: isolate; }
"""


@dataclass(frozen=True)
class Figure:
    """One image on the page: its path relative to index.html, its caption and a short alt text.

    ``alt`` is the caption when left empty. ``width`` and ``height`` (pixels) let the browser
    keep room for the image before it loads.
    """

    src: str
    caption: str
    alt: str = ""
    width: int | None = None
    height: int | None = None


@dataclass(frozen=True)
class ReportContent:
    """Everything the page shows; ``cli/_render.write_html_report`` assembles it.

    ``fingerprint`` (``runfolder.report_folder.report_fingerprint``) is printed in Provenance: it
    says which state of the run the page shows. ``damaged`` lists the lanes whose fit file did not
    validate and is left out of ``result``.
    """

    run_name: str
    gelpy_version: str
    recipe: Recipe
    status: RunStatus
    result: AnalysisResult
    fit_config: FitConfigStore
    gel: list[Figure] = field(default_factory=list)
    profiles: list[Figure] = field(default_factory=list)
    fits: list[Figure] = field(default_factory=list)
    fingerprint: str = ""
    damaged: list[int] = field(default_factory=list)


def build_report(
    content: ReportContent,
    *,
    theme_css: bytes | str | None = None,
    images: Mapping[str, bytes] | None = None,
) -> dict[str, bytes]:
    """Every file of the report, relative path -> bytes, in the order to write them.

    The images (``images``: the paths the figures name, and their bytes) come first, then
    ``report.css``, ``theme.css`` when ``theme_css`` is given (bytes are kept exactly as they
    are; text is UTF-8), and ``index.html`` last. The page is built, and encoded, before this
    returns: nothing has been written when it fails.
    """
    stylesheets = [STYLESHEET_FILE]
    theme = None
    if theme_css is not None:
        theme = theme_css.encode("utf-8") if isinstance(theme_css, str) else theme_css
        stylesheets.append(THEME_FILE)
    page = build_page(content, stylesheets).encode("utf-8")
    files = {name: images[name] for name in sorted(images or {})}
    files[STYLESHEET_FILE] = DEFAULT_CSS.encode("utf-8")
    if theme is not None:
        files[THEME_FILE] = theme
    files[INDEX_FILE] = page
    return files


def write_report(files: Mapping[str, bytes], out_dir: str | Path) -> Path:
    """Write ``files`` (``build_report``) into ``out_dir``, each atomically, in order; the page last.

    ``index.html`` goes last, so a reader never opens a page whose stylesheet or images are not
    there yet. Returns the path of the page.
    """
    out = Path(out_dir)
    for name, data in files.items():
        atomic_write(out / name, data)
    return out / INDEX_FILE


def render_html_report(
    content: ReportContent,
    out_dir: str | Path,
    *,
    theme_css: bytes | str | None = None,
    images: Mapping[str, bytes] | None = None,
) -> Path:
    """Build the report and write it into ``out_dir``; the path of ``index.html``.

    The folder is not checked: ``runfolder.report_folder`` decides which folder a report may be
    written into, and ``cli/_render.write_html_report`` asks it first.
    """
    return write_report(build_report(content, theme_css=theme_css, images=images), out_dir)


#: What ``_e`` shows instead of acting on it: the C0 and C1 controls but tab and line feed, DEL,
#: and the characters that reorder or isolate text (``<U+202E>`` ends a line the wrong way round).
_VISIBLE = {
    code: f"<U+{code:04X}>"
    for code in (
        *range(0x20),
        0x7F,
        *range(0x80, 0xA0),
        0x061C,
        0x200E,
        0x200F,
        *range(0x202A, 0x202F),
        *range(0x2066, 0x206A),
    )
    if code not in (0x09, 0x0A)
}


def _e(value: object) -> str:
    text = "" if value is None else str(value)
    return html.escape(text.translate(_VISIBLE), quote=True)


def _user(value: object) -> str:
    """A value in its own element, so that it cannot reorder the text around it."""
    return f'<span class="user">{_e(value)}</span>'


def _table(headers: list[str], rows: list[str]) -> str:
    head = "".join(f'<th scope="col">{_e(h)}</th>' for h in headers)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _figures(figures: list[Figure]) -> list[str]:
    out = []
    for f in figures:
        size = f' width="{f.width}" height="{f.height}"' if f.width and f.height else ""
        out.append(
            f'<figure><img src="{_e(f.src)}" alt="{_e(f.alt or f.caption)}"{size} loading="lazy">'
            f"<figcaption>{_e(f.caption)}</figcaption></figure>"
        )
    return out


_STAGE = re.compile(r"(?P<stage>[a-z]+):lane=(?P<lane>[0-9]+)")


def _stage_in_words(key: str) -> str:
    """``fit:lane=0`` as "lane 0: fit"; a key of another shape (``image``) as it is."""
    match = _STAGE.fullmatch(key)
    return f"lane {match['lane']}: {match['stage']}" if match else key


def build_page(c: ReportContent, stylesheets: list[str]) -> str:
    """The page, as text."""
    source = c.recipe.source_image
    parts = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>gelpy report: {_e(c.run_name)}</title>",
        *(f'<link rel="stylesheet" href="{_e(name)}">' for name in stylesheets),
        "</head>",
        "<body>",
        "<main>",
        f"<h1>{_e(c.run_name)}</h1>",
        f'<p class="meta">gelpy {_e(c.gelpy_version)} &middot; {_user(Path(source.path).name)} '
        f"&middot; {source.width} &times; {source.height} px &middot; "
        f"{len(c.recipe.lanes.lanes)} lanes</p>",
    ]
    if c.status.warnings:
        parts.append(
            '<p class="warnings">Warnings: ' + " ".join(_user(w) for w in c.status.warnings) + "</p>"
        )
    if c.status.stale_stages:
        stale = ", ".join(_stage_in_words(key) for key in c.status.stale_stages)
        parts.append(f'<p class="note">Stale cached stages, recomputed on next use: {_user(stale)}</p>')
    parts += ["<h2>Gel</h2>", *_figures(c.gel)]
    parts += ["<h2>Lanes</h2>", _lanes_table(c)]
    parts += ["<h2>Results</h2>", *_results(c)]
    if c.profiles:
        parts += ["<h2>Profiles</h2>", *_figures(c.profiles)]
    if c.fits:
        parts += ["<h2>Fits</h2>", *_figures(c.fits)]
    parts += ["<h2>Provenance</h2>", _provenance(c), "</main>", "</body>", "</html>", ""]
    return "\n".join(parts)


def _lanes_table(c: ReportContent) -> str:
    states = {lane.index: lane for lane in c.status.lanes}
    confidence = {lane.lane_index: lane.confidence for lane in c.result.lanes}
    rows = []
    for lane in c.recipe.lanes.lanes:
        state = states.get(lane.index)
        if lane.index in c.damaged:
            fit, bands = "damaged", ""
        elif state is None or not state.has_fit:
            fit, bands = "none", ""
        elif state.fit_current:
            fit, bands = "current", str(state.n_bands)
        else:
            fit, bands = "stale", ""
        conf = confidence.get(lane.index, "")
        cell = f'<td class="confidence-{_e(conf)}">{_e(conf)}</td>' if conf else "<td></td>"
        rows.append(
            f'<tr><td class="num">{lane.index}</td><td>{_e(lane.label)}</td>'
            f'<td class="num">{lane.x_center_px:.1f}</td><td>{fit}</td>'
            f'<td class="num">{bands}</td>{cell}</tr>'
        )
    return _table(["lane", "label", "centre x (px)", "fit", "bands", "confidence"], rows)


def _why(lane) -> str:
    """Why a fit is empty or low: its coverage and the reasons it carries."""
    parts = [f"coverage {lane.coverage:.3f}"] if lane.coverage is not None else []
    parts += lane.confidence_reasons
    return f"Lane {lane.lane_index}: " + "; ".join(parts) if parts else ""


def _results(c: ReportContent) -> list[str]:
    out = []
    if c.damaged:
        lanes = ", ".join(f"lane {i}" for i in c.damaged)
        out.append(
            f'<p class="note">Damaged fit file, not shown here: {_e(lanes)}. '
            "Refit the lane to read its results.</p>"
        )
    rows = []
    for lane in c.result.lanes:
        label = _e(lane.label)
        conf = f'<td class="confidence-{_e(lane.confidence)}">{_e(lane.confidence)}</td>'
        if lane.status != "ok" or not lane.bands:
            note = _e(lane.status) + (f": {_e(lane.error)}" if lane.error else "")
            rows.append(
                f'<tr><td class="num">{lane.lane_index}</td><td>{label}</td>'
                f'<td colspan="5">{note}</td>{conf}</tr>'
            )
        else:
            for band in lane.bands:
                shape = band.params.kind if band.params is not None else "region"
                rows.append(
                    f'<tr><td class="num">{lane.lane_index}</td><td>{label}</td>'
                    f'<td class="num">{band.band_number}</td><td>{_e(band.component_class)}</td>'
                    f'<td>{_e(shape)}</td><td class="num">{band.area_fraction:.3f}</td>'
                    f'<td class="num">{band.maximum_position:.1f}</td>{conf}</tr>'
                )
        why = _why(lane) if lane.status != "ok" or lane.confidence == "low" else ""
        if why:
            rows.append(f'<tr><td class="reasons" colspan="8">{_e(why)}</td></tr>')
    if not rows:
        return [*out, '<p class="note">No current fits.</p>']
    headers = [
        "lane", "label", "band", "class", "shape", "area fraction",
        "position from slice start (px)", "confidence",
    ]
    return [*out, _table(headers, rows)]


def _provenance(c: ReportContent) -> str:
    r = c.recipe
    s, m, g = r.source_image, r.image, r.contrast.gamma
    linear = r.contrast.intensity_range
    contrast = f"gamma {g.gamma}, gain {g.gain}" + (
        f", linear range {linear.low} to {linear.high} ({linear.mode})" if linear else ""
    )
    background = (
        "none"
        if r.background is None
        else f"{r.background.model} {json.dumps(r.background.params, sort_keys=True)}"
    )
    items = [
        ("gelpy", c.gelpy_version),
        *([("report fingerprint", c.fingerprint)] if c.fingerprint else []),
        ("source image", Path(s.path).name),
        ("source image sha256", s.sha256),
        ("size", f"{s.width} x {s.height} px, page {m.page + 1} of {m.source_pages}"),
        (
            "interpretation",
            f"channel {m.channel_strategy}, polarity {m.polarity} "
            f"({m.polarity_source}, confidence {m.polarity_confidence})",
        ),
        ("lanes", f"{len(r.lanes.lanes)}, profile width {r.lanes.profile_width_px} px"),
        ("display contrast", contrast),
        ("background", background),
    ]
    out = ["<dl>", *(f"<dt>{_e(k)}</dt><dd>{_e(v)}</dd>" for k, v in items), "</dl>"]
    if c.fit_config.lanes:
        rows = []
        for index in sorted(c.fit_config.lanes):
            cfg = c.fit_config.lanes[index]
            end = "end" if cfg.profile.slice_end is None else cfg.profile.slice_end
            baseline = cfg.fit.baseline.method + (
                f" order {cfg.fit.baseline.poly_order}" if cfg.fit.baseline.method == "poly" else ""
            )
            rows.append(
                f'<tr><td class="num">{index}</td><td>{_e(cfg.fit.method)}</td>'
                f"<td>{_e(baseline)}</td><td>{_e(', '.join(cfg.fit.shapes))}</td>"
                f"<td>{_e(cfg.profile.slice_start)} to {_e(end)}</td>"
                f'<td>{_e(cfg.profile.normalization)}</td><td class="num">{cfg.fit.seed}</td></tr>'
            )
        out += [
            "<h3>Fit configuration per lane</h3>",
            _table(["lane", "method", "baseline", "shapes", "slice", "normalisation", "seed"], rows),
            '<p class="note">Full configuration: fit_config.json</p>',
        ]
    return "\n".join(out)
