---
title: The HTML report
audience: both
summary: "`gelpy report html` writes a run as a static page: the lane overlay, clean gel views, the lane and results tables, profiles, fits and provenance. No scripts, every text escaped, styling through CSS variables. It replaces only its own earlier output, and `status` says when the page is out of date."
subject:
  - src/gelpy/render/html_report.py
  - src/gelpy/cli/_render.py
  - src/gelpy/runfolder/report_folder.py
  - src/gelpy/schemas/report.py
subject_digest: 560a8db2077ec886
---

`report.md` summarises a run for a terminal. `gelpy report html RUN` (the tool `report_html`)
writes the same run as a page to open in a browser: a folder with `index.html`, the stylesheet
`report.css`, the images under `img/`, `theme.css` when you pass a stylesheet, and
`.gelpy-report.json`, the file that marks the folder as a report of gelpy's (see "Where it
writes"). It leaves the analysis alone: the recipe, the fit configuration and the results do not
change, so it can be rebuilt after every step. It draws its images into `render/`, as `image show`
does, and holds the run lock while it works, like every command that writes in the run folder.

Two variables for every block below; the test that executes this page sets both itself:

```bash
# doc-test: skip
export IMAGE=examples/data/folding_screen.tif
export RUN=/tmp/gelpy-html-report
```

```bash
gelpy init "$RUN" "$IMAGE" --n-lanes 20 > /dev/null
gelpy lanes detect "$RUN" --n-lanes 20 --apply --no-plot > /dev/null
gelpy background remove "$RUN" > /dev/null
gelpy fit run "$RUN" --lane 0 --lane 1 --plot > /dev/null
gelpy report html "$RUN"
```

```console
{
  "index": "report/index.html",
  "files": [
    "img/fits/lane_000.png",
    "img/fits/lane_001.png",
    "img/linear.png",
    "img/non_linear.png",
    "img/non_linear_lanes.png",
    "img/profiles_overview.png",
    "index.html",
    "report.css"
  ]
}
```

`index` is relative to the run folder when the report is inside it, and absolute when `--out`
puts it elsewhere. `files` lists the report's files relative to its folder.

## What is on the page

- **Top:** the run's name, the gelpy version, the source image and the lane count. A **warnings**
  paragraph shows what `status` warns about (a polarity that was detected with low confidence, for
  example), and the stale cached stages are named in words ("lane 0: fit").
- **Gel:** the lane overlay (the profile strips gelpy measures, over the gamma view), then the
  clean linear and gamma views with the labels on top. They are rendered fresh on every build.
- **Lanes:** index, label, centre, whether the lane has a current, a stale or no fit, or a fit
  file that is damaged, its band count and its confidence.
- **Results:** one row per band -- class, shape, area fraction, position, the lane's confidence --
  assembled like `results show`, so stale fits are left out. A position is counted from the start
  of the lane's slice, as in `results.csv`; the slice is in Provenance. Under a lane that is empty
  or low in confidence there is a line with its coverage and the reasons. A lane whose fit file is
  damaged is listed as damaged and left out of the table; the report still builds.
- **Profiles:** all lanes overlaid, background-corrected and area-normalised, when background
  removal is configured.
- **Fits:** every current fit's plot. A plot is drawn again only when it shows another fit than
  the lane's current one, or another label: the fit hash leaves the label out, and a plot is
  titled with it.
- **Provenance:** the gelpy version, the report's fingerprint, the source image by name and
  sha256, how it was read, the lane layout, the display contrast, the background model and its
  parameters, and each fitted lane's method, baseline, shapes, slice and seed. The full
  configuration of a lane is in `fit_config.json`.

## Self-contained and escaped

The page carries no script, no inline style, no event handler and no reference outside its
folder, and every text value on it -- labels above all -- is escaped, so a label cannot become
markup.
<!-- claim: html-report-has-no-scripts -->
A control character, and a character that reorders or isolates text, is shown as `<U+202E>`
instead of acted on, and every element that holds a value is isolated from the text around it, so
a label cannot reverse what stands next to it.

## Restyling it

The default stylesheet, `report.css`, is neutral and built on CSS custom properties. `--css FILE`
is the path of a UTF-8 stylesheet file, a regular file of at most 1 MiB. It is copied byte for byte
as `theme.css`, linked after the default, so it can override any variable or rule. `--out DIR`
writes the report somewhere else than `RUN/report/`. gelpy does not parse a stylesheet you pass,
so its references are yours: a relative `url()` resolves against the report folder, not against the
folder the stylesheet came from, and one that points outside the folder points outside it.

```bash
printf ':root { --gelpy-accent: #8250df; --gelpy-font: Georgia, serif; }\n' > "$RUN/theme.css"
gelpy report html "$RUN" --css "$RUN/theme.css" > /dev/null
grep -c 'href="theme.css"' "$RUN/report/index.html"
```

The variables and their defaults:
<!-- claim: html-report-css-variables -->

| variable | default | used for |
|---|---|---|
| `--gelpy-font` | `system-ui, ...` | body text |
| `--gelpy-mono` | `ui-monospace, ...` | provenance values |
| `--gelpy-bg` | `#ffffff` | page and image background |
| `--gelpy-fg` | `#1f2328` | text |
| `--gelpy-muted` | `#59636e` | captions, notes, provenance labels, the reasons under a lane |
| `--gelpy-border` | `#d1d9e0` | table lines, image frames |
| `--gelpy-surface` | `#f6f8fa` | table headers |
| `--gelpy-accent` | `#0969da` | the rule under each section heading and under the table headers |
| `--gelpy-ok` | `#1a7f37` | confidence `ok` |
| `--gelpy-low` | `#9a6700` | confidence `low`, and the warnings |
| `--gelpy-max-width` | `72rem` | width of the page column |

## Where it writes

`report html` writes into `RUN/report/`, or into the folder `--out` names. That folder has to be
new, empty, or hold a report an earlier `report html` wrote, which it knows by
`.gelpy-report.json`: the gelpy version, the files that build wrote, and a fingerprint (below). Any
other non-empty folder is refused with `invalid_value`, and the `detail` says to pass an empty or
new folder, or one an earlier `report html` wrote. That covers the `report/` of a run from before
1.1 that holds your own files.
A rebuild replaces the files its marker lists, removes those the new build did not write (the plot
of a fit that went away, a `theme.css` you no longer pass) and never touches a file the marker does
not list: a build that would write over an existing file its marker does not list is refused with
`invalid_value`, before it writes anything, and the message names the files. A marker that lists
nothing, or only some of the files, is no licence for the others.
A marker is a plain file that anyone can write, so it is never enough on its own. It may list only
names a build writes (`index.html`, `report.css`, `theme.css`, and PNGs directly in `img/` and
`img/fits/`, each file name at most 255 bytes): a folder whose marker lists anything else counts as
foreign. And whatever marker a folder carries, the run folder, a folder inside it other than its own
`report/`, a folder that holds the run folder, and a folder that holds a `recipe.json` or a
`manifest.json` are refused.
Folders and files are compared by what they are, not by how the path is typed: a link to the run
folder is the run folder, and on a file system that does not tell names apart by case a file the
marker lists as `img/LINEAR.png` is the file a build writes as `img/linear.png`. A `report/` that is a
link is refused (delete the link and repeat the command), whether it is the default `RUN/report` or
typed as `--out RUN/report`: it lies inside the run, and `status` reads it as the run's report. A
link you pass from outside the run is followed, and checked as the folder it points to.
<!-- claim: html-report-replaces-only-its-own-files -->
Because the tool replaces only its own earlier output, `report_html` is not marked destructive.

```bash
# doc-test: expect-failure
mkdir -p "$RUN-mine" && echo notes > "$RUN-mine/notes.txt"
gelpy report html "$RUN" --out "$RUN-mine"
```

```console
{
  "error": "The folder '.../run-mine' is not empty and no earlier report html wrote it (it has no valid .gelpy-report.json).",
  "code": "invalid_value",
  "detail": "Pass an empty or new folder, or one an earlier report html wrote.",
  "available": null
}
```

The run folder is refused too, with the marker or without:

```bash
# doc-test: expect-failure
gelpy report html "$RUN" --out "$RUN"
```

```console
{
  "error": "The folder '.../run' is the run folder: a report is not written there.",
  "code": "invalid_value",
  "detail": "Pass a new or empty folder outside the run folder, or leave out --out to use report/.",
  "available": null
}
```

One report per folder. The run lock covers the run, not `--out`: two builds into one folder at once
can interleave their files, so give each its own folder.

Every file the report writes, and every plot gelpy draws under `render/`, is written to a temporary
file next to its destination and renamed over it, so a reader never opens half a file and a
failed build leaves no temporary file. A build that is killed (`kill -9`, a power cut) can leave
one, named `.<file>.<random>.tmp`: such a file does not make a folder non-empty (nor does a
`.DS_Store`, which Finder leaves in a folder it has shown), and the next `report html` and
`init --force` remove the ones in the report folder (regular files of exactly that name, in the
folder, `img/` and `img/fits/`), best effort: one that cannot be removed (an immutable flag, a
read-only file) stays, is among the files `init --force` says it left, and stops neither. A link where a file belongs is replaced and what
it points at is left alone; a link where the report needs a folder, or a plot of the run that is a
link, is refused with `invalid_value`.

`init --force` removes a report the same careful way: only the files the marker lists and the
temporary files of killed builds, then the folders that became empty, then the marker. A `report/`
without a valid marker stays as it is (unless it holds nothing but such temporary files), a
`report` that is a link is unlinked and what it points to is not touched, and the `warnings` of the
summary say what was removed and what was left.

## When it is out of date

The page is a snapshot. Its Provenance carries a fingerprint: a short hash of the recipe, the gelpy
version, the stored fit configuration (`fit_config.json`, which the page lists for every lane,
fitted or not) and the fit of every lane the page shows. The marker holds the same value, and `status`
compares it with the run's: `report` is `{path, current}` for the run's own `report/`, or null when
there is none, and `current` is false after a relabel, a refit, any other change of the recipe or of
a lane's stored fit configuration, or a new gelpy version.
<!-- claim: html-report-fingerprint-follows-the-run -->
A build that stopped half way is not current either. The recipe is hashed as it is now, so a hand
edit of `recipe.json` counts too. Build the report again with `gelpy report html RUN`.

```bash
gelpy status "$RUN" | grep -A3 '"report"'
seq -f 'cond_%02g' 1 20 > "$RUN/labels.txt"
gelpy lanes set "$RUN" --labels-file "$RUN/labels.txt" > /dev/null
gelpy status "$RUN" | grep -A3 '"report"'
gelpy report html "$RUN" > /dev/null
gelpy status "$RUN" | grep -A3 '"report"'
```

```console
  "report": {
    "path": "report/index.html",
    "current": true
  },
  "report": {
    "path": "report/index.html",
    "current": false
  },
  "report": {
    "path": "report/index.html",
    "current": true
  },
```

The first read is `true`, the relabel made it `false`, and the rebuild made it `true` again.
