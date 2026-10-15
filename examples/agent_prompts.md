# Prompts that work

gelpy is built to be driven by an agent, but "driven by an agent" is not a feature you can
verify by reading a README. So here are the prompts, what the agent does with each, and what
it hands back. Copy one, point it at your own gel.

These assume the `gelpy` plugin is installed (see [../plugin/README.md](../plugin/README.md))
and that `gelpy` is on PATH in the environment the agent runs commands in
([../docs/guide/install.md](../docs/guide/install.md)). The verified setup is Claude Code —
terminal or desktop app — working in a directory that has your images.

---

## The whole analysis, one prompt

```text
Analyze gels/my_gel.tif. It is a 20-lane agarose gel; the sample
sheet is in gels/labels.txt (left to right). I want the per-lane band fractions
and I want to know which lanes you would not trust.
```

What it does, and why in that order:

1. `gelpy init` — creates the run folder, records the image's content hash and its detected
   polarity.
2. `gelpy lanes detect --apply` — derives lane centers from the image, then **opens the
   overlay and looks at it**. This is the step that decides whether anything downstream is
   meaningful, and it is why the lane count belongs in your prompt: N is ground truth from
   your sheet, never something to infer.
3. `gelpy background remove` — mandatory before any profile or fit.
4. `gelpy fit run --plot` — fits every lane, then reads each lane's `confidence` and opens the
   fit figures for the ones flagged `low`.
5. `gelpy fit refit-lane` on the stragglers, climbing the strategy ladder.
6. `gelpy status` and `gelpy results show` — hands you the numbers and the run folder.

**Always give it the lane count and the sample sheet.** Without the count it has to guess, and
a guessed lane grid produces confident, wrong numbers. Without the labels you get lane indices
and have to map them yourself later.

---

## When you only want to look

```text
Give me a publication-ready image of gels/my_gel.tif with the lanes
labeled from gels/labels.txt. No fitting, no quantification.
```

You get a clean contrast-adjusted render with labels and no lane overlay drawn across the
sample. The red lane-width overlay is a QC view for checking where gelpy sliced; it is a
different image for a different purpose and should not be the one you put in a talk.

---

## Profiles without fitting

```text
Plot the line profiles for gels/purification.tif in groups of five lanes — one
plot per sample, each with its four fractions. Background-corrected, area
normalized. Don't fit peaks.
```

Common for purification gels. Two things the agent should get right and that you should check:
it must still remove the background first (otherwise each lane carries its own raw background
floor into an area-normalized plot, which distorts the baselines differently per lane), and it
must select lanes **by index**, because repeated fraction labels like `#3` match every group.

---

## Quantifying one species

```text
In runs/folding_screen, I care about the product band around profile pixel 875.
Give me its summed area fraction per lane as a CSV I can plot, and flag any lane
where that number is unreliable.
```

Worth knowing before you ask: a visually single band is often fitted as several overlapping
components, so the number you want is usually a **sum over a migration window**, not one row
of `results.csv`. On the example gel the product band is three components; the split between
them carries 5–34% relative uncertainty while the sum is solid. Asking for "the biggest band"
gets you a smaller, less stable number. See
[../docs/walkthrough/README.md](../docs/walkthrough/README.md#6-the-number-you-actually-want).

---

## Fixing a bad lane grid

```text
The overlay for runs/gel2 has the rightmost three lines off the lanes. The gel
only fills about the left two thirds of the crop. Fix the grid and re-fit only
what that changes.
```

This is the most common real failure, and it is a known limitation rather than a bug: the
grid fit assumes the lanes span roughly the whole crop, so a gel filling a fraction of the
frame fights that prior. The agent should read the outer lane centers off the overlay and set
them explicitly, then re-verify. Because the pipeline is content-hash cached, re-fitting after
a lane move recomputes only what actually changed.

---

## Checking an analysis someone handed you

```text
Someone gave me runs/screen_03. Tell me what was done to it, whether the numbers
are current, and anything you would not trust.
```

`gelpy status` answers most of this without recomputing anything: the source image, whether
background removal ran, the detected polarity and its confidence, and per lane whether a fit
exists, whether it is stale against the current recipe, and how it came out. Stale fits are
dropped from `results.csv`, so a run with `n_stale > 0` has numbers missing rather than wrong.

---

## What to be sceptical about

An agent will happily report numbers from a gel where the lane grid is wrong, because
`status: "ok"` only means the fit converged. Two things are worth insisting on in your prompt:

```text
...Show me the lane overlay and tell me explicitly whether every line sits on a
lane. Then show me the fit figure for any lane you flagged.
```

If it cannot show you the overlay, it did not check it.
