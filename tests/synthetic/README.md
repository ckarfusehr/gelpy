# Synthetic gel suite + evaluation harness

Ground-truth synthetic gels plus an evaluation harness for **lane detection**. Originally
built to rank four adaptive y-band designs against the full-height baseline; D4 won and
shipped as the default (`d4_union`), so the harness's job now is to keep it honest
and to score any future design against a fixed reference.

## Why
Full-height column projection (gelpy's original default, still the reference and the
per-image fallback) dilutes a thin band's signal with empty rows.
This suite models realistic **y-localized** bands (wide-in-x, thin-in-y streaks at a migration
depth) with ground-truth centers, so adaptive-band designs can be scored honestly.

## Layout
- `generator.py` — seeded parametric gel generator → `SynthGel(image, centers, pitch, band_y…)`.
- `conditions.py` — the `SUITE` of named `GelSpec`s (SNR ladder + pattern/geometry cases).
- `harness.py` — `evaluate({name: fn})` scores detection fns on accuracy **and** confidence.
- `real.py` + `real_labels.json` — hand-labeled real gels (`examples/data/*.tif`).
- `render_samples.py` — contact-sheet PNG for visual review.
- `test_suite.py` — determinism / well-posedness guardrails.

## Run
```
python -m tests.synthetic.harness          # full-height reference AND shipped default
python -m tests.synthetic.real             # baseline on the real TIFFs (skips if absent)
python -m tests.synthetic.render_samples out.png
pytest tests/synthetic/test_suite.py
```

## Metric (see `harness.py`)
A run **passes** when max per-lane center error ≤ `TOL` (0.25 × pitch). The `confidence` flag is
**correct** when `ok` ⇔ pass. Tracked separately: `false_ok` (says ok but wrong — silent error,
worst) and `false_low` (accurate but flagged — needless manual fallback). Composite =
`0.5·pass_rate + 0.5·flag_correct_rate − false_ok_rate`.

## Snapshot (8 seeds/condition, both rows printed by `python -m tests.synthetic.harness`)

| detector | medErr | pass | flagOK | falseOK | falseLow | score |
|---|---|---|---|---|---|---|
| `full_reference` (`projection_strategy="full"`) | 0.04p | 94% | 82% | 0% | 18% | 0.88 |
| `shipped_default` (all defaults, i.e. `d4_union`) | 0.02p | 96% | 100% | 0% | 0% | 0.98 |

**Read the detector label, not just the numbers.** Full height is the fixed *reference* a new
design must beat; it stopped being the default, when `d4_union` took over (adopted
per-image only when its comb-SNR beats full height, so it is never worse). An earlier version
of this snapshot recorded the full-height numbers while claiming to describe the default, and
`harness.baseline_detect` had meanwhile drifted to inherit the module default — so both
`test_d4_harness_composite_at_least_matches_baseline` and
`test_baseline_never_silently_wrong_and_has_headroom` were comparing the shipped detector
against itself. `baseline_detect` now pins `projection_strategy="full"` and must stay pinned.

`d4_union` closed **all** the headroom full height had left: `snr_015`, `snr_020`,
`narrowband_hard` (0.50 → 1.00), `gradient_bg` (0.50 → 1.00), `empty_edge_and_interior`
(0.69 → 1.00), `empty_edge`, `smile`, `n8_wide` (0.94 → 1.00) — every false-low is gone.

Guardrails a design must not break: `falseOK` stays 0 (never silently wrong); `partial_fill`
stays flagged low (it is the deliberate known limitation — a gel filling only part of the crop
fights the width/N prior, and both detectors get it wrong at 4.64p while correctly saying so);
`control_*` stay pass+ok. Real `folding_screen`: 4.44 px = 0.019×pitch, `ok`.
