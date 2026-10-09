# Decision records

> **These documents are history, not documentation.** Each one records what was intended at a
> point in time. Several describe behaviour that shipped *differently*, and one describes a
> problem that is still open. **Never** treat a page in this directory as a description of how
> gelpy currently works — for that, use [../guide/](../guide/) and
> [../reference/](../reference/), both of which are checked against the code by tests.

They are kept because the *reasoning* is valuable and hard to reconstruct: why a low-order
polynomial baseline rather than arPLS, why the lane-grid prior is what it is, what was measured
before a default was chosen. When you change one of those decisions, the record of why it was
made the other way is the thing you want to read first.

## Where the risk is

An audit of this repository found three of these documents specifying features that had
shipped differently, and an implementation plan with none of its 58 steps ticked
despite being fully implemented and merged. A reader who wandered in could reasonably have
concluded that gelpy's baseline default was arPLS (it is an order-2 polynomial), or that the
adaptive lane projections were still four unbuilt candidate designs (one of them, `d4_union`,
is the default). Each page below now says at the top what its
status is, and `tests/docs/test_links.py` asserts that every document in this directory
labels itself.

## The records

| record | status |
|---|---|
| [2026-07-12-image-input-canonicalization-design.md](2026-07-12-image-input-canonicalization-design.md) | **Shipped.** The single-input-funnel design: any dtype/RGB/multi-page source to canonical float64 [0,1], signal-bright, with polarity detected and recorded. Current behaviour: [../guide/image-input.md](../guide/image-input.md). |
| [2026-07-12-image-input-canonicalization-plan.md](2026-07-12-image-input-canonicalization-plan.md) | **Shipped and merged**. An implementation plan, executed. Its unticked checkboxes mean nothing — do not work through them. |
| [2026-07-12-robust-background-fit-design.md](2026-07-12-robust-background-fit-design.md) | **Shipped** as `robust_plane`, now the default background model. Current behaviour: [../guide/background.md](../guide/background.md). |
| [2026-07-14-robust-peak-fitting-design.md](2026-07-14-robust-peak-fitting-design.md) | **Shipped with divergences**. Notably its step 1 specifies an arPLS baseline kept inside the joint fit; what shipped subtracts a conservative order-2 polynomial up front, because arPLS was measured absorbing ~34% of a broad smear. Current behaviour: [../guide/fit-engine.md](../guide/fit-engine.md). |
| [2026-07-15-baseline-redundancy-and-well-boundary.md](2026-07-15-baseline-redundancy-and-well-boundary.md) | **OPEN.** Two real problems found in use: the fit-time 1-D baseline may be redundant and net-harmful after a good 2-D subtraction, and the well/gel-start boundary is located by a fixed fraction of the window rather than detected from the image. Carries a 2026-09-10 addendum with the first measurements. Both are listed in [../guide/known-limitations.md](../guide/known-limitations.md). |
| [adaptive-band/](adaptive-band/) | **One of four designs shipped.** D4 (multi-band union) became the default projection strategy; D1, D2 and D3 were never built as standalone strategies, though D2's comb-periodicity row score shipped inside D4. The baseline numbers quoted in `00-context.md` are full-height numbers, which stopped being the default. Current behaviour and measured accuracy: [../guide/lane-placement.md](../guide/lane-placement.md). |
