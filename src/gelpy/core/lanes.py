"""Lane positioning, detection, profile-width heuristic, and selector resolution."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from ..schemas.lanes import LaneLayout, LaneSelector
from ._registry import Registry
from .errors import LaneSelectionError, shown

# Fraction of the detected lane pitch used as the default per-lane profile-strip width.
_PROFILE_WIDTH_FRACTION = 0.7

# Fraction of the expected pitch (width/N) used as the default moving-window-average width
# applied to the projection before peak detection. A loaded well is a plateau, not a sharp
# peak, and real wells are often brighter on one edge than the other (comb-tooth shadow,
# meniscus, a bubble) — which side varies well-to-well, so it isn't a uniform crop-wide
# offset the grid fit can absorb, it's per-lane jitter. A fixed small-px blur does nothing
# to a plateau this wide; scaling the window to the (known) pitch collapses that per-well
# asymmetry before find_peaks ever runs. Validated on real gel images and re-measured
# 2026-09-10 on the synthetic suite + folding_screen:
# 0.7 gives composite 0.98 with a 0% false-ok rate and 0.019x-pitch error on folding_screen;
# 0.35 gives 0.94, a 2.67% FALSE-OK rate (it reports 'ok' while being wrong -- the one failure
# mode the suite forbids) and 0.044x-pitch on the same gel. Do not lower it without re-running
# `python -m tests.synthetic.harness`.
_SMOOTHING_WINDOW_FRACTION = 0.7

# --- matched-filter grid registration (see _matched_filter_pitch) -----------------------
# Search pitch over this fraction of the expected pitch (width/N); values validated on real
# gel data (a cleanly loaded gel and one with a faint right third) plus synthetic empty-well cases.
_PITCH_BAND = (0.6, 1.3)
_PITCH_PRIOR_FRACTION = 0.08  # Gaussian pitch-prior sigma, as a fraction of width/N
_PITCH_STEP = 0.25  # pitch search resolution (px)
_OFFSET_STEP = 1.0  # offset search resolution (px); the searched offset is discarded
# --- grid refine / confidence ------------------------------------------------------------
_REFINE_INLIER_FRACTION = 0.25  # keep only peaks this close (× pitch) to a node when refining
_COVERAGE_DISTANCE_FRACTION = 0.5  # a lane node counts as "covered" if a peak is within this × pitch
_MIN_COVERAGE = 0.6  # minimum covered-node fraction for "ok" confidence
_MAX_RESIDUAL_FRACTION = 0.3  # max matched-peak → node residual (× pitch) for "ok" confidence
_BAND_EDGE_FRACTION = 0.05  # flag "low" if the final pitch is within this fraction of a band edge
# --- adaptive projection strategy (see detect_lane_centers / projection_strategies) ------
# Minimum comb-SNR gain (in normalized 0..1 projection units) a candidate row-aggregation
# strategy must beat the full-height baseline by before it is adopted. Small and positive so
# a strategy that only ties full height (controls, partial_fill) never displaces it — the
# fallback keeps "never worse than full height" a hard guarantee, not a hope.
_ADAPTIVE_MARGIN = 0.02


def compute_lane_positions(left: float, right: float, n: int) -> list[float]:
    """Evenly spaced lane x-centers from ``left`` to ``right`` inclusive (``n`` lanes)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    return list(np.linspace(left, right, n))


def column_projection(
    image: np.ndarray, y_start: int | None = None, y_end: int | None = None
) -> np.ndarray:
    """Mean intensity per image column over an optional ``[y_start:y_end]`` row band.

    This is the orthogonal projection to ``line_profiles.extract_line_profile`` (which
    averages *within* a lane strip): here we collapse the y-axis so loaded lanes show up
    as bright peaks along x, which is what ``detect_lane_centers`` finds. The row band is
    exposed because restricting to the well region sometimes separates lanes more cleanly
    than the full height.

    A thresholded/height-weighted alternative (rejecting a column's bump unless it's
    tall/extended, not just bright) was tried and reverted: on real gel data it either
    saturated identically for a real lane and the trough beside it (destroying the very
    separation needed to place adjacent lanes), or its unbounded dynamic range let one
    unusually bright lane suppress detection of dimmer real ones once normalized. Plain
    mean, on real data, keeps a well-behaved, comparable-across-columns scale; a plateau's
    internal asymmetry (not dust) is instead handled by the pitch-scaled moving-window
    average in ``detect_lane_centers`` before peak detection, and dust/spike rejection by
    ``min_width`` below.
    """
    if image.ndim != 2:
        raise ValueError(f"expected a 2D image, got shape {image.shape}")
    band = image[y_start:y_end, :]
    if band.shape[0] == 0:
        raise ValueError(f"empty row band for y_start={y_start}, y_end={y_end}")
    return band.mean(axis=0).astype(np.float64)


@dataclass
class AdaptiveProjection:
    """What a projection strategy returns: the 1-D projection plus optional diagnostics.

    ``row_weights`` (per-row weights used to build the projection) and ``y_band`` (an explicit
    ``(y_start, y_end)`` window, or a list of windows for multi-band designs) are ``None`` for
    the plain full-height baseline; adaptive strategies (D1–D4) populate whichever they produce,
    for the overlay to surface.
    """

    projection: np.ndarray
    row_weights: np.ndarray | None = None
    y_band: object | None = None


@dataclass
class _CoreDetection:
    """Detection derived from a single 1-D projection (pre-fallback). Internal to this module."""

    centers: np.ndarray
    pitch: float
    profile_width: int
    peak_indices: np.ndarray
    n_detected: int
    band_edge_frac: float
    confidence: str
    reason: str
    blurred: np.ndarray


# name -> fn(image, n_lanes, *, expected_pitch, y_start, y_end) -> AdaptiveProjection.
# The row-aggregation is pluggable so each adaptive-band design (D1–D4) registers one strategy
# and is compared to the "full" baseline through the same comb-SNR fallback in
# detect_lane_centers. See docs/decisions/adaptive-band/.
projection_strategies: Registry = Registry("projection strategy")


@projection_strategies.register(
    "full", description="Full-height column mean — the baseline every strategy must beat."
)
def _strategy_full(
    image: np.ndarray,
    n_lanes: int,
    *,
    expected_pitch: float,
    y_start: int | None,
    y_end: int | None,
) -> AdaptiveProjection:
    return AdaptiveProjection(column_projection(image, y_start, y_end))


# --- D4: multi-band union (see docs/decisions/adaptive-band/D4-multiband-union.md) -----------------
# Light y-smoothing applied to the per-row comb weights before thresholding, as a fraction of
# the pass-1 pitch — mirrors D2's weight smoothing so a single noisy dropout row inside a real
# band doesn't fragment it before clustering even sees it.
_D4_WEIGHT_SMOOTH_FRACTION = 0.05
# Foreground-row threshold: median(score) + k * (1.4826 * MAD(score)) — a robust ("how many
# scaled-MADs above typical") cutoff rather than an absolute value, so it adapts to each
# image's own noise floor instead of being tuned to one image's brightness scale.
_D4_MAD_K = 3.0
# Rows separated by a gap <= this many px are merged into one candidate band — bridges a single
# dropout row inside one real band without merging genuinely distinct bands (a ladder cluster
# and a sample cluster are typically tens of px apart, far more than this).
_D4_GAP_MERGE_PX = 3
# A kept-set candidate must raise all-N node coverage by more than this to be accepted -- guards
# the greedy keep loop against churn from float noise sitting exactly on a coverage plateau.
_D4_COVERAGE_EPS = 1e-9


def _comb_row_scores(band: np.ndarray, n: int, pitch: float) -> np.ndarray:
    """Per-row comb-periodicity score: matched-filter response of an ``n``-tooth comb at
    ``pitch`` against each row, minus an anti-phase (half-pitch-shifted) comb, at the best offset.

    A row crossing a band lights up ~N columns spaced at the pitch, so the in-phase comb sums N
    band tips while the anti-phase comb samples the troughs between lanes -> large positive
    difference. A single dust spike lands on at most one tooth, a flat/DC row or wide smear
    samples the same mean in both phases (~0). This is the same discriminator D2 uses to
    separate real bands from dust/background; D4 reuses it as the per-row informativeness score
    that drives which rows become candidate y-bands.
    """
    rows, w = band.shape
    if pitch <= 0 or rows == 0 or w < 2:
        return np.zeros(rows, dtype=np.float64)
    k = np.arange(n, dtype=np.float64)
    offsets = np.arange(0.0, pitch, _OFFSET_STEP)
    if offsets.size == 0:
        offsets = np.zeros(1)
    on_grid = offsets[:, None] + k[None, :] * pitch  # (O, N)
    off_grid = on_grid + pitch / 2.0  # anti-phase comb

    def _sample_sum(grid: np.ndarray) -> np.ndarray:
        pos = np.clip(grid.ravel(), 0.0, w - 1.0)
        lo = np.floor(pos).astype(np.intp)
        hi = np.minimum(lo + 1, w - 1)
        frac = pos - lo
        vals = band[:, lo] * (1.0 - frac) + band[:, hi] * frac  # (R, O*N)
        return vals.reshape(rows, grid.shape[0], grid.shape[1]).sum(axis=2)  # (R, O)

    contrast = _sample_sum(on_grid) - _sample_sum(off_grid)  # (R, O)
    return contrast.max(axis=1)  # best offset per row


def _node_coverage(peak_indices: np.ndarray, centers: np.ndarray, pitch: float) -> float:
    """Fraction of the N grid nodes with a real detected peak nearby.

    Mirrors the ``cov_peak`` check inside ``_detection_confidence`` (same
    ``_COVERAGE_DISTANCE_FRACTION`` radius). Used as the greedy keep criterion below: a
    candidate band that only lights up a bright subset of lanes (e.g. dropping the
    ladder/scaffold rows on a multi-y gel like Cy3) does not raise coverage, so it is skipped
    even if its own comb score is high.
    """
    centers = np.asarray(centers, dtype=np.float64)
    if pitch <= 0 or centers.size == 0:
        return 0.0
    peaks = np.sort(np.asarray(peak_indices, dtype=np.float64))
    if peaks.size == 0:
        return 0.0
    covered = np.any(
        np.abs(centers[:, None] - peaks[None, :]) <= _COVERAGE_DISTANCE_FRACTION * pitch, axis=1
    )
    return float(covered.mean())


def _cluster_rows(mask: np.ndarray, gap_merge: int) -> list[tuple[int, int]]:
    """Contiguous True-runs in ``mask``, merging runs separated by a gap <= ``gap_merge`` rows.

    Returns a list of half-open ``(lo, hi)`` row ranges (indices into ``mask``), in ascending
    row order.
    """
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > gap_merge + 1)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [len(idx) - 1]))
    return [(int(idx[s]), int(idx[e]) + 1) for s, e in zip(starts, ends)]


@projection_strategies.register(
    "d4_union",
    description="Multi-band union — cluster rows into candidate y-bands by comb-periodicity, "
    "greedily keep the bands that add all-N lane coverage, and project over their union. "
    "Targets gels where different lanes carry bands at different y (e.g. ladder/scaffold vs "
    "samples), where no single y-window covers every lane.",
)
def _strategy_d4_union(
    image: np.ndarray,
    n_lanes: int,
    *,
    expected_pitch: float,
    y_start: int | None,
    y_end: int | None,
) -> AdaptiveProjection:
    """D4: cluster informative rows into candidate y-bands, keep the union that covers all N lanes.

    Pass-1 pitch comes from the full-height smoothed projection (mirrors ``_strategy_d2_comb``'s
    derivation), then every row in ``[y_start:y_end]`` is scored by ``_comb_row_scores`` — the
    same comb-periodicity discriminator D2 uses, reused here as the per-row informativeness
    measure the design doc calls for. Rows are thresholded to a foreground mask via a robust
    (median + k*MAD) cutoff and clustered into candidate y-bands (``_cluster_rows``, gap-merged
    so one noisy dropout row doesn't split a real band). Candidates are then added to a running
    union in descending mean-score order, keeping each only if it raises all-N node coverage
    (``_node_coverage``) — the guard that keeps a ladder/scaffold cluster AND a sample cluster
    both in the union on a multi-y gel (Cy3), rather than the union collapsing onto whichever
    cluster is brightest.

    If no row clears the threshold (a flat/uniform image), or the greedy loop keeps nothing,
    this falls back to the untouched full-height projection — the caller's comb-SNR fallback
    then decides adoption exactly as for every other strategy.
    """
    y0 = 0 if y_start is None else int(y_start)
    y1 = image.shape[0] if y_end is None else int(y_end)
    full_proj = column_projection(image, y_start, y_end)
    band_full = image[y0:y1, :]
    rows = band_full.shape[0]
    if rows == 0:
        raise ValueError(f"empty row band for y_start={y_start}, y_end={y_end}")

    span = float(full_proj.max() - full_proj.min())
    norm = (full_proj - full_proj.min()) / span if span > 0 else full_proj - full_proj.min()
    window = max(1, round(_SMOOTHING_WINDOW_FRACTION * expected_pitch))
    blurred = uniform_filter1d(norm, size=window)
    pitch, _ = _matched_filter_pitch(blurred, n_lanes, expected_pitch)

    scores = _comb_row_scores(band_full.astype(np.float64, copy=False), n_lanes, pitch)
    smooth = max(1, round(_D4_WEIGHT_SMOOTH_FRACTION * pitch)) if pitch > 0 else 1
    if smooth > 1:
        scores = uniform_filter1d(scores, size=smooth)

    med = float(np.median(scores))
    mad = float(np.median(np.abs(scores - med)))
    threshold = med + _D4_MAD_K * 1.4826 * mad
    mask = scores > threshold
    candidates = _cluster_rows(mask, _D4_GAP_MERGE_PX)
    if not candidates:
        return AdaptiveProjection(full_proj)

    candidates.sort(key=lambda lohi: float(scores[lohi[0]:lohi[1]].mean()), reverse=True)

    def _project_union(ranges: list[tuple[int, int]]) -> np.ndarray:
        rows_mask = np.zeros(rows, dtype=bool)
        for lo, hi in ranges:
            rows_mask[lo:hi] = True
        return band_full[rows_mask].mean(axis=0).astype(np.float64)

    kept: list[tuple[int, int]] = []
    best_coverage = 0.0
    for lo, hi in candidates:
        trial = kept + [(lo, hi)]
        proj = _project_union(trial)
        det = _detect_from_projection(
            proj,
            n_lanes,
            expected_pitch=expected_pitch,
            image_width=image.shape[1],
            smoothing_window=None,
            min_prominence=None,
            min_distance=None,
            min_width=None,
        )
        coverage = _node_coverage(det.peak_indices, det.centers, det.pitch)
        if coverage > best_coverage + _D4_COVERAGE_EPS:
            kept = trial
            best_coverage = coverage

    if not kept:
        return AdaptiveProjection(full_proj)

    kept.sort()
    rows_mask = np.zeros(rows, dtype=bool)
    for lo, hi in kept:
        rows_mask[lo:hi] = True
    projection = band_full[rows_mask].mean(axis=0).astype(np.float64)
    y_band = [(y0 + lo, y0 + hi) for lo, hi in kept]
    return AdaptiveProjection(projection, row_weights=rows_mask.astype(np.float64), y_band=y_band)


@dataclass
class LaneDetection:
    """Result of ``detect_lane_centers`` — the N fitted grid centers plus diagnostics."""

    centers: list[float]
    pitch: float
    profile_width: int
    projection: np.ndarray
    peak_indices: np.ndarray
    n_detected: int
    n_expected: int
    confidence: str  # "ok" | "low"
    confidence_reason: str = ""  # short human reason when confidence == "low"
    row_weights: np.ndarray | None = None  # per-row weights, if an adaptive strategy was used
    y_band: object | None = None  # chosen (y_start, y_end) window(s), if any
    projection_source: str = "full"  # which registered strategy produced ``projection``
    comb_snr: float = 0.0  # comb-SNR of the chosen projection (grid separability score)


def detect_lane_centers(
    image: np.ndarray,
    n_lanes: int,
    *,
    y_start: int | None = None,
    y_end: int | None = None,
    smoothing_window: float | None = None,
    min_prominence: float | None = None,
    min_distance: float | None = None,
    min_width: float | None = None,
    projection_strategy: str = "d4_union",
) -> LaneDetection:
    """Place ``n_lanes`` evenly-spaced lane centers by fitting a uniform grid to the image.

    Wells come from a comb, so lane centers lie on a uniform-pitch grid, and ``n_lanes``
    is ground truth (from the plate sheet / user) — never inferred here. Pitch is found by
    registering an ``n_lanes``-tooth comb against the y-projection (``_matched_filter_pitch``)
    with a soft prior toward ``width/n_lanes``; the offset is then anchored to the detected
    peaks. This exploits the two strong priors (N is known, spacing is uniform) instead of
    reading a pitch off noisy peak gaps — the median-gap estimate collapsed to a wrong pitch
    when faint lanes went undetected and the bright lanes clustered.

    Before peak detection, the projection is smoothed with a moving-window average sized to
    a fraction of the expected pitch (``smoothing_window``, see ``_SMOOTHING_WINDOW_FRACTION``):
    a loaded well is a plateau, not a sharp peak, and is often brighter on one edge than the
    other, which side varies well-to-well. A fixed small-px blur leaves that asymmetry intact
    and ``find_peaks`` locks onto the brighter edge instead of the well's true center; a
    window scaled to the (known) pitch collapses it first.

    The prior assumes the crop is roughly tight to the loaded lanes (true pitch within the
    search band around ``width/n_lanes``) — the usual case. Interior wells may be skipped
    (empty lanes yield no peak) and are handled. A gel that fills only a fraction of the crop
    fights the prior and is surfaced as ``confidence == "low"`` (with a reason) for the caller
    to resolve via manual endpoints, rather than silently mis-fit.

    ``projection_strategy`` selects how the 2-D image is collapsed to the 1-D projection the
    grid is fit to. ``"d4_union"`` (default; see ``_strategy_d4_union``) is only *adopted* if
    its comb-SNR beats the ``"full"`` (full-height column mean) baseline by ``_ADAPTIVE_MARGIN``
    and its confidence is not worse — otherwise detection falls back to full height, so the
    default is never worse than plain full-height projection.
    """
    if n_lanes < 1:
        raise ValueError("n_lanes must be >= 1")

    width = image.shape[1]
    expected_pitch = width / n_lanes

    def _detect(proj: np.ndarray) -> _CoreDetection:
        return _detect_from_projection(
            proj,
            n_lanes,
            expected_pitch=expected_pitch,
            image_width=width,
            smoothing_window=smoothing_window,
            min_prominence=min_prominence,
            min_distance=min_distance,
            min_width=min_width,
        )

    # Always compute the full-height baseline; it is both the default result and the yardstick
    # an adaptive strategy has to beat.
    full_ap = _strategy_full(
        image, n_lanes, expected_pitch=expected_pitch, y_start=y_start, y_end=y_end
    )
    full_det = _detect(full_ap.projection)
    full_snr = _comb_snr(full_ap.projection, full_det.centers, full_det.pitch)

    chosen_ap, chosen_det, chosen_snr, source = full_ap, full_det, full_snr, "full"
    if projection_strategy != "full":
        strategy = projection_strategies.get(projection_strategy)
        cand_ap = strategy(
            image, n_lanes, expected_pitch=expected_pitch, y_start=y_start, y_end=y_end
        )
        cand_det = _detect(cand_ap.projection)
        cand_snr = _comb_snr(cand_ap.projection, cand_det.centers, cand_det.pitch)
        # Adopt the candidate only if it is measurably crisper AND does not turn a correct
        # full-height "ok" into a "low" (never trade a right answer for a flagged one).
        confidence_regresses = cand_det.confidence == "low" and full_det.confidence == "ok"
        if cand_snr > full_snr + _ADAPTIVE_MARGIN and not confidence_regresses:
            chosen_ap, chosen_det, chosen_snr, source = (
                cand_ap,
                cand_det,
                cand_snr,
                projection_strategy,
            )

    return LaneDetection(
        centers=[float(c) for c in chosen_det.centers],
        pitch=float(chosen_det.pitch),
        profile_width=chosen_det.profile_width,
        projection=chosen_ap.projection,
        peak_indices=chosen_det.peak_indices,
        n_detected=chosen_det.n_detected,
        n_expected=n_lanes,
        confidence=chosen_det.confidence,
        confidence_reason=chosen_det.reason,
        row_weights=chosen_ap.row_weights,
        y_band=chosen_ap.y_band,
        projection_source=source,
        comb_snr=float(chosen_snr),
    )


def _detect_from_projection(
    proj: np.ndarray,
    n_lanes: int,
    *,
    expected_pitch: float,
    image_width: int,
    smoothing_window: float | None,
    min_prominence: float | None,
    min_distance: float | None,
    min_width: float | None,
) -> _CoreDetection:
    """Fit the uniform grid to a single 1-D projection (the strategy-agnostic detection core).

    This is a mechanical extraction of the normalize → smooth → ``find_peaks`` →
    ``_fit_uniform_grid`` → ``_detection_confidence`` pipeline so it can be run identically on
    the full-height projection and on any adaptive strategy's projection for a fair comb-SNR
    comparison. See ``detect_lane_centers`` for the rationale behind each step.
    """
    span = float(proj.max() - proj.min())
    norm = (proj - proj.min()) / span if span > 0 else proj - proj.min()
    # Round here, not only on the default branch: the public API, the request schema and the
    # CLI all type `smoothing_window` as a float (it is a pixel *width*, and a caller may well
    # scale it off a measured pitch), while `uniform_filter1d(size=...)` demands an int and
    # raises TypeError on any float -- including an integral one like 40.0.
    window = (
        smoothing_window
        if smoothing_window is not None
        else _SMOOTHING_WINDOW_FRACTION * expected_pitch
    )
    blurred = uniform_filter1d(norm, size=max(1, round(window)))

    # Peak-separation floor must NOT be tied to width/N: when the gel fills only part of
    # the crop the true pitch is far smaller, and a width-derived distance would merge
    # adjacent lanes (the very crop-dependence this feature exists to eliminate). A small
    # smoothing-scaled floor is safe — a lane projects to a single column bump regardless
    # of how many bands it holds, so there is nothing to over-split — while prominence
    # does the real work of rejecting inter-lane noise.
    distance = min_distance if min_distance is not None else max(1.0, window / 3.0)
    prominence = min_prominence if min_prominence is not None else 0.05
    # No pitch/sigma-scaled default here: on real gels, a genuine (if faint) band can be
    # just as narrow as a dust fleck — width alone doesn't reliably separate them across
    # different images, so guessing a "safe" scaled default risks silently rejecting real
    # lanes. Leave it off (1px, effectively a no-op) unless the caller has looked at the
    # overlay and knows the spurious peaks are narrower than the genuine ones on *this*
    # image specifically.
    width_floor = min_width if min_width is not None else 1.0

    peak_indices, _ = find_peaks(
        blurred, prominence=prominence, distance=distance, width=width_floor
    )
    n_detected = int(len(peak_indices))

    centers, pitch, band_edge_frac = _fit_uniform_grid(
        peak_indices, n_lanes, expected_pitch=expected_pitch, image_width=image_width,
        blurred=blurred,
    )
    profile_width = max(1, int(pitch * _PROFILE_WIDTH_FRACTION))

    confidence, reason = _detection_confidence(
        peak_indices, centers, pitch, n_lanes, n_detected, image_width, band_edge_frac
    )
    return _CoreDetection(
        centers=centers,
        pitch=float(pitch),
        profile_width=profile_width,
        peak_indices=peak_indices,
        n_detected=n_detected,
        band_edge_frac=band_edge_frac,
        confidence=confidence,
        reason=reason,
        blurred=blurred,
    )


def _comb_snr(proj: np.ndarray, centers: np.ndarray, pitch: float) -> float:
    """Grid separability of a projection: mean height at the N grid nodes minus the mean at
    the anti-phase midpoints (± half a pitch), on the min-max-normalized projection.

    The internal, ground-truth-free objective that gates the adaptive-strategy fallback.
    Because ``on`` averages over **all N** grid nodes, a strategy whose projection lights up
    only a bright subset of lanes (e.g. dropping the ladder/scaffold lanes on a multi-y gel)
    scores *lower*, not higher — that is what keeps a greedy adaptive band honest.
    """
    centers = np.asarray(centers, dtype=np.float64)
    span = float(proj.max() - proj.min())
    if span <= 0 or pitch <= 0 or centers.size == 0:
        return 0.0
    p = (proj - proj.min()) / span
    x = np.arange(len(p), dtype=np.float64)
    on = float(np.interp(centers, x, p).mean())
    off_positions = np.clip(
        np.concatenate([centers - pitch / 2.0, centers + pitch / 2.0]), 0.0, len(p) - 1.0
    )
    off = float(np.interp(off_positions, x, p).mean())
    return on - off


def _matched_filter_pitch(
    blurred: np.ndarray, n: int, expected_pitch: float
) -> tuple[float, float]:
    """Robust pitch via matched-filter grid registration with a soft ``width/n`` prior.

    Registers an ``n``-tooth comb against the (unchanged) smoothed projection at every
    ``(offset, pitch)`` on a coarse grid, scoring by the summed comb-tooth intensity times
    a Gaussian prior centered on ``expected_pitch = width/n``. Only the pitch of the
    best-scoring comb is used — the winning *offset* is discarded and re-derived from the
    detected peaks afterward (see ``_fit_uniform_grid``): empirically the offset landscape
    is close to degenerate, and a few empty interior wells can flip the matched-filter's
    own offset by half a pitch, whereas anchoring to peaks is robust to that.

    The prior is required, not optional polish: without it the raw (unweighted) score is
    monotonically biased toward the *smallest* pitch in the search band, because packing
    more teeth into whichever sub-region is brightest always raises the unweighted sum —
    this is what let a faint-lane gel's pitch collapse onto its brighter half in the old
    median-peak-gap estimator, and a matched filter without the prior reproduces the same
    failure (confirmed on both a faint-lane and a cleanly-loaded real gel).

    Returns ``(pitch, band_edge_frac)`` — the latter is 0 when the winning pitch sits
    exactly on a search-band edge (a sign the prior is fighting the data — e.g. the gel
    fills only a fraction of the crop) and 0.5 at the band center.
    """
    width = len(blurred)
    x = np.arange(width, dtype=np.float64)
    lo_pitch = _PITCH_BAND[0] * expected_pitch
    hi_pitch = _PITCH_BAND[1] * expected_pitch
    sigma_p = _PITCH_PRIOR_FRACTION * expected_pitch
    k = np.arange(n, dtype=np.float64)

    best_pitch, best_score = float(expected_pitch), -np.inf
    for pitch in np.arange(lo_pitch, hi_pitch + _PITCH_STEP, _PITCH_STEP):
        offsets = np.arange(-pitch, width + _OFFSET_STEP, _OFFSET_STEP)
        grid = offsets[:, None] + k[None, :] * pitch
        values = np.interp(grid.ravel(), x, blurred, left=0.0, right=0.0).reshape(grid.shape)
        raw_score = float(values.sum(axis=1).max())
        prior = float(np.exp(-0.5 * ((pitch - expected_pitch) / sigma_p) ** 2)) if sigma_p > 0 else 1.0
        score = raw_score * prior
        if score > best_score:
            best_score, best_pitch = score, float(pitch)

    band_width = hi_pitch - lo_pitch
    band_edge_frac = (
        min(best_pitch - lo_pitch, hi_pitch - best_pitch) / band_width if band_width > 0 else 0.5
    )
    return best_pitch, band_edge_frac


def _fit_uniform_grid(
    peak_indices: np.ndarray,
    n: int,
    *,
    expected_pitch: float,
    image_width: int,
    blurred: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Fit ``n`` evenly-spaced centers (offset + k*pitch) to the projection + peaks.

    Pitch comes from matched-filter grid registration against the smoothed projection,
    using the known ``n`` and a soft prior toward ``expected_pitch`` (see
    ``_matched_filter_pitch``) — robust to faint/undetected lanes that would otherwise skew
    a peak-gap-based pitch estimate. The grid origin is then anchored to the detected
    peaks: the discrete left-shift (0..N-1-intervals lanes) that best aligns the grid to
    them, refined by a gated least-squares fit (see ``_refine_grid_lsq_gated``) — so an
    empty *interior* lane simply lands on an unmatched grid point, and dust/noise peaks far
    from any grid node are excluded from the refine rather than dragging it off.

    Returns ``(centers, pitch, band_edge_frac)``.
    """
    if n == 1:
        c = float(peak_indices[0]) if len(peak_indices) else image_width / 2.0
        return np.array([c]), float(expected_pitch), 1.0

    peaks = np.sort(np.asarray(peak_indices, dtype=np.float64))
    if len(peaks) < 2:
        # Not enough structure to measure a pitch: spread N lanes at the expected pitch
        # centered on the single detected peak (or the image center).
        anchor = peaks[0] if len(peaks) else image_width / 2.0
        pitch = float(expected_pitch)
        left = anchor - pitch * (n - 1) / 2.0
        return left + np.arange(n) * pitch, pitch, 1.0

    pitch, band_edge_frac = _matched_filter_pitch(blurred, n, expected_pitch)
    intervals = int(round((peaks[-1] - peaks[0]) / pitch)) if pitch > 0 else n - 1
    grid_k = np.arange(n)

    # Anchor the grid coarsely: the detected span must fit inside it, and the only freedom
    # is how many lanes sit to the LEFT of the first detected peak (0..N-1-intervals).
    max_left_shift = max(0, (n - 1) - intervals)
    best_offset, best_cost, best_in_bounds = peaks[0], np.inf, False
    tie_eps = 1e-6 * max(pitch, 1.0)
    for m in range(max_left_shift + 1):
        offset = peaks[0] - m * pitch
        grid = offset + grid_k * pitch
        cost = float(np.sum(np.min(np.abs(peaks[:, None] - grid[None, :]), axis=1)))
        in_bounds = bool(np.all((grid >= 0) & (grid < image_width)))
        # Strict "<" with a tolerance: when the detected span covers fewer than N-1
        # intervals, every candidate m in range can score EXACTLY the same (the nearest-
        # grid-point distance for the detected peaks is invariant to a whole-pitch shift
        # of the grid) — a bare "<" then lets floating-point summation noise pick an
        # arbitrary, occasionally wildly-out-of-bounds m. Within that tie, prefer a shift
        # that keeps every center inside the image (a real lane cannot sit off-frame, so an
        # in-bounds candidate is strictly more plausible than an out-of-bounds one at equal
        # peak-fit cost — this is what correctly places a single empty *edge* well, e.g. an
        # unloaded lane 1, to the left instead of conjuring a phantom lane past the right
        # edge); otherwise resolved to the smallest m deterministically. Genuine ambiguity
        # is still surfaced by ``_detection_confidence``'s coverage/bounds checks, not
        # silently hidden.
        better = cost < best_cost - tie_eps
        tied = cost < best_cost + tie_eps
        if better or (tied and in_bounds and not best_in_bounds):
            best_cost, best_offset, best_in_bounds = cost, offset, in_bounds

    offset, pitch = _refine_grid_lsq_gated(peaks, best_offset, pitch)
    return offset + grid_k * pitch, pitch, band_edge_frac


def _refine_grid_lsq_gated(
    peaks: np.ndarray, offset: float, pitch: float
) -> tuple[float, float]:
    """Least-squares (offset, pitch) refine, gated against dust/noise peaks.

    Only peaks within ``_REFINE_INLIER_FRACTION`` of a pitch from their nearest coarse
    grid node are used: an ungated fit lets a handful of dust/artifact peaks (common on
    faint gels — real data measured 25 detected peaks for 20 real lanes) drag the whole
    grid off by tens of pixels, since polyfit assigns every peak to its nearest node
    regardless of how far that is. The refined result is only kept if it stays close to
    the coarse estimate and to a plausible pitch; otherwise the coarse values are kept.
    """
    if pitch <= 0:
        return offset, pitch
    k_all = np.round((peaks - offset) / pitch)
    residual = np.abs(peaks - (offset + k_all * pitch))
    inliers = residual <= _REFINE_INLIER_FRACTION * pitch
    k, p = k_all[inliers], peaks[inliers]
    if len(np.unique(k)) < 2:
        return offset, pitch
    slope, intercept = np.polyfit(k, p, 1)
    if slope <= 0:  # degenerate; keep the coarse estimate
        return offset, pitch
    if abs(intercept - offset) > 0.5 * pitch or abs(slope - pitch) > 0.2 * pitch:
        return offset, pitch  # refine moved implausibly far; distrust it
    return float(intercept), float(slope)


def _detection_confidence(
    peak_indices: np.ndarray,
    centers: np.ndarray,
    pitch: float,
    n: int,
    n_detected: int,
    image_width: int,
    band_edge_frac: float,
) -> tuple[str, str]:
    """Flag a detection as "low" confidence, with a short reason, when it warrants a
    manual look at the overlay.

    ``cov_peak`` (fraction of grid NODES with a nearby peak) and a residual gated to only
    plausibly-real peaks replace the old "every detected peak must sit near the grid"
    check: on a faint gel, dust/noise peaks legitimately sit far from any grid node
    without the fit being wrong, whereas the old fully-collapsed pitch could still have
    every *detected* peak close to the grid (all pulled from the same bright sub-region)
    while covering only a fraction of the true lane positions — coverage is what actually
    distinguishes the two.
    """
    if n == 1:
        return ("ok", "") if n_detected >= 1 else ("low", "no peak detected")
    if n_detected < max(2, n // 2):
        return "low", f"only {n_detected} peaks detected for {n} lanes"
    if n_detected > n + 2:
        return "low", f"{n_detected} peaks detected, expected {n} (+/- a few) — check for noise/dust"
    if pitch <= 0:
        return "low", "degenerate pitch (<=0)"
    if any(c < 0 or c >= image_width for c in centers):
        return "low", "a lane center fell outside the image"
    if band_edge_frac < _BAND_EDGE_FRACTION:
        return "low", (
            "matched-filter pitch pinned at the search-band edge — the prior may be "
            "fighting the data (e.g. the gel doesn't fill the crop); check endpoints"
        )

    peaks = np.sort(np.asarray(peak_indices, dtype=np.float64))
    node_covered = np.any(
        np.abs(centers[:, None] - peaks[None, :]) <= _COVERAGE_DISTANCE_FRACTION * pitch, axis=1
    )
    cov_peak = float(node_covered.mean())
    if cov_peak < _MIN_COVERAGE:
        return "low", f"only {cov_peak:.0%} of lane positions have a nearby detected peak"

    residual = np.min(np.abs(peaks[:, None] - centers[None, :]), axis=1)
    matched = residual <= _REFINE_INLIER_FRACTION * pitch
    max_residual = float(residual[matched].max()) if matched.any() else 0.0
    if max_residual > _MAX_RESIDUAL_FRACTION * pitch:
        return "low", "detected peaks don't sit close to the fitted grid"
    return "ok", ""


def guess_profile_width(n_lanes: int, image_width: int) -> int:
    """Heuristic per-lane strip width: ``image_width / (n_lanes * 2.5)``.

    The 2.5 factor is empirical and depends on the gel comb used; callers should
    override it when they know the true lane spacing.
    """
    if n_lanes < 1:
        raise ValueError("n_lanes must be >= 1")
    return max(1, int(image_width / (n_lanes * 2.5)))


def resolve_lane_selector(selector: LaneSelector, layout: LaneLayout) -> list[int]:
    """Resolve a selector to a sorted, de-duplicated list of lane indices.

    Grammar (see ``schemas.lanes``):
      - ``"all"``            -> every lane
      - ``int``              -> that index
      - ``"re:<pattern>"``   -> regex search against labels
      - ``"label:<text>"``   -> forced-literal label match
      - ``"<text>"``         -> literal label match

    Raises ``LaneSelectionError`` (listing available labels) if nothing matches or
    an index is out of range — replacing the old silent "return all lanes" fallback.
    """
    labels = layout.labels()
    n = len(labels)

    if selector == "all":
        return list(range(n))

    terms = selector if isinstance(selector, list) else [selector]

    indices: list[int] = []
    for term in terms:
        indices.extend(_resolve_term(term, labels, n))

    if not indices:
        raise LaneSelectionError(
            f"Lane selector {shown(selector)} matched no lanes.", available=labels
        )
    return sorted(set(indices))


def _resolve_term(term: int | str, labels: list[str], n: int) -> list[int]:
    if isinstance(term, bool):  # guard: bool is an int subclass
        raise LaneSelectionError(f"Invalid lane term {shown(term)}.", available=labels)
    if isinstance(term, int):
        if not (0 <= term < n):
            raise LaneSelectionError(
                f"Lane index {term} out of range (0..{n - 1}).", available=labels
            )
        return [term]
    if isinstance(term, str):
        if term.startswith("re:"):
            pattern = term[3:]
            try:
                regex = re.compile(pattern)
            except re.error as exc:
                raise LaneSelectionError(
                    f"Invalid regular expression {shown(pattern)} in a lane selector ({exc}).",
                    available=labels,
                ) from None
            return [i for i, label in enumerate(labels) if regex.search(label)]
        literal = term[6:] if term.startswith("label:") else term
        return [i for i, label in enumerate(labels) if label == literal]
    raise LaneSelectionError(f"Invalid lane term {shown(term)}.", available=labels)
