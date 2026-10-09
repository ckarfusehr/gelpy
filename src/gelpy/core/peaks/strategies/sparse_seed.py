"""Tier 2 - sparse_seed: seed components by non-negative deconvolution, then refine.

For heavily overlapped lanes where grow_select's residual-argmax seeding is ambiguous, this
proposes components objectively (no explicit N): it solves a non-negative least-squares
deconvolution of the signal against a dictionary of shifted, multi-width Gaussian atoms
(``scipy.optimize.nnls``), which yields a sparse non-negative weight map. Clusters of weight
become seed ``(position, scale, height)`` triples, which are handed to the shared
grow_select refinement (consolidate -> shape-select -> prune -> joint refit).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

from ....schemas.fit_config import FitConfig
from ....schemas.peaks import LaneFitResult
from ..driver import Component
from ._common import assemble_result
from .grow_select import prepare, refine
from .registry import fit_strategy_registry

_N_WIDTHS = 3
_MAX_ATOMS = 240  # cap dictionary size to keep NNLS within budget


def _dictionary(x: np.ndarray, widths: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Unit-height Gaussian atoms at each (position, width); columns of the design matrix."""
    cols = [
        np.exp(-((x - p) ** 2) / (2.0 * w**2))
        for w in widths
        for p in positions
    ]
    return np.stack(cols, axis=1)


def _nnls_seeds(signal, x, noise, config) -> list[tuple[float, float, float]]:
    n = signal.size
    widths = np.geomspace(1.5, max(4.0, n / 8.0), _N_WIDTHS)
    step = max(2, int(np.ceil(_N_WIDTHS * n / _MAX_ATOMS)))
    positions = np.arange(0, n, step, dtype=np.float64)
    meta = [(p, w) for w in widths for p in positions]
    design = _dictionary(x, widths, positions)

    weights, _ = nnls(design, np.clip(signal, 0.0, None), maxiter=5 * design.shape[1])
    thresh = max(weights.max() * 0.02, 0.0) if weights.size else 0.0
    active = [(meta[i][0], meta[i][1], w) for i, w in enumerate(weights) if w > thresh]
    if not active:
        return []

    # Cluster active atoms by position (within ~1 atom-width) into seeds.
    active.sort(key=lambda a: a[0])
    seeds: list[tuple[float, float, float]] = []
    cluster = [active[0]]
    for pos, width, wgt in active[1:]:
        if pos - cluster[-1][0] <= max(width, cluster[-1][1]):
            cluster.append((pos, width, wgt))
        else:
            seeds.append(_seed_from_cluster(cluster, signal))
            cluster = [(pos, width, wgt)]
    seeds.append(_seed_from_cluster(cluster, signal))
    # Gate on the local signal height vs noise (drop noise-level clusters).
    return [(p, s, h) for (p, s, h) in seeds if h >= config.priors.min_snr * noise]


def _seed_from_cluster(cluster, signal) -> tuple[float, float, float]:
    wsum = sum(w for _p, _s, w in cluster) or 1.0
    pos = sum(p * w for p, _s, w in cluster) / wsum
    scale = sum(s * w for _p, s, w in cluster) / wsum
    height = float(max(signal[int(round(pos))], 0.0))
    return float(pos), float(scale), height


@fit_strategy_registry.register("sparse_seed", description="Non-negative-deconvolution seeding (Tier 2).")
def fit(profile: np.ndarray, config: FitConfig, *, lane_index: int, label: str) -> LaneFitResult:
    profile, baseline, signal, x, noise, models, kwargs = prepare(profile, config)
    gauss = models.get("gaussian") or next(iter(models.values()))
    seeds = _nnls_seeds(signal, x, noise, config)[: config.priors.max_components]
    components = [
        Component(gauss, tuple(gauss.initial_guess(p, h, s)), p, s) for (p, s, h) in seeds
    ]
    final = refine(signal, x, components, config, models, noise, kwargs)
    return assemble_result(
        profile, baseline, final, config, lane_index=lane_index, label=label, method="sparse_seed"
    )
