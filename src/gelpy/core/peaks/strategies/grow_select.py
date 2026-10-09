"""Tier 1 — grow-and-select: the default, deterministic fit strategy.

Baseline -> greedy residual-driven growth (add the strongest residual bump, keep it only
if AICc improves) -> consolidation (merge over-split components into one smear) ->
per-component shape selection (local AICc) -> final joint robust-MAP refit -> assemble.

The peak count is *selected*, not tuned: the AICc stop replaces prominence/threshold knobs,
and overlapping peaks / shoulders separate over iterations because each round seeds on the
current residual.
"""

from __future__ import annotations

import numpy as np

from ....schemas.fit_config import FitConfig
from ....schemas.peaks import LaneFitResult
from .. import peak_registry
from ..detect import strongest_bump
from ..driver import Component, aicc, estimate_noise, evaluate_composite, fit_components
from ._common import assemble_result, compute_baseline
from .registry import fit_strategy_registry

_AICC_EPS = 2.0  # shape-selection: swap a shape only if AICc improves by at least this
_GROW_EPS = 6.0  # growth: a NEW component must improve AICc by at least this (suppress marginal splits)
_CONSOLIDATE_EPS = 4.0  # a merge is accepted if AICc worsens by no more than this (favor fewer)
_COLLAPSE_DIST = 2.0  # components whose maxima are within this many px are one feature (collapse)
_VALLEY_FILL = 0.8  # adjacent peaks whose between-valley fills >80% of the smaller are one hump


def _fit_kwargs(config: FitConfig, noise: float) -> dict:
    return dict(
        robust_loss=config.robust_loss,
        width_prior_strength=config.priors.width_prior_strength,
        tail_shrinkage=config.priors.tail_shrinkage,
        max_width_frac=config.priors.max_width_frac,
        noise_sigma=noise,
    )


def _clone(components: list[Component]) -> list[Component]:
    return [Component(c.model, c.params, c.seed_center, c.seed_scale) for c in components]


def _prune(components: list[Component], length: int, noise: float) -> list[Component]:
    """Drop degenerate components: below-noise height, or a maximum well off the profile."""
    kept: list[Component] = []
    for c in components:
        mx = c.model.peak_maximum_x(*c.params)
        width = max(c.model.effective_width(*c.params), 1.0)
        height = float(c.evaluate(np.array([mx]))[0])
        if height < noise:
            continue
        if mx < -2.0 * width or mx > (length - 1) + 2.0 * width:
            continue
        kept.append(c)
    return kept


def _greedy_grow(signal, x, config, seed_model, noise, kwargs):
    n = signal.size
    best = fit_components(signal, [], **kwargs)
    best_aicc = aicc(best.sse, n, 0)
    for _ in range(config.priors.max_components):
        resid = signal - evaluate_composite(best.components, x)
        bump = strongest_bump(resid, noise)
        if bump is None:
            break
        pos, scale, height, snr = bump
        if snr < config.priors.min_snr:
            break
        seed = Component(
            seed_model, tuple(seed_model.initial_guess(pos, height, scale)), float(pos), float(scale)
        )
        trial = _clone(best.components) + [seed]
        ft = fit_components(signal, trial, **kwargs)
        a = aicc(ft.sse, n, ft.n_params)
        if a < best_aicc - _GROW_EPS:
            best, best_aicc = ft, a
        else:
            break
    return best


def _consolidate(signal, x, components, config, models, noise, kwargs):
    """Merge adjacent components that form one smooth hump into a single component.

    The merged candidate is tried in *every* allowed shape and accepted only if AICc does not
    materially worsen. This is what distinguishes one tailed/broad feature (collapses to a
    single EMG / bi-Gaussian) from two genuinely separate bands (no single shape fits both, so
    the merge is rejected and the shoulder stays two peaks).
    """
    n = signal.size
    shape_models = list(models.values())
    changed = True
    while changed and len(components) > 1:
        changed = False
        order = sorted(range(len(components)), key=lambda i: components[i].model.peak_maximum_x(*components[i].params))
        components = [components[i] for i in order]
        cur = fit_components(signal, _clone(components), **kwargs)
        cur_aicc = aicc(cur.sse, n, cur.n_params)
        components = cur.components
        for i in range(len(components) - 1):
            ci, cj = components[i], components[i + 1]
            xi, xj = ci.model.peak_maximum_x(*ci.params), cj.model.peak_maximum_x(*cj.params)
            if xj <= xi:
                continue
            seg = np.arange(int(xi), int(xj) + 1, dtype=np.float64)
            pair = ci.evaluate(seg) + cj.evaluate(seg)
            hi, hj = ci.evaluate(np.array([xi]))[0], cj.evaluate(np.array([xj]))[0]
            valley = float(pair.min())
            if min(hi, hj) <= 0 or valley / min(hi, hj) < _VALLEY_FILL:
                continue  # a real dip separates them -> keep both
            center = 0.5 * (xi + xj)
            scale = max((xj - xi) / 2.0, ci.model.effective_width(*ci.params), cj.model.effective_width(*cj.params))
            height = max(hi, hj)
            rest = [c for k, c in enumerate(components) if k not in (i, i + 1)]
            best_ft = None
            best_a = np.inf
            for model in shape_models:
                merged = Component(model, tuple(model.initial_guess(center, height, scale)), center, scale)
                ft = fit_components(signal, _clone(rest) + [merged], **kwargs)
                a = aicc(ft.sse, n, ft.n_params)
                if a < best_a:
                    best_a, best_ft = a, ft
            if best_ft is not None and best_a <= cur_aicc + _CONSOLIDATE_EPS:
                components = best_ft.components
                changed = True
                break
    return components


def _select_shapes(signal, x, components, config, models, noise, kwargs):
    """Pick each component's shape by AICc, fitting one component at a time to the *full-trace*
    residual with the others frozen.

    Fitting over the full x (not a local window) still sees a component's tail where it
    overlaps a neighbor — the reason a local window fails — but costs a single-component fit
    instead of an all-component joint refit per trial, which is what keeps dense real lanes
    inside the compute budget. A final joint refit (in the caller) reconciles the choices.
    """
    n = signal.size
    if len(models) <= 1:
        return components
    current = _clone(components)
    for i in range(len(current)):
        others = evaluate_composite([c for k, c in enumerate(current) if k != i], x)
        local = signal - others
        ci = current[i]
        ctr = ci.model.peak_maximum_x(*ci.params)
        width = max(ci.model.effective_width(*ci.params), 1.0)
        height = float(max(ci.evaluate(np.array([ctr]))[0], 1e-9))
        base = fit_components(local, [_clone([ci])[0]], x=x, **kwargs)
        best_a = aicc(base.sse, n, base.n_params)
        best_model, best_params = ci.model, base.components[0].params
        for model in models.values():
            if model is ci.model:
                continue
            seed = Component(model, tuple(model.initial_guess(ctr, height, width)), ci.seed_center, ci.seed_scale)
            ft = fit_components(local, [seed], x=x, **kwargs)
            a = aicc(ft.sse, n, ft.n_params)
            if a < best_a - _AICC_EPS:
                best_a, best_model, best_params = a, model, ft.components[0].params
        current[i] = Component(best_model, best_params, ci.seed_center, ci.seed_scale)
    return current


def _collapse_duplicates(components: list[Component]) -> list[Component]:
    """Collapse components whose maxima sit within ``_COLLAPSE_DIST`` px into one.

    A sharp well (a ~1 px spike) can't be captured by a single parametric shape, so the greedy
    loop stacks several near-identical components on it; AICc-gated consolidation keeps them
    because two shapes fit the spike better than one. Two components at essentially the same
    position are unidentifiable regardless of AICc, so collapse them unconditionally (keeping
    the larger-area one; the final joint refit lets it absorb the rest). Safe for real
    doublets/shoulders, which are >> 2 px apart."""
    if len(components) <= 1:
        return components
    order = sorted(components, key=lambda c: c.model.peak_maximum_x(*c.params))
    kept = [order[0]]
    for comp in order[1:]:
        prev = kept[-1]
        if abs(comp.model.peak_maximum_x(*comp.params) - prev.model.peak_maximum_x(*prev.params)) < _COLLAPSE_DIST:
            if comp.area() > prev.area():
                kept[-1] = comp  # keep the larger; drop the smaller duplicate
        else:
            kept.append(comp)
    return kept


def refine(signal, x, components, config, models, noise, kwargs):
    """Shared post-seed refinement: consolidate -> shape-select -> collapse -> prune -> joint refit.

    Both grow_select (greedy seeds) and sparse_seed (NNLS seeds) funnel through this so the
    tiers differ only in how components are proposed, never in how they are finalized.
    """
    if components:
        components = _consolidate(signal, x, components, config, models, noise, kwargs)
        components = _select_shapes(signal, x, components, config, models, noise, kwargs)
        components = _collapse_duplicates(components)
        components = _prune(components, signal.size, noise)
    final = fit_components(signal, _clone(components), **kwargs)
    pruned = _prune(final.components, signal.size, noise)
    if len(pruned) != len(final.components):
        final = fit_components(signal, _clone(pruned), **kwargs)
    return final


def prepare(profile: np.ndarray, config: FitConfig):
    """Baseline-subtract and set up the shared per-lane fit context."""
    profile = np.asarray(profile, dtype=np.float64)
    baseline = compute_baseline(profile, config.baseline)
    signal = profile - baseline
    x = np.arange(signal.size, dtype=np.float64)
    noise = estimate_noise(signal)
    models = {name: peak_registry.get(name)() for name in config.shapes}
    return profile, baseline, signal, x, noise, models, _fit_kwargs(config, noise)


@fit_strategy_registry.register("grow_select", description="Residual-driven grow-and-select (default).")
def fit(profile: np.ndarray, config: FitConfig, *, lane_index: int, label: str) -> LaneFitResult:
    profile, baseline, signal, x, noise, models, kwargs = prepare(profile, config)
    seed_model = models.get("gaussian") or next(iter(models.values()))
    best = _greedy_grow(signal, x, config, seed_model, noise, kwargs)
    final = refine(signal, x, best.components, config, models, noise, kwargs)
    return assemble_result(
        profile, baseline, final, config, lane_index=lane_index, label=label, method="grow_select"
    )
