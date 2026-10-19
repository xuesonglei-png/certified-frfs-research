"""Signed witness-forest extension for a changed reduced metric.

Only reduced-nearest-enemy stability and nonsaturation are needed. Full
membership is bounded using the same enemy as the reduced witness. Signed
retained-coordinate coefficients couple the shared table across objects.
"""
import numpy as np
from .core import _inputs, interval_bounds, _distance, _interval_distances, _exact_stability_mask
from .redistributed_loss import redistributed_loss_bound
from .signed_forest import signed_forest_maximum


def coupled_redistributed_bound(X, y, features, w_full, w_reduced, *, eps=.01):
    X, y, fs, wf = _inputs(X, y, features, w_full); wr = np.asarray(w_reduced, float)
    base = redistributed_loss_bound(X, y, fs, wf, wr, eps=eps)
    n, d = X.shape
    if len(np.unique(y)) < 2 or 'redistributed_object_bounds' not in base:
        return {**base, 'coupled_signed_bound': base['upper_bound'], 'stable_unsaturated_count': 0}
    wide = np.zeros(d); wide[fs] = wr; lo, hi = interval_bounds(X, eps)
    cross = y[:, None] != y[None, :]; D = _distance(X, fs, wide)
    nearest = np.argmin(np.where(cross, D, np.inf), axis=1)
    stable, margins = _exact_stability_mask(lo, hi, nearest, fs, wide, cross)
    _, ub = _interval_distances(lo, hi, fs, wide, need_lower=False)
    selected = stable & (ub[np.arange(n), nearest] <= 1.-1e-12)
    edges = [(int(i), int(nearest[i])) for i in np.flatnonzero(selected)]
    coefficients = wf-wide; total = 0.; maxima = []; witnesses = []
    for a in range(d):
        if coefficients[a] == 0: maxima.append(0.); witnesses.append(X[:, a].tolist()); continue
        solved = signed_forest_maximum(lo[:, a], hi[:, a], [(i, j, coefficients[a]) for i, j in edges])
        total += solved['maximum']; maxima.append(solved['maximum']); witnesses.append(solved['witness'])
    remaining = float(np.asarray(base['redistributed_object_bounds'])[~selected].sum())
    signed = float(np.clip((total+remaining)/n+1e-10*(1+abs(total+remaining)), 0, 1))
    return {**base, 'upper_bound': min(base['upper_bound'], signed), 'objectwise_bound': base['upper_bound'],
        'coupled_signed_bound': signed, 'stable_unsaturated_count': int(selected.sum()),
        'stable_unsaturated_indices': np.flatnonzero(selected).tolist(), 'nearest_enemies': nearest.tolist(),
        'coordinate_maxima': maxima, 'forest_maximizer': np.asarray(witnesses).T.tolist(),
        'remaining_objectwise_sum': remaining,
        'scope': 'shared_table_signed_reduced_witness_forest_dependency_loss_not_test_risk_or_full_loss_exactness'}
