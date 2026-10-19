"""One-sided metric envelope exploiting FRS membership monotonicity.

For every shared table Z and nonnegative metric w,
rho_full(Z)-rho_w(Z) <= r_anchor+Q @ (anchor-w)_+.
Only decreases can lower min(1,min_enemy weighted L1 distance).
Shared-star terms and all stability/unsaturation conditions are unchanged.
"""
import numpy as np
from frs_class_cell import build_class_cell as build_symmetric_cell
from frs_class_cell import evaluate_class_cell as evaluate_symmetric_cell


def evaluate_class_cell(info, weights):
    weights = np.asarray(weights, float); anchor = np.asarray(info['anchor']); fs = info['features']
    # Retain input and fixed-cell validation from the independently tested
    # symmetric evaluator, then tighten only its remaining-object envelope.
    report = evaluate_symmetric_cell(info, weights)
    decreases = np.maximum(anchor-weights, 0.)
    full = np.asarray(info['full_weights']); raw = []
    for r in info['classes']:
        value = r['remaining_signed_anchor_sum']+r['inflation']+r['omitted_constant']
        value += np.asarray(r['remaining_lipschitz_coefficients'])@decreases
        if r['mode'] == 'shared_star':
            coefficient = full[fs]-weights
            value += np.where(coefficient >= 0, np.asarray(r['H'])[fs], np.asarray(r['L'])[fs])@coefficient
        raw.append(float(value/r['count']))
    return {**report, 'raw_upper_bounds_with_inflation': raw, 'class_upper_bounds': np.clip(raw, 0, 1).tolist(),
        'feasible': bool(report['cell_constraints_verified'] and all(v <= info['eta'] for v in raw))}


def build_class_cell(A, y, features, anchor, full, *, eps=.01, eta=.1):
    C, rhs, info = build_symmetric_cell(A, y, features, anchor, full, eps=eps, eta=eta)
    info['metric_envelope'] = 'Q_times_coordinate_weight_decreases_only'
    info['anchor_evaluation'] = evaluate_class_cell(info, anchor)
    return C, rhs, info


def append_class_cell_lp(info, mass, p, n, append):
    # (anchor-w)_+ = (abs(w-anchor)+anchor-w)/2.
    # Existing L1 epigraph u >= abs(w_normalized-anchor_normalized) is shared
    # without introducing new bounds or changing the objective penalty.
    offset = 2*p+n; extras = []; full = np.asarray(info['full_weights']); fs = info['features']
    for r in info['classes']:
        Q = np.asarray(r['remaining_lipschitz_coefficients'])
        cols = list(range(p+n, 2*p+n))+list(range(p))
        vals = (.5*mass*Q).tolist()+(-.5*mass*Q).tolist()
        constant = r['remaining_signed_anchor_sum']+r['omitted_constant']+r['inflation']+.5*Q@np.asarray(info['anchor'])
        if r['mode'] == 'shared_star':
            for j, a in enumerate(fs):
                z = offset+len(extras); extras.append((None, None)); cols.append(z); vals.append(1.)
                for slope in (r['H'][a], r['L'][a]):
                    append([j, z], [-mass*slope, -1.], -full[a]*slope)
        append(cols, vals, info['eta']*r['count']-constant)
    return offset+len(extras), extras
