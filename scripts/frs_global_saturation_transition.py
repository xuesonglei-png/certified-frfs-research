"""Cell-free signed transition prototype; no fixed enemy or saturation rows.

This is an alternative sufficient region, not a dominance claim over shared
star bounds. The plain signed anchor bound may itself be infeasible. Always
keep that unavailability explicit; dropping star rows without replacing the
bound would be invalid.
"""
import numpy as np
from frs_saturation_transition_cell import evaluate_class_cell as saturation_value, append_class_cell_lp
from certified_frfs.core import _inputs, interval_bounds
from signed_classwise_dependency import signed_object_witness_bounds


def evaluate_class_cell(info, weights):
    value = saturation_value(info, weights)
    return {**value, 'scope':'global_shared_box_signed_class_dependency_bound_without_fixed_enemy_cell'}


def build_class_cell(A, y, features, anchor, full, *, eps=.01, eta=.1):
    A, y, fs, full = _inputs(A, y, features, full); fs = fs.tolist(); anchor = np.asarray(anchor, float)
    if (not fs or anchor.shape != (len(fs),) or not np.isfinite(anchor).all()
        or anchor.min() < 0 or not np.isfinite(eta) or not 0 <= eta <= 1):
        raise ValueError('valid global transition support/anchor/tolerance required')
    n, d = A.shape; wide = np.zeros(d); wide[fs] = anchor; identity = np.array_equal(wide, full)
    signed = np.zeros(n) if identity else np.asarray(signed_object_witness_bounds(A, y, fs, full, anchor, eps=eps)['signed_object_upper_bounds'])
    info = {'features':fs, 'anchor':anchor.tolist(), 'full_weights':full.tolist(),
        'anchor_identity':identity, 'signed_anchor_object_bounds':signed.tolist(), 'eps':eps, 'eta':eta}
    lo, hi = interval_bounds(A, eps); objects, classes = [], []
    for i in range(n):
        enemies = np.flatnonzero(y != y[i])
        low = np.maximum(0., np.maximum(lo[enemies][:, fs]-hi[i, fs], lo[i, fs]-hi[enemies][:, fs]))
        high = np.maximum(abs(lo[i, fs]-hi[enemies][:, fs]), abs(hi[i, fs]-lo[enemies][:, fs]))
        cap = min([1.]+[float(v@anchor) for v in high])
        objects.append({'object':i, 'enemies':enemies.tolist(), 'distance_lower':low.tolist(),
            'distance_upper':high.tolist(), 'anchor_membership_upper':cap, 'saturation_floor':cap-1.})
    for label in np.unique(y):
        own = np.flatnonzero(y == label); total = float(signed[own].sum()); raw = total/len(own)
        Q = np.sum([np.max(np.asarray(objects[i]['distance_upper']), axis=0)
                    if objects[i]['enemies'] else np.zeros(len(fs)) for i in own], axis=0)
        classes.append({'class':label.item(), 'count':len(own), 'mode':'signed_Lipschitz',
            'selected_indices':[], 'edges':[], 'H':[0.]*d, 'L':[0.]*d,
            'remaining_indices':own.tolist(), 'remaining_signed_anchor_sum':total,
            'omitted_constant':0., 'remaining_lipschitz_coefficients':Q.tolist(),
            'anchor_raw_bound':raw, 'inflation':1e-10*len(own)*(1+abs(raw))})
    info.update({'classes':classes, 'selected_objects':[], 'linear_cell_rows':[],
        'pair_transition_objects':objects, 'metric_envelope':'global_signed_saturation_transition_no_stability_cell',
        'scope':'global_same_table_signed_class_loss_bound_no_fixed_nearest_enemy_or_unsaturation_constraints'})
    info['anchor_evaluation'] = evaluate_class_cell(info, anchor)
    info['anchor_strictly_feasible'] = bool(all(v < eta for v in info['anchor_evaluation']['raw_upper_bounds_with_inflation']))
    return np.empty((0, len(fs))), np.empty(0), info
