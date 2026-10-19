"""Independent monotone bounds with actual-coordinate numerical checks."""
import numpy as np
from independent_frs_one_sided_cell import reconstruct_cell, independent_append_cell_lp
from independent_frs_cell_stored_coordinates import scalar_cell_evaluation as symmetric_check


def scalar_cell_evaluation(info, weights):
    weights = np.asarray(weights, float); fs = info['features']; anchor = np.asarray(info['anchor'])
    full = np.asarray(info['full_weights']); symmetric = symmetric_check(info, weights)
    decreases = np.maximum(anchor-weights, 0.); raw = []
    for r in info['classes']:
        Q = np.asarray(r['remaining_lipschitz_coefficients'])
        value = r['remaining_signed_anchor_sum']+r['inflation']+r['omitted_constant']
        value += Q@decreases; coefficient = full[fs]-weights
        support = np.where(coefficient >= 0, np.asarray(r['H'])[fs], np.asarray(r['L'])[fs])
        if r['mode'] == 'shared_star':
            value += support@coefficient
        raw.append(float(value/r['count']))
        if raw[-1] <= info['eta']:
            extended = np.longdouble(r['remaining_signed_anchor_sum'])+np.longdouble(r['omitted_constant'])
            extended += Q.astype(np.longdouble)@np.maximum(anchor.astype(np.longdouble)-weights.astype(np.longdouble), 0.)
            if r['mode'] == 'shared_star':
                extended += support.astype(np.longdouble)@(full[fs].astype(np.longdouble)-weights.astype(np.longdouble))
            assert extended/r['count'] <= info['eta']
    return {**symmetric, 'raw_upper_bounds_with_inflation': raw, 'class_upper_bounds': np.clip(raw, 0, 1).tolist(),
        'feasible': bool(symmetric['cell_constraints_verified'] and all(v <= info['eta'] for v in raw))}
