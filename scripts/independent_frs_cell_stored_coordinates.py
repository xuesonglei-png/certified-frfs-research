"""Scalar proof checks plus canonical stored-coordinate floating rechecks.

The independent scalar row sums can differ from batched BLAS at a floor
by one rounding unit. Floors are strictly positive buffers; mathematical
enemy stability requires positivity, and unsaturation requires <1. These
conditions are independently checked in longdouble. Class proofs are
checked before their positive numerical inflation without relaxing eta.
"""
import numpy as np
from independent_frs_class_cell import reconstruct_cell, independent_append_cell_lp


def scalar_cell_evaluation(info, weights):
    weights = np.asarray(weights, float); fs = info['features']; anchor = np.asarray(info['anchor'])
    full = np.asarray(info['full_weights']); stable = True; raw = []
    for item in info['selected_objects']:
        C = np.asarray(item['contrast']).reshape(-1, len(fs))
        floors = np.asarray(item['floors']); upper = np.asarray(item['unsaturation_coefficients'])
        canonical = bool((not len(C) or np.all(C@weights >= floors)) and upper@weights <= item['unsaturation_ceiling'])
        stable &= canonical
        if canonical:
            exactish = C.astype(np.longdouble)@weights.astype(np.longdouble)
            assert not len(C) or np.all(exactish > 0)
            assert upper.astype(np.longdouble)@weights.astype(np.longdouble) < 1
    for r in info['classes']:
        Q = np.asarray(r['remaining_lipschitz_coefficients'])
        value = r['remaining_signed_anchor_sum']+Q@abs(weights-anchor)
        coefficient = full[fs]-weights
        support = np.where(coefficient >= 0, np.asarray(r['H'])[fs], np.asarray(r['L'])[fs])
        if r['mode'] == 'shared_star':
            value += r['omitted_constant']+support@coefficient
        raw.append(float((value+r['inflation'])/r['count']))
        if raw[-1] <= info['eta']:
            extended = np.longdouble(r['remaining_signed_anchor_sum'])+Q.astype(np.longdouble)@abs(weights.astype(np.longdouble)-anchor.astype(np.longdouble))
            if r['mode'] == 'shared_star':
                extended += np.longdouble(r['omitted_constant'])+support.astype(np.longdouble)@(full[fs].astype(np.longdouble)-weights.astype(np.longdouble))
            assert extended/r['count'] <= info['eta']
    return {'raw_upper_bounds_with_inflation': raw, 'class_upper_bounds': np.clip(raw, 0, 1).tolist(),
        'cell_constraints_verified': bool(stable), 'feasible': bool(stable and all(v <= info['eta'] for v in raw)),
        'scope': 'shared_box_class_mean_dependency_bound_on_fixed_stable_cell'}
