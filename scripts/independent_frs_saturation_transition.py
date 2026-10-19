"""Independent scalar signed-saturation geometry/evaluation/LP prototype."""
import numpy as np
from independent_frs_pair_transition_cell import reconstruct_cell as pair_geometry
from independent_frs_pair_transition_cell import scalar_cell_evaluation as pair_evaluation
from audit_forest_compact_transfer import equivalent


def reconstruct_cell(A, y, info, counts):
    assert info['metric_envelope'] == 'signed_enemy_transition_with_fixed_anchor_saturation_floor'
    original = {**info, 'metric_envelope':'per_object_max_enemy_signed_pair_transition'}
    C, rhs = pair_geometry(A, y, original, counts)
    for item in info['pair_transition_objects']:
        hi_caps = [sum(u*v for u, v in zip(upper, info['anchor'], strict=True)) for upper in item['distance_upper']]
        cap = min([1.]+hi_caps)
        equivalent(item['anchor_membership_upper'], cap); equivalent(item['saturation_floor'], cap-1.)
        assert -1 <= item['saturation_floor'] <= 0
        counts['independent_saturation_floors'] += 1
    return C, rhs


def scalar_cell_evaluation(info, weights):
    previous = pair_evaluation(info, weights); t = {}
    for item in info['pair_transition_objects']:
        scores = [sum((v-w)*(hi if v-w >= 0 else lo)
            for v, w, lo, hi in zip(info['anchor'], weights, lower, upper, strict=True))
            for lower, upper in zip(item['distance_lower'], item['distance_upper'], strict=True)]
        t[item['object']] = max([item['saturation_floor']]+scores)
    raw = []
    for r in info['classes']:
        total = r['remaining_signed_anchor_sum']+r['omitted_constant']+r['inflation']
        total += sum(t[i] for i in r['remaining_indices'])
        if r['mode'] == 'shared_star':
            total += sum((info['full_weights'][a]-w)*(r['H'][a] if info['full_weights'][a]-w >= 0 else r['L'][a])
                for a, w in zip(info['features'], weights, strict=True))
        raw.append(float(total/r['count']))
    return {**previous, 'raw_upper_bounds_with_inflation':raw,
        'class_upper_bounds':np.clip(raw, 0, 1).tolist(),
        'feasible':bool(previous['cell_constraints_verified'] and all(v <= info['eta'] for v in raw))}


def independent_append_cell_lp(info, mass, p, n, add):
    total, extras = 2*p+n, []; objects = {r['object']:r for r in info['pair_transition_objects']}
    for r in info['classes']:
        cols, values = [], []
        constant = r['remaining_signed_anchor_sum']+r['omitted_constant']+r['inflation']
        for i in r['remaining_indices']:
            item = objects[i]; z = total; total += 1
            extras.append((item['saturation_floor'], None)); cols.append(z); values.append(1.)
            for lower, upper in zip(item['distance_lower'], item['distance_upper'], strict=True):
                lo, hi = np.asarray(lower), np.asarray(upper)
                add(list(range(p))+list(range(p+n, 2*p+n))+[z],
                    (-.5*mass*(hi+lo)).tolist()+(.5*mass*(hi-lo)).tolist()+[-1.],
                    -.5*float((hi+lo)@np.asarray(info['anchor'])))
        if r['mode'] == 'shared_star':
            for j, a in enumerate(info['features']):
                z = total; total += 1; extras.append((None, None)); cols.append(z); values.append(1.)
                for slope in (r['H'][a], r['L'][a]):
                    add([j, z], [-mass*slope, -1.], -info['full_weights'][a]*slope)
        add(cols, values, info['eta']*r['count']-constant)
    return total, extras
