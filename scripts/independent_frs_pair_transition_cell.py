"""Scalar reconstruction of the new residual pair-transition integration."""
import numpy as np
from independent_frs_one_sided_cell import reconstruct_cell as reconstruct_monotone_cell
from independent_frs_cell_stored_coordinates import scalar_cell_evaluation as stored_stability_check
from audit_forest_compact_transfer import equivalent


def reconstruct_cell(A, y, info, counts):
    C, rhs = reconstruct_monotone_cell(A, y, info, counts)
    assert info['metric_envelope'] == 'per_object_max_enemy_signed_pair_transition'
    expected_objects = sorted({i for r in info['classes'] for i in r['remaining_indices']})
    assert [r['object'] for r in info['pair_transition_objects']] == expected_objects
    for item in info['pair_transition_objects']:
        i = item['object']; enemies = [e for e in range(len(A)) if y[e] != y[i]]
        assert item['enemies'] == enemies
        lows, highs = [], []
        for e in enemies:
            low, high = [], []
            for a in info['features']:
                ilo, ihi = max(0., A[i, a]-.01), min(1., A[i, a]+.01)
                elo, ehi = max(0., A[e, a]-.01), min(1., A[e, a]+.01)
                low.append(max(0., elo-ihi, ilo-ehi))
                high.append(max(abs(ilo-ehi), abs(ihi-elo)))
            lows.append(low); highs.append(high)
        equivalent(lows, item['distance_lower']); equivalent(highs, item['distance_upper'])
        counts['independent_transition_enemy_pairs'] += len(enemies)
    return C, rhs


def scalar_cell_evaluation(info, weights):
    previous = stored_stability_check(info, weights); transitions = {}
    for item in info['pair_transition_objects']:
        scores = [sum((v-w)*(hi if v-w >= 0 else lo)
            for v, w, lo, hi in zip(info['anchor'], weights, lower, upper, strict=True))
            for lower, upper in zip(item['distance_lower'], item['distance_upper'], strict=True)]
        transitions[item['object']] = max([0.]+scores)
    raw = []
    for r in info['classes']:
        value = r['remaining_signed_anchor_sum']+r['omitted_constant']+r['inflation']
        value += sum(transitions[i] for i in r['remaining_indices'])
        if r['mode'] == 'shared_star':
            value += sum((info['full_weights'][a]-w)*(r['H'][a] if info['full_weights'][a]-w >= 0 else r['L'][a])
                         for a, w in zip(info['features'], weights, strict=True))
        raw.append(float(value/r['count']))
    return {**previous, 'raw_upper_bounds_with_inflation': raw, 'class_upper_bounds': np.clip(raw, 0, 1).tolist(),
        'feasible': bool(previous['cell_constraints_verified'] and all(v <= info['eta'] for v in raw))}


def independent_append_cell_lp(info, mass, p, n, append):
    total = 2*p+n; extras = []; objects = {r['object']: r for r in info['pair_transition_objects']}
    for r in info['classes']:
        cols, vals = [], []
        constant = r['remaining_signed_anchor_sum']+r['omitted_constant']+r['inflation']
        for i in r['remaining_indices']:
            variable = total; total += 1; extras.append((0, None)); cols.append(variable); vals.append(1.)
            for lows, highs in zip(objects[i]['distance_lower'], objects[i]['distance_upper'], strict=True):
                lows, highs = np.asarray(lows), np.asarray(highs)
                # Separate implementation with independently reconstructed
                # interval endpoints and recorded normalized-weight layout.
                row_cols = list(range(p))+list(range(p+n, 2*p+n))+[variable]
                row_values = (-.5*mass*(highs+lows)).tolist()+(.5*mass*(highs-lows)).tolist()+[-1.]
                intercept = -.5*float((highs+lows)@np.asarray(info['anchor']))
                append(row_cols, row_values, intercept)
        if r['mode'] == 'shared_star':
            for j, a in enumerate(info['features']):
                variable = total; total += 1; extras.append((None, None)); cols.append(variable); vals.append(1.)
                for slope in (r['H'][a], r['L'][a]):
                    append([j, variable], [-mass*slope, -1.], -info['full_weights'][a]*slope)
        append(cols, vals, info['eta']*r['count']-constant)
    return total, extras
