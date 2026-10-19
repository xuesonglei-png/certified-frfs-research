"""Independent scalar global bounds; no discarded star condition is assumed."""
import numpy as np
from independent_signed_classwise import SignedWitnessValidator
from independent_frs_saturation_transition import scalar_cell_evaluation as saturation_value, independent_append_cell_lp
from audit_forest_compact_transfer import equivalent


def scalar_cell_evaluation(info, weights):
    value = saturation_value(info, weights)
    return {**value, 'scope':'global_shared_box_signed_class_dependency_bound_without_fixed_enemy_cell'}


def reconstruct_cell(A, y, info, counts):
    A, y = np.asarray(A, float), np.asarray(y); n, d = A.shape
    assert info['eps'] == .01 and info['eta'] == .1
    assert info['metric_envelope'] == 'global_signed_saturation_transition_no_stability_cell'
    assert info['selected_objects'] == info['linear_cell_rows'] == []
    fs, anchor, full = info['features'], np.asarray(info['anchor']), np.asarray(info['full_weights'])
    wide = np.zeros(d); wide[fs] = anchor; identity = np.array_equal(wide, full)
    assert info['anchor_identity'] == identity
    validator = SignedWitnessValidator(A, y, full)
    signed = [0. if identity or not any(y != y[i]) else validator.object(i, fs, anchor)['upper_bound'] for i in range(n)]
    equivalent(signed, info['signed_anchor_object_bounds'])
    assert len(info['pair_transition_objects']) == n; Q = []
    for i, item in enumerate(info['pair_transition_objects']):
        assert item['object'] == i
        enemies = [e for e in range(n) if y[e] != y[i]]; assert item['enemies'] == enemies
        lows, highs = [], []
        for e in enemies:
            lower, upper = [], []
            for a in fs:
                il, ih = max(0., A[i, a]-.01), min(1., A[i, a]+.01)
                el, eh = max(0., A[e, a]-.01), min(1., A[e, a]+.01)
                lower.append(max(0., el-ih, il-eh)); upper.append(max(abs(il-eh), abs(ih-el)))
            lows.append(lower); highs.append(upper)
        equivalent(lows, item['distance_lower']); equivalent(highs, item['distance_upper'])
        cap = min([1.]+[sum(v*w for v, w in zip(row, anchor, strict=True)) for row in highs])
        equivalent(cap, item['anchor_membership_upper']); equivalent(cap-1., item['saturation_floor'])
        Q.append([max([0.]+[row[j] for row in highs]) for j in range(len(fs))])
        counts['independent_global_transition_enemy_pairs'] += len(enemies)
    for label, row in zip(np.unique(y), info['classes'], strict=True):
        own = [i for i in range(n) if y[i] == label]; total = sum(signed[i] for i in own); raw = total/len(own)
        expected = {'class':label.item(), 'count':len(own), 'mode':'signed_Lipschitz',
            'selected_indices':[], 'edges':[], 'H':[0.]*d, 'L':[0.]*d, 'remaining_indices':own,
            'remaining_signed_anchor_sum':total, 'omitted_constant':0.,
            'remaining_lipschitz_coefficients':[sum(Q[i][j] for i in own) for j in range(len(fs))],
            'anchor_raw_bound':raw, 'inflation':1e-10*len(own)*(1+abs(raw))}
        equivalent(expected, row)
    value = scalar_cell_evaluation(info, anchor); equivalent(value, info['anchor_evaluation'])
    assert info['anchor_strictly_feasible'] == bool(all(v < .1 for v in value['raw_upper_bounds_with_inflation']))
    return np.empty((0, len(fs))), np.empty(0)
