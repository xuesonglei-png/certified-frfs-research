"""Saturation-aware signed residual prototype; not used by existing experiments.

For memberships rho_v=min(1,min_e D_v), let u=min(1,min_e U_e v).
Then rho_v-rho_w <= max(u-1,max_e interval_support(D_v-D_w)).
The lower floor can be negative, preserving compensation without requiring
a fixed nearest enemy. The elementary saturation identity has prior art.
"""
import numpy as np
from frs_pair_transition_cell import build_class_cell as pair_cell
from frs_pair_transition_cell import evaluate_class_cell as pair_value


def transition_bound(item, anchor, weights):
    delta = np.asarray(anchor)-np.asarray(weights)
    scores = [np.asarray(lo)@delta+(np.asarray(hi)-np.asarray(lo))@np.maximum(delta, 0.)
              for lo, hi in zip(item['distance_lower'], item['distance_upper'], strict=True)]
    return float(max([item['saturation_floor']]+scores))


def evaluate_class_cell(info, weights):
    previous = pair_value(info, weights); anchor = np.asarray(info['anchor'])
    weights = np.asarray(weights, float); fs = info['features']; full = np.asarray(info['full_weights'])
    transitions = {r['object']:transition_bound(r, anchor, weights) for r in info['pair_transition_objects']}
    raw = []
    for r in info['classes']:
        value = r['remaining_signed_anchor_sum']+r['omitted_constant']+r['inflation']
        value += sum(transitions[i] for i in r['remaining_indices'])
        if r['mode'] == 'shared_star':
            delta = full[fs]-weights
            value += np.where(delta >= 0, np.asarray(r['H'])[fs], np.asarray(r['L'])[fs])@delta
        raw.append(float(value/r['count']))
    return {**previous, 'raw_upper_bounds_with_inflation':raw,
        'class_upper_bounds':np.clip(raw, 0, 1).tolist(),
        'feasible':bool(previous['cell_constraints_verified'] and all(v <= info['eta'] for v in raw))}


def build_class_cell(A, y, features, anchor, full, *, eps=.01, eta=.1):
    C, rhs, info = pair_cell(A, y, features, anchor, full, eps=eps, eta=eta)
    for item in info['pair_transition_objects']:
        cap = min([1.]+[float(np.asarray(hi)@anchor) for hi in item['distance_upper']])
        item['anchor_membership_upper'] = float(cap)
        item['saturation_floor'] = float(cap-1.)
    info['metric_envelope'] = 'signed_enemy_transition_with_fixed_anchor_saturation_floor'
    info['anchor_evaluation'] = evaluate_class_cell(info, anchor)
    info['anchor_strictly_feasible'] = bool((not len(C) or np.all(C@anchor > rhs))
        and all(v < eta for v in info['anchor_evaluation']['raw_upper_bounds_with_inflation']))
    return C, rhs, info


def append_class_cell_lp(info, mass, p, n, append):
    offset, extras = 2*p+n, []; anchor = np.asarray(info['anchor'])
    objects = {r['object']:r for r in info['pair_transition_objects']}
    for r in info['classes']:
        cols, vals = [], []
        constant = r['remaining_signed_anchor_sum']+r['omitted_constant']+r['inflation']
        for i in r['remaining_indices']:
            item = objects[i]; t = offset+len(extras)
            extras.append((item['saturation_floor'], None)); cols.append(t); vals.append(1.)
            for lower, upper in zip(item['distance_lower'], item['distance_upper'], strict=True):
                lo, hi = np.asarray(lower), np.asarray(upper)
                append(list(range(p))+list(range(p+n, 2*p+n))+[t],
                    (-.5*mass*(hi+lo)).tolist()+(.5*mass*(hi-lo)).tolist()+[-1.],
                    -.5*float((hi+lo)@anchor))
        if r['mode'] == 'shared_star':
            for j, a in enumerate(info['features']):
                t = offset+len(extras); extras.append((None, None)); cols.append(t); vals.append(1.)
                for slope in (r['H'][a], r['L'][a]):
                    append([j, t], [-mass*slope, -1.], -info['full_weights'][a]*slope)
        append(cols, vals, info['eta']*r['count']-constant)
    return offset+len(extras), extras
