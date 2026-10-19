"""Tighter residual FRS cell via existing signed enemy-pair interval bounds.

Refines the one-sided residual Lipschitz envelope; stable shared-star terms
and their cell remain identical. The underlying pair bound already exists
in certified_frfs.joint_milp. This integration is not a novelty claim.
"""
import numpy as np
from frs_one_sided_cell import build_class_cell as build_monotone_cell
from frs_one_sided_cell import evaluate_class_cell as evaluate_monotone_cell
from certified_frfs.core import interval_bounds


def transition_bound(item, anchor, weights):
    delta = np.asarray(anchor)-np.asarray(weights)
    if not item['enemies']:
        return 0.
    lower, upper = np.asarray(item['distance_lower']), np.asarray(item['distance_upper'])
    return float(max(0., np.max(lower@delta+(upper-lower)@np.maximum(delta, 0.))))


def evaluate_class_cell(info, weights):
    previous = evaluate_monotone_cell(info, weights)
    fs, full = info['features'], np.asarray(info['full_weights'])
    weights = np.asarray(weights, float); anchor = np.asarray(info['anchor'])
    transitions = {r['object']: transition_bound(r, anchor, weights) for r in info['pair_transition_objects']}
    raw = []
    for r in info['classes']:
        value = r['remaining_signed_anchor_sum']+r['omitted_constant']+r['inflation']
        value += sum(transitions[i] for i in r['remaining_indices'])
        if r['mode'] == 'shared_star':
            delta = full[fs]-weights
            value += np.where(delta >= 0, np.asarray(r['H'])[fs], np.asarray(r['L'])[fs])@delta
        raw.append(float(value/r['count']))
    return {**previous, 'raw_upper_bounds_with_inflation': raw, 'class_upper_bounds': np.clip(raw, 0, 1).tolist(),
        'feasible': bool(previous['cell_constraints_verified'] and all(v <= info['eta'] for v in raw))}


def build_class_cell(A, y, features, anchor, full, *, eps=.01, eta=.1):
    C, rhs, info = build_monotone_cell(A, y, features, anchor, full, eps=eps, eta=eta)
    A, y = np.asarray(A), np.asarray(y); fs = info['features']; lo, hi = interval_bounds(A, eps)
    remaining = sorted({i for r in info['classes'] for i in r['remaining_indices']})
    objects = []
    for i in remaining:
        enemies = np.flatnonzero(y != y[i]); elo, ehi = lo[enemies][:, fs], hi[enemies][:, fs]
        lower = np.maximum(0., np.maximum(elo-hi[i, fs], lo[i, fs]-ehi))
        upper = np.maximum(abs(lo[i, fs]-ehi), abs(hi[i, fs]-elo))
        objects.append({'object': i, 'enemies': enemies.tolist(), 'distance_lower': lower.tolist(), 'distance_upper': upper.tolist()})
    info['pair_transition_objects'] = objects
    info['metric_envelope'] = 'per_object_max_enemy_signed_pair_transition'
    info['prior_bound_source'] = 'certified_frfs.joint_milp.pairwise_loss_envelope'
    info['anchor_evaluation'] = evaluate_class_cell(info, anchor)
    info['anchor_strictly_feasible'] = bool((not len(C) or np.all(C@anchor > rhs))
        and all(v < eta for v in info['anchor_evaluation']['raw_upper_bounds_with_inflation']))
    return C, rhs, info


def append_class_cell_lp(info, mass, p, n, append):
    offset = 2*p+n; extras = []
    full, fs, anchor = np.asarray(info['full_weights']), info['features'], np.asarray(info['anchor'])
    objects = {r['object']: r for r in info['pair_transition_objects']}
    for r in info['classes']:
        cols, vals = [], []
        constant = r['remaining_signed_anchor_sum']+r['omitted_constant']+r['inflation']
        for i in r['remaining_indices']:
            item = objects[i]; t = offset+len(extras); extras.append((0, None)); cols.append(t); vals.append(1.)
            for low, high in zip(item['distance_lower'], item['distance_upper'], strict=True):
                low, high = np.asarray(low), np.asarray(high)
                # Reuse u >= |w_normalized-anchor_normalized|. Each enemy
                # upper is .5*(high+low)*(anchor-w)+.5*(high-low)*|anchor-w|.
                append(list(range(p))+list(range(p+n, 2*p+n))+[t],
                    (-.5*mass*(high+low)).tolist()+(.5*mass*(high-low)).tolist()+[-1.],
                    -.5*float((high+low)@anchor))
        if r['mode'] == 'shared_star':
            for j, a in enumerate(fs):
                z = offset+len(extras); extras.append((None, None)); cols.append(z); vals.append(1.)
                for slope in (r['H'][a], r['L'][a]):
                    append([j, z], [-mass*slope, -1.], -full[a]*slope)
        append(cols, vals, info['eta']*r['count']-constant)
    return offset+len(extras), extras
