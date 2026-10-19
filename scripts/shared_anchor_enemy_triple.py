"""Finite-bound Lagrangian LP certificate for three enemies at one anchor."""
from itertools import combinations
import numpy as np
from scipy.optimize import linprog


def triple_lp(lower, upper, anchor, enemies, friend, weights, *, time_limit=2.):
    lo, hi, w = np.asarray(lower, float), np.asarray(upper, float), np.asarray(weights, float)
    enemies = np.asarray(enemies, int)
    if (lo.ndim != 2 or lo.shape != hi.shape or np.any(lo > hi) or not np.isfinite(lo).all()
            or not np.isfinite(hi).all() or np.any(lo < 0) or np.any(hi > 1)
            or w.shape != (lo.shape[1],) or np.any(w < 0) or not np.isfinite(w).all()
            or len(enemies) != 3 or len(set(enemies)) != 3
            or len(set([anchor, friend, *enemies])) != 5 or min(anchor, friend, *enemies) < 0
            or max(anchor, friend, *enemies) >= len(lo)):
        raise ValueError('invalid unit-cube boxes, distinct objects, or nonnegative weights')
    p, mass = len(w), float(w.sum())
    left, right = lo[anchor], hi[anchor]
    far_left = np.maximum(abs(left-lo[friend]), abs(left-hi[friend]))
    far_right = np.maximum(abs(right-lo[friend]), abs(right-hi[friend]))
    slope = np.divide(far_right-far_left, right-left, out=np.zeros(p), where=right > left)
    intercept = far_left-slope*left
    # variables anchor[0:p], 3*p enemy-near epigraphs, shared gap.
    size = 4*p+1; c = np.zeros(size); c[-1] = 1.
    rows, rhs = [], []
    for e, enemy in enumerate(enemies):
        for a in range(p):
            r = np.zeros(size); r[a] = 1.; r[p+e*p+a] = -1.
            rows.append(r); rhs.append(hi[enemy, a])
            r = np.zeros(size); r[a] = -1.; r[p+e*p+a] = -1.
            rows.append(r); rhs.append(-lo[enemy, a])
        r = np.zeros(size); r[:p] = -w*slope; r[p+e*p:p+(e+1)*p] = w; r[-1] = -1.
        rows.append(r); rhs.append(float(w@intercept))
    A, b = np.array(rows), np.array(rhs)
    lb = np.r_[left, np.zeros(3*p), -mass]
    ub = np.r_[right, np.ones(3*p), mass]
    res = linprog(c, A_ub=A, b_ub=b, bounds=list(zip(lb, ub)), method='highs', options={'time_limit': time_limit})
    multipliers = getattr(getattr(res, 'ineqlin', None), 'marginals', None)
    if multipliers is None or not np.isfinite(multipliers).all():
        dual = np.zeros(len(b))
    else:
        dual = np.minimum(np.asarray(multipliers), 0.)
    residual = c-A.T@dual
    value = float(dual@b+np.where(residual >= 0, residual*lb, residual*ub).sum())
    lower_bound = value-1e-9*(1+abs(value)+np.sum(abs(dual*b))+np.sum(abs(residual)*np.maximum(abs(lb), abs(ub))))
    return {'lower_bound': lower_bound, 'raw_lagrangian_bound': value, 'multipliers': dual.tolist(),
            'solver_status': int(res.status), 'solver_success': bool(res.success),
            'primal_objective_not_certificate': None if res.fun is None else float(res.fun),
            'anchor': int(anchor), 'enemies': enemies.tolist(), 'friend': int(friend),
            'scope': 'shared_anchor_enemy_triple_chord_Lagrangian_lower_bound_not_population_risk',
            'rounding': 'ordinary_float_with_padding_not_directed_rounding'}


def refine_anchor(lower, upper, anchor, friends, enemies, weights, *, max_threats=8, max_lps=168):
    lo, hi, w = np.asarray(lower), np.asarray(upper), np.asarray(weights)
    friends, enemies = np.asarray(friends, int), np.asarray(enemies, int)
    def at(x):
        near = np.maximum(0., np.maximum(lo[enemies]-x, x-hi[enemies]))
        far = np.maximum(abs(x-lo[friends]), abs(x-hi[friends]))
        return near[:, None, :]-far[None, :, :]
    pair = np.minimum(at(lo[anchor]), at(hi[anchor]))@w
    scalar = pair.min(axis=1)
    threats = enemies[scalar <= 1e-10*(1+abs(scalar))]
    record = {'anchor': int(anchor), 'friends': friends.tolist(), 'threats': threats.tolist(), 'checks': [],
              'certified': len(threats) <= 2, 'status': 'old_pairwise_certificate' if len(threats) <= 2 else 'pending'}
    if record['certified']: return record
    if len(enemies) >= 3:
        per_friend = np.partition(pair, 2, axis=0)[2]
        if np.min(per_friend) > 1e-10*(1+np.max(abs(per_friend))):
            return {**record, 'certified': True, 'status': 'per_friend_third_enemy_certified',
                    'per_friend_third_bounds': per_friend.tolist()}
    if len(threats) > max_threats:
        return {**record, 'status': 'threat_budget_unresolved'}
    for triple in combinations(threats, 3):
        for friend in friends:
            col = int(np.flatnonzero(friends == friend)[0])
            ids = np.array([int(np.flatnonzero(enemies == e)[0]) for e in triple])
            scalar_lower = float(pair[ids, col].max())
            if scalar_lower > 1e-10*(1+abs(scalar_lower)):
                continue
            if len(record['checks']) >= max_lps:
                return {**record, 'status': 'LP_budget_unresolved'}
            check = triple_lp(lo, hi, anchor, triple, int(friend), w)
            record['checks'].append(check)
            if check['lower_bound'] <= 1e-10:
                return {**record, 'status': 'nonpositive_lower_bound_unresolved'}
    return {**record, 'certified': True, 'status': 'all_enemy_triples_certified'}
