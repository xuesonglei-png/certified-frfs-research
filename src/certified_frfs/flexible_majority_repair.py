"""Prototype flexible-target/enemy MILP, accepted only by original certificates.

Friends are fixed using the transferred reduced metric. Targets and up to two
enemy exceptions per target are optimized, unlike the earlier fixed polytope.
Failure never proves absence of another metric. No classification/query API.
"""
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix
from certified_frfs.majority_metric import MajorityTerms
from certified_frfs.redistributed_loss import transfer_weights
from certified_frfs.capped_sparse import capped_sparse_projection
from certified_frfs.feasible_transport import weight_certificate


def capped_simplex_minimum(coefficients, cap):
    C = np.asarray(coefficients, float)
    if C.ndim != 2 or not 0 < cap <= 1 or C.shape[1]*cap < 1-1e-12:
        raise ValueError('invalid capped simplex')
    allocation = np.clip(1-cap*np.arange(C.shape[1]), 0, cap)
    return np.sort(C, axis=1)@allocation


def fit_flexible_majority_repair(X, y, features, *, time_limit=10.):
    X, y = np.asarray(X, float), np.asarray(y)
    fs = list(features)
    if X.ndim != 2 or y.shape != (len(X),) or not fs or fs != sorted(set(fs)) or any(int(a) != a or a < 0 or a >= X.shape[1] for a in fs):
        raise ValueError('invalid reference/support')
    n, d = X.shape
    p, mass, cap = len(fs), np.sqrt(d), 2/d
    if p*cap < 1-1e-12:
        return None, {'status': 'mass_cap_infeasible', 'features': fs}
    full = np.ones(d)/mass
    original = MajorityTerms(X, y, weights=full)
    protected = original.bounds(range(d), allowed_enemies=2) > 1e-10
    del original
    initial = capped_sparse_projection(transfer_weights(X, fs, full)/mass, p, cap)
    workspace = MajorityTerms(X[:, fs], y, weights=initial*mass)
    anchors = np.flatnonzero(protected)
    labels = np.unique(y)
    required = [int(np.ceil(.9*np.sum(protected & (y == c))-1e-12)) for c in labels]
    threshold = 1e-8/mass
    # Discard only enemies whose all three constraints hold over the ENTIRE
    # capped simplex. The bound is analytic, not sampled from candidate weights.
    blocks, exception_count = [], 0
    for i in anchors:
        enemies = np.flatnonzero(y != y[i])
        coefficients = workspace.terms[:, i, enemies, :].transpose(1, 2, 0)
        lower = capped_simplex_minimum(coefficients.reshape(-1, p), cap).reshape(len(enemies), 3)
        threat = lower.min(axis=1) < threshold
        C, L, ids = coefficients[threat], lower[threat], enemies[threat]
        blocks.append((int(i), C, L, ids, exception_count))
        exception_count += len(ids)
    del workspace
    K = len(anchors)
    variables = 2*p+K+exception_count
    row_ids, col_ids, data, upper = [], [], [], []
    def add(cols, values, bound):
        r = len(upper)
        row_ids.extend([r]*len(cols)); col_ids.extend(cols); data.extend(values)
        upper.append(bound)
    for j in range(p):
        add([j, p+j], [1., -1.], initial[j])
        add([j, p+j], [-1., -1.], -initial[j])
    for c, count in zip(labels, required):
        ids = [2*p+k for k, i in enumerate(anchors) if y[i] == c]
        if count:
            add(ids, [-1.]*len(ids), -count)
    for k, (anchor, C, lower, enemies, offset) in enumerate(blocks):
        keep = 2*p+k
        exception_ids = list(range(2*p+K+offset, 2*p+K+offset+len(enemies)))
        if exception_ids:
            add(exception_ids+[keep], [1.]*len(exception_ids)+[-2.], 0.)
        for e, exception in enumerate(exception_ids):
            for f in range(3):
                big_M = max(0., threshold-lower[e, f])+1e-12
                add(list(range(p))+[keep, exception], (-C[e, f]).tolist()+[big_M, -big_M], big_M-threshold)
    matrix = coo_matrix((data, (row_ids, col_ids)), shape=(len(upper), variables)).tocsr()
    objective = np.r_[np.zeros(p), np.ones(p), np.zeros(K), np.full(exception_count, 1e-7)]
    lower_bounds = np.zeros(variables)
    upper_bounds = np.r_[np.full(p, cap), np.ones(p), np.ones(K+exception_count)]
    integrality = np.r_[np.zeros(2*p), np.ones(K+exception_count)]
    mass_row = coo_matrix((np.ones(p), (np.zeros(p, int), np.arange(p))), shape=(1, variables)).tocsr()
    result = milp(objective, integrality=integrality, bounds=Bounds(lower_bounds, upper_bounds),
                  constraints=[LinearConstraint(matrix, -np.inf, np.array(upper)), LinearConstraint(mass_row, [1.], [1.])],
                  options={'time_limit': time_limit, 'mip_rel_gap': 0.})
    info = {'features': fs, 'solver_status': int(result.status), 'solver_message': str(result.message),
            'solver_success': bool(result.success), 'time_limit': float(time_limit),
            'protected_anchors': K, 'enemy_exception_variables': exception_count,
            'linear_rows': len(upper), 'required': required, 'cap_normalized': cap,
            'initial_weights': (initial*mass).tolist(), 'certificate': None,
            'scope': 'fixed_reduced_friend_flexible_target_exception_MILP_not_global_feasibility_or_test_risk'}
    if result.x is None:
        return None, {**info, 'status': 'no_incumbent'}
    solution = result.x
    valid = (np.max(matrix@solution-np.array(upper), initial=0.) <= 1e-8
             and abs(np.sum(solution[:p])-1) <= 1e-8
             and np.max(abs(solution[2*p:]-np.round(solution[2*p:])), initial=0.) <= 1e-8)
    if not valid:
        return None, {**info, 'status': 'incumbent_constraints_failed'}
    v = np.maximum(0., solution[:p])
    v /= v.sum()
    if np.max(v) > cap+1e-10:
        return None, {**info, 'status': 'normalized_cap_failed'}
    weights = v*mass
    cert = weight_certificate(X, y, fs, full, weights, protected=protected)
    info.update({'weights': weights.tolist(), 'certificate': cert,
                 'selected_targets': anchors[np.round(solution[2*p:2*p+K]).astype(bool)].tolist(),
                 'objective': float(result.fun)})
    if not cert['feasible']:
        return None, {**info, 'status': 'original_certificate_failed'}
    return weights, {**info, 'status': 'original_certificate_passed'}
