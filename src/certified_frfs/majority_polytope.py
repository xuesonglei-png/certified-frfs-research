"""Nearest transferred metric inside a fixed robust-majority polytope.

Targets, three friends and two excluded enemies are fixed from training.
Feasibility is sufficient only; infeasibility concerns this chosen polytope.
"""
import numpy as np
from scipy.optimize import linprog
from .core import _inputs
from .majority_metric import MajorityTerms
from .redistributed_loss import transfer_weights, redistributed_loss_bound


def fit_majority_polytope(X, y, features, w_full, *, eps=.01, retention=.9,
                          dependency_eta=.1, threshold=1e-9,
                          target_rule='full'):
    X, y, fs, w = _inputs(X, y, features, w_full)
    if not len(fs) or not 0 < retention <= 1 or threshold <= 3e-10 or not np.isfinite(threshold):
        raise ValueError('nonempty support, valid retention and positive robust margin required')
    if not 0 <= dependency_eta <= 1:
        raise ValueError('invalid dependency threshold')
    if target_rule not in ('full', 'reduced'):
        raise ValueError('target_rule must be full or reduced')
    workspace = MajorityTerms(X, y, weights=w, eps=eps)
    full_bounds = workspace.bounds(range(X.shape[1]), allowed_enemies=2)
    protected = full_bounds > 1e-10
    initial = transfer_weights(X, fs, w)
    labels = np.unique(y); required = []; targets = []
    reduced_bounds = MajorityTerms(X[:, fs], y, weights=initial, eps=eps).bounds(range(len(fs)), allowed_enemies=2)
    for label in labels:
        ids = np.flatnonzero((y == label) & protected)
        count = int(np.ceil(retention*len(ids)-1e-12)); required.append(count)
        score = reduced_bounds if target_rule == 'reduced' else full_bounds
        targets.extend(ids[np.lexsort((ids, -score[ids]))][:count].tolist())
    matrices = []; exceptions = []
    for i in targets:
        enemies = np.flatnonzero(y != y[i])
        coefficients = workspace.terms[fs][:, i, enemies, :].transpose(1, 2, 0)
        lower = np.einsum('efp,p->ef', coefficients, initial).min(axis=1)
        order = np.lexsort((enemies, lower))
        excluded = order[:min(2, len(order))]
        retained = order[min(2, len(order)):]
        matrices.append(coefficients[retained].reshape(-1, len(fs)))
        exceptions.append({'anchor': i, 'excluded_enemies': enemies[excluded].tolist(),
                           'constrained_enemies': enemies[retained].tolist()})
    C = np.concatenate(matrices, axis=0) if matrices else np.empty((0, len(fs)))
    mass = float(w.sum()); p = len(fs)
    # Epigraph of L1 distance to the transferred metric. Constraint row scaling
    # aids the solver; certification always rechecks unscaled coefficients.
    norms = np.maximum(np.max(np.abs(C), axis=1), 1e-12) if len(C) else np.empty(0)
    upper = np.vstack([np.column_stack([-C/norms[:, None], np.zeros_like(C)]),
                       np.column_stack([np.eye(p), -np.eye(p)]),
                       np.column_stack([-np.eye(p), -np.eye(p)])])
    rhs = np.r_[-threshold/norms, initial, -initial]
    fit = linprog(np.r_[np.zeros(p), np.ones(p)], A_ub=upper, b_ub=rhs,
        A_eq=np.array([np.r_[np.ones(p), np.zeros(p)]]), b_eq=[mass],
        bounds=[(0., mass)]*(2*p), method='highs',
        options={'primal_feasibility_tolerance': 1e-9, 'dual_feasibility_tolerance': 1e-9})
    info = {'solver_success': bool(fit.success), 'solver_status': int(fit.status),
        'solver_message': str(fit.message), 'features': fs.tolist(), 'targets': targets,
        'protected_counts': [int(((y == q) & protected).sum()) for q in labels],
        'required': required, 'exceptions': exceptions, 'constraint_rows': len(C),
        'threshold': threshold, 'initial_weights': initial.tolist(), 'eps': eps,
        'retention': retention, 'dependency_eta': dependency_eta,
        'target_rule': target_rule,
        'scope': 'fixed_training_witness_polytope_sufficient_not_global_feasibility_or_test_risk'}
    if not fit.success:
        return None, {**info, 'status': 'no_solver_feasible_point', 'certificate': None}
    v = np.maximum(fit.x[:p], 0.)
    if v.sum() == 0:
        return None, {**info, 'status': 'invalid_solver_mass', 'certificate': None}
    v *= mass/v.sum()
    values = C@v
    minimum = float(values.min()) if len(values) else None
    verified = bool(not len(values) or np.all(values >= threshold-1e-12))
    info.update({'minimum_constraint': minimum, 'linear_constraints_verified': verified,
        'weight_shift_l1': float(np.abs(v-initial).sum()), 'weights': v.tolist()})
    if not verified:
        return None, {**info, 'status': 'unscaled_constraint_recheck_failed', 'certificate': None}
    dep = redistributed_loss_bound(X, y, fs, w, v, eps=eps)
    certificate = {'feasible': dep['upper_bound'] <= dependency_eta, 'majority_feasible': True,
        'required': required, 'certified_target_counts': [sum(y[i] == q for i in targets) for q in labels],
        'dependency_certificate': dep, 'majority_rule': 'fixed_original_friends_fixed_two_enemy_exceptions_verified_polytope'}
    certificate['certified_target_counts'] = [int(c) for c in certificate['certified_target_counts']]
    return v, {**info, 'status': 'polytope_verified', 'certificate': certificate}
