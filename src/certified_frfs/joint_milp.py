"""Joint support/metric MILP under fixed witnesses and convex loss envelope."""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy import sparse
from .core import _inputs, interval_bounds
from .majority_metric import MajorityTerms


def pairwise_loss_envelope(X, y, original, reduced, *, eps=.01):
    """Box-uniform upper on dependency loss, clipping negative object losses."""
    lo, hi = interval_bounds(X, eps)
    delta = original-reduced; positive = np.maximum(delta, 0.)
    object_bounds = []
    for i in range(len(X)):
        enemies = np.flatnonzero(y != y[i])
        if len(enemies) == 0:
            object_bounds.append(0.); continue
        lower = np.maximum(0., np.maximum(lo[enemies]-hi[i], lo[i]-hi[enemies]))
        upper = np.maximum(np.abs(lo[i]-hi[enemies]), np.abs(hi[i]-lo[enemies]))
        pair_bounds = lower@delta+(upper-lower)@positive
        object_bounds.append(float(max(0., pair_bounds.max())))
    return {'upper_bound': float(np.mean(object_bounds)), 'object_bounds': object_bounds,
            'scope': 'all_enemy_pair_signed_interval_envelope_empirical_dependency_not_test_risk'}


def joint_certified_metric(X, y, *, eps=.01, retention=.9, dependency_eta=.1,
                          time_limit=10., threshold=1e-9,
                          support_costs=None, continuous=False, loss_slack=0.,
                          support_cap_ratio=None):
    X, y, _, w = _inputs(X, y)
    n, d = X.shape; mass = float(w.sum()); variables = 3*d+n
    costs = np.ones(d) if support_costs is None else np.asarray(support_costs, float)
    cap = mass if support_cap_ratio is None else support_cap_ratio*mass/d
    if support_cap_ratio is not None and (not np.isfinite(support_cap_ratio) or support_cap_ratio < 1):
        raise ValueError('support cap ratio must be at least one')
    if costs.shape != (d,) or not np.isfinite(costs).all() or np.any(costs <= 0):
        raise ValueError('positive finite full-dimension support costs required')
    if not 0 < retention <= 1 or not 0 <= dependency_eta <= 1 or time_limit <= 0:
        raise ValueError('invalid certificate constraints or solver budget')
    if not np.isfinite(threshold) or threshold <= 3e-10:
        raise ValueError('strict positive verification margin required')
    if not np.isfinite(loss_slack) or not 0 <= loss_slack <= dependency_eta:
        raise ValueError('loss slack must lie between zero and dependency threshold')
    lo, hi = interval_bounds(X, eps); workspace = MajorityTerms(X, y, weights=w, eps=eps)
    full_bounds = workspace.bounds(range(d), allowed_enemies=2); protected = full_bounds > 1e-10
    targets = []; required = []; exceptions = []; C = []
    for q in np.unique(y):
        ids = np.flatnonzero((y == q) & protected)
        count = int(np.ceil(retention*len(ids)-1e-12)); required.append(count)
        targets.extend(ids[np.lexsort((ids, -full_bounds[ids]))][:count].tolist())
    for i in targets:
        enemies = np.flatnonzero(y != y[i]); coefficients = workspace.terms[:, i, enemies, :].transpose(1, 2, 0)
        margins = (coefficients@w).min(axis=1); order = np.lexsort((enemies, margins))
        C.append(coefficients[order[2:]].reshape(-1, d))
        exceptions.append({'anchor': i, 'excluded_enemies': enemies[order[:2]].tolist()})
    C = np.concatenate(C, axis=0) if C else np.empty((0, d))
    identity = sparse.eye(d, format='csr'); zero = sparse.csr_matrix((d, d)); tail = sparse.csr_matrix((d, n))
    blocks = [sparse.hstack([-identity, -identity, zero, tail]),
              sparse.hstack([identity, zero, -mass*identity, tail])]
    bounds = [(np.full(d, -np.inf), -w), (np.full(d, -np.inf), np.zeros(d))]
    if len(C):
        blocks.append(sparse.hstack([sparse.csr_matrix(C), sparse.csr_matrix((len(C), 2*d+n))]))
        bounds.append((np.full(len(C), threshold), np.full(len(C), np.inf)))
    pair_count = 0
    for i in range(n):
        enemies = np.flatnonzero(y != y[i])
        if not len(enemies): continue
        lower = np.maximum(0., np.maximum(lo[enemies]-hi[i], lo[i]-hi[enemies]))
        upper = np.maximum(np.abs(lo[i]-hi[enemies]), np.abs(hi[i]-lo[enemies]))
        object_column = sparse.coo_matrix((-np.ones(len(enemies)), (np.arange(len(enemies)), np.full(len(enemies), i))), shape=(len(enemies), n)).tocsr()
        blocks.append(sparse.hstack([-sparse.csr_matrix(lower), sparse.csr_matrix(upper-lower),
            sparse.csr_matrix((len(enemies), d)), object_column]))
        bounds.append((np.full(len(enemies), -np.inf), -(lower@w)))
        pair_count += len(enemies)
    mass_row = sparse.csr_matrix(np.r_[np.ones(d), np.zeros(2*d+n)][None, :])
    loss_row = sparse.csr_matrix(np.r_[np.zeros(3*d), np.ones(n)/n][None, :])
    blocks.extend([mass_row, loss_row]); bounds.extend([(np.array([mass]), np.array([mass])),
                                                        (np.array([-np.inf]), np.array([dependency_eta-loss_slack]))])
    matrix = sparse.vstack(blocks, format='csc')
    constraint = LinearConstraint(matrix, np.concatenate([b[0] for b in bounds]), np.concatenate([b[1] for b in bounds]))
    # Cardinality dominates the bounded tie-break cost of downward weight shift.
    objective = np.r_[np.zeros(d), np.full(d, .001*costs.min()/mass), costs, np.zeros(n)]
    fit = milp(objective, integrality=np.r_[np.zeros(2*d), np.zeros(d) if continuous else np.ones(d), np.zeros(n)],
        bounds=Bounds(np.zeros(variables), np.r_[np.full(d, cap), np.full(d, mass), np.ones(d), np.full(n, mass)]),
        constraints=constraint, options={'time_limit': time_limit, 'mip_rel_gap': 0.})
    info = {'solver_status': int(fit.status), 'solver_message': str(fit.message),
        'solver_success': bool(fit.success), 'targets': targets, 'required': required,
        'continuous': bool(continuous), 'support_costs': costs.tolist(),
        'solver_objective': None if fit.fun is None else float(fit.fun),
        'exceptions': exceptions, 'majority_rows': len(C), 'enemy_pair_rows': pair_count,
        'time_limit': time_limit, 'threshold': threshold, 'eps': eps,
        'retention': retention, 'dependency_eta': dependency_eta,
        'loss_slack': loss_slack,
        'solver_gap': None if getattr(fit, 'mip_gap', None) is None else float(fit.mip_gap),
        'scope': 'fixed_training_polytope_joint_cardinality_metric_not_prediction_or_global_original_problem'}
    candidate = w.copy(); fallback = True
    if fit.x is not None and np.isfinite(fit.x).all():
        v = np.maximum(fit.x[:d], 0.)
        # Remove only coordinates explicitly switched off by binary decisions.
        if not continuous:
            v[fit.x[2*d:3*d] < .5] = 0.
        if v.sum() > 0:
            v *= mass/v.sum()
            if np.any(v > cap+1e-10): v = w.copy()
            linear = bool(not len(C) or np.all(C@v >= threshold-1e-12))
            loss = pairwise_loss_envelope(X, y, w, v, eps=eps)
            info['incumbent_linear_verified'] = linear
            info['incumbent_loss_bound'] = loss['upper_bound']
            if linear and loss['upper_bound'] <= dependency_eta-1e-10:
                candidate = v; fallback = False
    certificate = pairwise_loss_envelope(X, y, w, candidate, eps=eps)
    feasible = certificate['upper_bound'] <= dependency_eta
    if not fallback:
        feasible = feasible and (not len(C) or bool(np.all(C@candidate >= threshold-1e-12)))
    else:
        # Full fallback uses the original mask definition, not the slightly
        # stronger numerical LP margin threshold.
        feasible = feasible and all(sum(y[i] == q for i in targets) >= r for q, r in zip(np.unique(y), required))
    if not feasible: raise RuntimeError('no verified metric or original fallback')
    return {'features': np.flatnonzero(candidate > 0).tolist(), 'weights': candidate.tolist(),
        'full_fallback': fallback, 'fit': info, 'certificate': {'feasible': True,
            'majority_rule': 'original_fixed_witness_polytope' if not fallback else 'original_full_protected_mask',
            'dependency_certificate': certificate},
        'verified_minimum_constraint': None if not len(C) or fallback else float((C@candidate).min())}
