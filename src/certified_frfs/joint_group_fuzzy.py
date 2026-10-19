"""Convex grouped score fitting under exact raw-box classification constraints."""
import time
import cvxpy as cp
import numpy as np
from .triangular_classifier import triangular_memberships, triangular_box_margins


def joint_group_scores(X, y, teacher_W, teacher_b, classes, *, C=1., group_strength=.01, ratio=.5, class_balanced=False):
    X, y, classes = np.asarray(X, float), np.asarray(y), np.asarray(classes)
    n, d = X.shape
    q = len(classes)
    if not np.isfinite(C) or C <= 0 or not np.isfinite(group_strength) or group_strength < 0 or not np.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError('invalid regularization')
    original = triangular_box_margins(X, y, teacher_W, teacher_b, classes)
    protected = original > 1e-10
    own = np.argmax(y[:, None] == classes[None, :], axis=1)
    Z = triangular_memberships(X).reshape(n, -1)
    W, b = cp.Variable((q, d*3)), cp.Variable(q)
    centered = W-cp.reshape(cp.sum(W, axis=0)/q, (1, d*3), order='C')
    scores = Z@W.T+cp.reshape(b, (1, q), order='C')
    point_loss = cp.log_sum_exp(scores, axis=1)-scores[np.arange(n), own]
    weights = n/(q*np.bincount(own, minlength=q)[own]) if class_balanced else np.ones(n)
    empirical = cp.sum(cp.multiply(weights, point_loss))
    regularizer = cp.sum_squares(W[1] if q == 2 else centered)/(2*C)
    groups = sum(cp.norm(centered[:, 3*a:3*a+3], 'fro') for a in range(d))
    constraints = [W[0] == 0, b[0] == 0]
    ids, rivals = np.where(protected[:, None] & (np.arange(q)[None, :] != own[:, None]))
    if len(ids):
        lo, hi = np.maximum(0, X-.01), np.minimum(1, X+.01)
        members = triangular_memberships(np.stack([lo, hi, np.clip(.5, lo, hi)], axis=-1))
        contrast = W[own[ids]]-W[rivals]
        lower = cp.Variable((len(ids), d))
        for point in range(3):
            values = sum(cp.multiply(contrast[:, m::3], members[ids, :, point, m]) for m in range(3))
            constraints.append(lower <= values)
        constraints.append(cp.sum(lower, axis=1)+b[own[ids]]-b[rivals] >=
                           np.maximum(ratio*original[ids], 1e-7)+1e-7)
    problem = cp.Problem(cp.Minimize(empirical+regularizer+n*group_strength*groups), constraints)
    start = time.perf_counter()
    solver_attempts = []
    for solver, options in (
        ('CLARABEL', {'max_iter': 200, 'time_limit': 20., 'tol_gap_abs': 1e-8, 'tol_feas': 1e-8}),
        ('SCS', {'max_iters': 100000, 'eps': 1e-6, 'time_limit_secs': 20.}),
    ):
        try:
            problem.solve(solver=solver, **options)
            solver_attempts.append({'solver': solver, 'status': str(problem.status)})
        except cp.error.SolverError as error:
            solver_attempts.append({'solver': solver, 'status': 'solver_error', 'message': str(error)})
            continue
        if W.value is not None and b.value is not None:
            # SCS may return a numerically usable point even when it reports
            # inaccurate; the raw-box verifier below remains authoritative.
            if str(problem.status) in ('optimal', 'optimal_inaccurate'):
                break
    else:
        return {'feasible': False, 'status': 'solver_error', 'solver_attempts': solver_attempts}
    if W.value is None or b.value is None:
        return {'feasible': False, 'status': str(problem.status)}
    coef = W.value.reshape(q, d, 3)
    margins = triangular_box_margins(X, y, coef, b.value, classes)
    feasible = bool(np.all(margins[protected] >= ratio*original[protected]-1e-8)
                    and np.all(margins[protected] > 1e-10))
    contrasts = coef-coef.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(contrasts, axis=(0, 2))
    ranking = sorted(range(d), key=lambda a: (-norms[a], a))
    return {'feasible': feasible, 'coef': coef.tolist(), 'intercept': b.value.tolist(),
            'group_norms': norms.tolist(), 'ranking': ranking, 'status': str(problem.status),
            'original_margins': original.tolist(), 'deployed_margins': margins.tolist(),
            'objective': float(problem.value), 'seconds': time.perf_counter()-start,
            'ratio': float(ratio),
            'class_balanced': bool(class_balanced),
            'solver_attempts': solver_attempts,
            'scope': 'convex_group_surrogate_not_exact_cardinality_or_global_FR_reduct'}
