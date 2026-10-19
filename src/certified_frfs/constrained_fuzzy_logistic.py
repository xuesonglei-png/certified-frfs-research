"""Convex logistic training subject to exact shared-raw-box teacher margins."""
import time
import cvxpy as cp
import numpy as np
from .triangular_classifier import triangular_memberships, triangular_box_margins


def fit_constrained_logistic(X, y, teacher_W, teacher_b, classes, features, *, C=1., eps=.01, ratio=.5, class_balanced=False):
    X, y, classes = np.asarray(X, float), np.asarray(y), np.asarray(classes)
    n, d = X.shape
    fs = list(features)
    if not np.isfinite(C) or C <= 0 or not np.isfinite(ratio) or not 0 < ratio <= 1 or not fs or len(set(fs)) != len(fs):
        raise ValueError('invalid support or C')
    if min(fs) < 0 or max(fs) >= d:
        raise ValueError('out of range support')
    original = triangular_box_margins(X, y, teacher_W, teacher_b, classes, eps=eps)
    protected = original > 1e-10
    own = np.argmax(y[:, None] == classes[None, :], axis=1)
    q, k = len(classes), len(fs)
    Z = triangular_memberships(X[:, fs]).reshape(n, -1)
    W, b = cp.Variable((q, k*3)), cp.Variable(q)
    scores = Z@W.T + cp.reshape(b, (1, q), order='C')
    point_loss = cp.log_sum_exp(scores, axis=1)-scores[np.arange(n), own]
    if class_balanced:
        weights = np.array([n/(q*np.sum(y == label)) for label in y])
        loss = cp.sum(cp.multiply(weights, point_loss))
    else:
        loss = cp.sum(point_loss)
    if q == 2:
        penalty = cp.sum_squares(W[1])/(2*C)
    else:
        centered = W-cp.reshape(cp.sum(W, axis=0)/q, (1, k*3), order='C')
        penalty = cp.sum_squares(centered)/(2*C)
    constraints = [W[0] == 0, b[0] == 0]
    ids, rivals = np.where(protected[:, None] & (np.arange(q)[None, :] != own[:, None]))
    if len(ids):
        lo, hi = np.maximum(0, X[:, fs]-eps), np.minimum(1, X[:, fs]+eps)
        members = triangular_memberships(np.stack([lo, hi, np.clip(.5, lo, hi)], axis=-1))
        contrast = W[own[ids]]-W[rivals]
        lower = cp.Variable((len(ids), k))
        for p in range(3):
            values = sum(cp.multiply(contrast[:, m::3], members[ids, :, p, m]) for m in range(3))
            constraints.append(lower <= values)
        required = np.maximum(ratio*original[ids], 1e-7)+1e-7
        constraints.append(cp.sum(lower, axis=1)+b[own[ids]]-b[rivals] >= required)
    problem = cp.Problem(cp.Minimize(loss+penalty), constraints)
    start = time.perf_counter()
    try:
        problem.solve(solver='CLARABEL', max_iter=200, time_limit=20.,
                      tol_gap_abs=1e-8, tol_feas=1e-8)
    except cp.error.SolverError as error:
        return {'feasible': False, 'status': 'solver_error', 'message': str(error)}
    if W.value is None or b.value is None:
        return {'feasible': False, 'status': str(problem.status), 'seconds': time.perf_counter()-start}
    coef = np.zeros((q, d, 3))
    coef[:, fs] = W.value.reshape(q, k, 3)
    margins = triangular_box_margins(X, y, coef, b.value, classes, eps=eps)
    feasible = bool(np.all(margins[protected] >= ratio*original[protected]-1e-8)
                    and np.all(margins[protected] > 1e-10))
    return {'feasible': feasible, 'coef': coef.tolist(), 'intercept': b.value.tolist(),
            'status': str(problem.status), 'objective': float(problem.value),
            'original_margins': original.tolist(), 'deployed_margins': margins.tolist(),
            'seconds': time.perf_counter()-start,
            'ratio': float(ratio),
            'class_balanced': bool(class_balanced),
            'scope': 'exact_constrained_logistic_fixed_support_not_global_reduct_optimization'}
