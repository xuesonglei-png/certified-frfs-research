"""Convex overlapping node groups for training-only forest feature ranking.

Each edge coefficient contributes half its squared norm to each incident
node. Vanishing node groups eliminate all incident interactions. This is an
experimental selector, not a claimed novel group-lasso objective or FRS bound.
"""
import cvxpy as cp
import numpy as np
from .triangular_classifier import triangular_memberships
from .orthogonal_forest_logistic import CONTRAST


def fit_grouped_forest(X, y, edges, *, C=1., pair_penalty=1., relative_strength=.1):
    X, y = np.asarray(X, float), np.asarray(y); edges = [list(e) for e in edges]
    if not np.isfinite(C) or C <= 0 or not np.isfinite(pair_penalty) or pair_penalty <= 0 or not np.isfinite(relative_strength) or relative_strength < 0:
        raise ValueError('invalid group regularization')
    n, d = X.shape; classes = np.unique(y); q = len(classes); rows = 1 if q == 2 else q
    if q < 2: raise ValueError('at least two classes required')
    # The common score oracle enforces forest and canonical edge validity.
    from .forest_fuzzy_classifier import forest_scores
    forest_scores(X[:1], np.zeros((q, d, 3)), np.zeros(q), edges, np.zeros((q, len(edges), 3, 3)))
    Z = triangular_memberships(X) @ CONTRAST
    blocks = [Z.reshape(n, -1)]
    blocks += [(Z[:, u, :, None]*Z[:, v, None, :]).reshape(n, 4)/np.sqrt(pair_penalty) for u, v in edges]
    design = np.column_stack(blocks); p = design.shape[1]
    own = np.argmax(y[:, None] == classes[None], axis=1)
    balance = np.array([1/(q*sum(y == c)) for c in y])
    if q == 2:
        gradient = ((.5-own)*balance) @ design
        gradient = gradient[None]
    else:
        gradient = (balance[:, None]*(1/q-np.eye(q)[own])).T @ design
    group_columns = []
    for a in range(d):
        cols = [2*a, 2*a+1]; factors = [1., 1.]
        for e, (u, v) in enumerate(edges):
            if a in (u, v):
                cols += list(range(2*d+4*e, 2*d+4*(e+1))); factors += [1/np.sqrt(2)]*4
        group_columns.append((cols, np.array(factors)))
    # Explicit feasible dual allocation of each pair gradient to its two
    # endpoints. This gives a sufficient zero-solution threshold, not the
    # smallest possible lambda_max for overlapping groups.
    threshold = max(np.linalg.norm(gradient[:, cols]*factors[None]) for cols, factors in group_columns)
    strength = float(relative_strength*threshold)
    theta = cp.Variable((rows, p)); bias = cp.Variable(rows)
    logits = design @ theta.T+cp.reshape(bias, (1, rows), order='C')
    constraints = []
    if q == 2:
        margin = logits[:, 0]
        empirical = cp.sum(cp.multiply(balance, cp.logistic(margin)-cp.multiply(own, margin)))
    else:
        empirical = cp.sum(cp.multiply(balance, cp.log_sum_exp(logits, axis=1)-logits[np.arange(n), own]))
        constraints = [cp.sum(theta, axis=0) == 0, cp.sum(bias) == 0]
    groups = sum(cp.norm(cp.multiply(theta[:, cols], factors[None]), 'fro') for cols, factors in group_columns)
    problem = cp.Problem(cp.Minimize(empirical+cp.sum_squares(theta)/(2*C*n)+strength*groups), constraints)
    try:
        problem.solve(solver='CLARABEL', max_iter=300, time_limit=30., tol_gap_abs=1e-11, tol_gap_rel=1e-11, tol_feas=1e-11)
    except cp.error.SolverError as error:
        return {'usable': False, 'status': 'solver_error', 'message': str(error)}
    if problem.status != 'optimal' or theta.value is None or bias.value is None:
        return {'usable': False, 'status': str(problem.status)}
    beta = np.asarray(theta.value); b = np.asarray(bias.value)
    norms = np.array([np.linalg.norm(beta[:, cols]*factors[None]) for cols, factors in group_columns])
    if q == 2:
        beta = np.vstack([np.zeros_like(beta[0]), beta[0]]); b = np.r_[0., b[0]]
    W = np.einsum('qdi,ki->qdk', beta[:, :2*d].reshape(q, d, 2), CONTRAST)
    V = np.zeros((q, len(edges), 3, 3))
    for e in range(len(edges)):
        tensor = beta[:, 2*d+4*e:2*d+4*(e+1)].reshape(q, 2, 2)/np.sqrt(pair_penalty)
        V[:, e] = np.einsum('ki,qij,lj->qkl', CONTRAST, tensor, CONTRAST)
    W -= W[:1].copy(); V -= V[:1].copy(); b -= b[:1].copy()
    return {'usable': True, 'status': str(problem.status), 'objective': float(problem.value),
        'coef': W.tolist(), 'pair_coef': V.tolist(), 'edges': edges, 'intercept': b.tolist(),
        'features': list(range(d)), 'classes': classes.tolist(), 'C': float(C), 'pair_penalty': float(pair_penalty),
        'relative_strength': float(relative_strength), 'strength': strength, 'sufficient_zero_threshold': float(threshold),
        'node_group_norms': norms.tolist(), 'teacher_guards_enforced': False,
        'scope': 'experimental_balanced_overlapping_group_selector_no_FRS_or_population_guarantee'}
