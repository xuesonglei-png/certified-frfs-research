"""Convex fuzzy forest logistic fitting via exact box separation cuts.

Limits yield an explicit uncertified failure. Solver labels do not establish
optimality; final exact shared-raw-box margins determine deployment feasibility.
"""
import time
import cvxpy as cp
import numpy as np
from .triangular_classifier import triangular_memberships, triangular_box_margins
from .forest_fuzzy_classifier import forest_box_margins


def forest_design(X, features, edges):
    Z = triangular_memberships(X)
    fs = list(features)
    blocks = [Z[:, fs].reshape(len(X), -1)]
    for u, v in edges:
        blocks.append((Z[:, u, :, None]*Z[:, v, None, :]).reshape(len(X), 9))
    return np.column_stack(blocks)


def fit_forest_logistic(X, y, teacher_W, teacher_b, classes, features, edges, *,
                        C=1., eps=.01, ratio=.5, max_rounds=40, class_balanced=True):
    X, y, classes = np.asarray(X, float), np.asarray(y), np.asarray(classes)
    fs = list(features); d = X.shape[1]; q = len(classes)
    if not np.isfinite(C) or C <= 0 or not np.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError('invalid C or margin ratio')
    if not fs or len(set(fs)) != len(fs) or min(fs) < 0 or max(fs) >= d:
        raise ValueError('invalid feature support')
    if int(max_rounds) != max_rounds or max_rounds < 1:
        raise ValueError('positive integer separation limit required')
    raw_edges = [tuple(e) for e in edges]
    # Validate the full graph before inducing the selected support.
    forest_box_margins(X, y, np.zeros((q, d, 3)), np.zeros(q), classes,
                       raw_edges, np.zeros((q, len(raw_edges), 3, 3)), eps=eps)
    retained = [edge for edge in raw_edges if all(a in fs for a in edge)]
    original = triangular_box_margins(X, y, teacher_W, teacher_b, classes, eps=eps)
    protected = original > 1e-10
    required = np.maximum(ratio*original, 1e-7)+1e-7
    own = np.argmax(y[:, None] == classes, axis=1)
    Z = forest_design(X, fs, retained); p = Z.shape[1]
    B, b = cp.Variable((q, p)), cp.Variable(q)
    score = Z@B.T+cp.reshape(b, (1, q), order='C')
    losses = cp.log_sum_exp(score, axis=1)-score[np.arange(len(y)), own]
    weights = np.array([len(y)/(q*np.sum(y == label)) for label in y]) if class_balanced else np.ones(len(y))
    centered = B-cp.reshape(cp.sum(B, axis=0)/q, (1, p), order='C')
    penalty = cp.sum_squares(B[1] if q == 2 else centered)/(2*C)
    objective = cp.Minimize(weights@losses+penalty)
    constraints = [B[0] == 0, b[0] == 0]
    trace = []; start = time.perf_counter(); seen = set()
    for iteration in range(int(max_rounds)):
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver='CLARABEL', max_iter=200, time_limit=20.,
                          tol_gap_abs=1e-8, tol_feas=1e-8)
        except cp.error.SolverError as error:
            return {'feasible': False, 'status': 'solver_error', 'message': str(error), 'trace': trace}
        if B.value is None or b.value is None:
            return {'feasible': False, 'status': str(problem.status), 'trace': trace}
        W = np.zeros((q, d, 3)); W[:, fs] = B.value[:, :3*len(fs)].reshape(q, len(fs), 3)
        V = B.value[:, 3*len(fs):].reshape(q, len(retained), 3, 3)
        oracle = forest_box_margins(X, y, W, b.value, classes, retained, V, eps=eps, witnesses=True)
        margins = oracle['margins']
        # A small guard above the declared ratio prevents boundary false passes.
        violated = np.flatnonzero(protected & (margins < required-1e-8))
        verified = bool(np.all(margins[protected] >= ratio*original[protected]-1e-8)
                        and np.all(margins[protected] > 1e-10))
        trace.append({'round': iteration, 'status': str(problem.status), 'objective': float(problem.value),
                      'violated_objects': violated.tolist(), 'constraints': len(constraints)})
        result = {'feasible': verified and not len(violated), 'coef': W.tolist(), 'pair_coef': V.tolist(),
                  'intercept': b.value.tolist(), 'edges': [list(e) for e in retained],
                  'status': str(problem.status), 'objective': float(problem.value),
                  'original_margins': original.tolist(), 'deployed_margins': margins.tolist(),
                  'class_balanced': bool(class_balanced), 'C': float(C), 'ratio': float(ratio),
                  'eps': float(eps), 'trace': trace, 'seconds': time.perf_counter()-start,
                  'scope': 'convex_fixed_forest_box_separation_verified_feasibility_not_global_reduct_or_test_risk'}
        if result['feasible']:
            return result
        added = 0
        for i in violated:
            rival = int(oracle['rivals'][i]); witness = oracle['witness_raw'][i]
            identity = (int(i), rival, tuple(witness.tolist()))
            if identity in seen:
                continue
            seen.add(identity)
            row = forest_design(witness[None], fs, retained)[0]
            constraints.append((B[own[i]]-B[rival])@row+b[own[i]]-b[rival] >= required[i])
            added += 1
        if not added:
            result['feasible'] = False; result['status'] = 'stalled_separation'
            return result
    result['feasible'] = False; result['status'] = 'separation_limit'
    return result
