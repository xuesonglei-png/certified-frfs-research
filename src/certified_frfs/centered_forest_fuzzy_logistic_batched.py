"""Equivalent centered pairwise fitting with batched separation constraints.

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
                        C=1., eps=.01, ratio=.5, max_rounds=40, class_balanced=True, pair_penalty=1.):
    X, y, classes = np.asarray(X, float), np.asarray(y), np.asarray(classes)
    fs = list(features); d = X.shape[1]; q = len(classes)
    if not np.isfinite(C) or C <= 0 or not np.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError('invalid C or margin ratio')
    if not fs or len(set(fs)) != len(fs) or min(fs) < 0 or max(fs) >= d:
        raise ValueError('invalid feature support')
    if int(max_rounds) != max_rounds or max_rounds < 1:
        raise ValueError('positive integer separation limit required')
    if not np.isfinite(pair_penalty) or pair_penalty <= 0:
        raise ValueError('positive finite interaction penalty required')
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
    penalty_B = cp.reshape(B[1], (1, p), order='C') if q == 2 else centered
    main_penalty = cp.sum_squares(penalty_B[:, :3*len(fs)])
    interaction_penalty = cp.sum_squares(penalty_B[:, 3*len(fs):]) if retained else 0.
    penalty = (main_penalty+pair_penalty*interaction_penalty)/(2*C)
    objective = cp.Minimize(weights@losses+penalty)
    constraints = [B[0] == 0, b[0] == 0]
    # Partition of unity permits uncentered pair terms to duplicate main effects.
    # Zero row/column sums give a pure interaction tensor, isolating capacity.
    for e in range(len(retained)):
        block = B[:, 3*len(fs)+9*e:3*len(fs)+9*(e+1)]
        column_sums = block[:, :3]+block[:, 3:6]+block[:, 6:9]
        row_sums = cp.hstack([cp.sum(block[:, 3*k:3*k+3], axis=1, keepdims=True) for k in range(3)])
        constraints += [column_sums == 0, row_sums == 0]
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
                  'centered_interactions': True,
                  'centering_constraint_layout': 'two_dimensional_equivalent',
                  'separation_constraint_layout': 'batched_equivalent',
                  'pair_penalty': float(pair_penalty),
                  'scope': 'convex_fixed_forest_box_separation_verified_feasibility_not_global_reduct_or_test_risk'}
        if result['feasible']:
            if retained and max(np.max(np.abs(V.sum(axis=2))), np.max(np.abs(V.sum(axis=3)))) > 1e-7:
                result['feasible'] = False; result['status'] = 'centering_verification_failed'
            return result
        added = 0; cut_ids = []; cut_rivals = []; cut_rows = []
        for i in violated:
            rival = int(oracle['rivals'][i]); witness = oracle['witness_raw'][i]
            identity = (int(i), rival, tuple(witness.tolist()))
            if identity in seen:
                continue
            seen.add(identity)
            row = forest_design(witness[None], fs, retained)[0]
            cut_ids.append(int(i)); cut_rivals.append(rival); cut_rows.append(row)
            added += 1
        if added:
            ids = np.asarray(cut_ids, int); rivals = np.asarray(cut_rivals, int)
            rows = np.asarray(cut_rows)
            cut_scores = cp.sum(cp.multiply(B[own[ids]]-B[rivals], rows), axis=1)
            constraints.append(cut_scores+b[own[ids]]-b[rivals] >= required[ids])
        if not added:
            result['feasible'] = False; result['status'] = 'stalled_separation'
            return result
    result['feasible'] = False; result['status'] = 'separation_limit'
    return result
