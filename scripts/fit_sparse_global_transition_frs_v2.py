"""V2: classwise CCCP with scalar AND vector verified inward repair.

Returns only a verified nonincreasing actual DC objective. Subproblem solver
failures remain failures even if a preceding feasible iterate is available.
"""
import hashlib
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from capped_third_enemy_hinge import CappedThirdEnemyHinge
from normalized_log_support_penalty import penalty, tangent
from soft_topk_predictive_metric import anchor_witness_polytope
from frs_global_saturation_transition import build_class_cell, append_class_cell_lp
from verified_global_cell_v2 import evaluate_class_cell
from verified_global_cell_v2 import repair_cell_metric


def fit_sparse_global_transition_target(A, y, features, anchor, *, full_reference, reference_scope, initialization=None, initialization_scope='anchor', constrained=True, sparsity_lambda=.05, sparsity_delta=.01, margin=.02,
                          reg=.01, max_major=8, max_iter=200, gap_tolerance=1e-7):
    A, y, anchor = np.asarray(A, float), np.asarray(y), np.asarray(anchor, float)
    fs = list(features)
    if (not fs or fs != sorted(set(fs)) or any(int(a) != a or not 0 <= a < A.shape[1] for a in fs)
        or anchor.shape != (len(fs),) or not np.isfinite(anchor).all() or anchor.min() < 0
        or not np.isfinite(sparsity_lambda) or sparsity_lambda < 0 or not np.isfinite(sparsity_delta) or sparsity_delta <= 0
        or not np.isfinite(reg) or reg < 0 or int(max_major) != max_major or max_major < 1
        or int(max_iter) != max_iter or max_iter < 1 or not np.isfinite(gap_tolerance) or gap_tolerance <= 0):
        raise ValueError('invalid third-enemy target settings')
    mass = float(np.sqrt(A.shape[1])); initial = anchor / mass
    if abs(initial.sum()-1) > 1e-8:
        raise ValueError('original mass required')
    oracle = CappedThirdEnemyHinge(A[:, fs], y, initial, margin=margin)
    actual_objective = lambda v: oracle.objective(v, reg=reg)+sparsity_lambda*penalty(v, sparsity_delta)
    p, n = len(fs), len(A)
    C, rhs, polytope = anchor_witness_polytope(A, y, fs, anchor) if constrained else (np.empty((0, p)), np.empty(0), None)
    cell_C, cell_rhs, cell = build_class_cell(A, y, fs, anchor, full_reference)
    if not cell['anchor_strictly_feasible']:
        raise ValueError('strictly feasible class-cell anchor required')
    C = np.concatenate((C, cell_C)); rhs = np.r_[rhs, cell_rhs]
    cuts, keys, points, steps = [], set(), [], []
    bounds = [(0, 1)]*p + [(r['constant_loss'], r['constant_loss']) if r['constant_loss'] is not None else (0, None)
                              for r in oracle.rows] + [(0, None)]*p

    def add(point, faces):
        pid = len(points); points.append(point.tolist())
        for face in faces:
            if oracle.rows[face['query']]['constant_loss'] is not None:
                continue
            key = (face['query'], tuple(face['gradient']), face['intercept'])
            if key not in keys:
                keys.add(key); cuts.append({**face, 'point_id': pid})

    def master(linear_gradient, support_gradient):
        rid, columns, values, b = [], [], [], []
        def append(cols, vals, bound):
            rid.extend([len(b)]*len(cols)); columns.extend(cols); values.extend(vals); b.append(bound)
        for j in range(p):
            append([j, p+n+j], [1, -1], initial[j]); append([j, p+n+j], [-1, -1], -initial[j])
        for row, bound in zip(C, rhs):
            append(list(range(p)), (-row*mass).tolist(), -bound)
        size, extras = append_class_cell_lp(cell, mass, p, n, append)
        for cut in cuts:
            append(list(range(p))+[p+cut['query']], cut['gradient']+[-1], -cut['intercept'])
        matrix = coo_matrix((values, (rid, columns)), shape=(len(b), size)).tocsr()
        eq = coo_matrix((np.ones(p), (np.zeros(p, int), np.arange(p))), shape=(1, size)).tocsr()
        objective = np.r_[-linear_gradient+sparsity_lambda*support_gradient, oracle.alpha, np.full(p, reg), np.zeros(len(extras))]
        fit = linprog(objective, A_ub=matrix, b_ub=b, A_eq=eq, b_eq=[1.], bounds=bounds+extras, method='highs',
            options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
        return fit, matrix, np.asarray(b), objective

    start = actual_objective(initial)
    requested_initial = anchor.copy() if initialization is None else np.asarray(initialization, float)
    if (requested_initial.shape != anchor.shape or not np.isfinite(requested_initial).all()
        or requested_initial.min() < 0 or abs(requested_initial.sum()-mass) > 1e-8
        or (len(C) and not np.all(C@requested_initial >= rhs))
        or not evaluate_class_cell(cell, requested_initial)['feasible']):
        raise ValueError('feasible actual-coordinate initialization with original mass required')
    initial_value = actual_objective(requested_initial/mass)
    use_requested = bool(initial_value <= start)
    current_actual = requested_initial.copy() if use_requested else anchor.copy()
    current = current_actual/mass
    start_point = current_actual.copy(); start_point_value = actual_objective(current)
    current_value = start_point_value; solver_failures = 0; status = 'fixed_major_budget_stop_not_stationarity'
    _, faces = oracle.top_sum(current, 3); add(current, faces)
    for major in range(max_major):
        two, minor_faces, minor_shifts = oracle.minorant_at(current)
        support_gradient, support_intercept = tangent(current, sparsity_delta)
        linear_gradient = sum((alpha*np.asarray(face['gradient']) for alpha, face in zip(oracle.alpha, minor_faces)), start=np.zeros(p))
        linear_intercept = float(sum(alpha*(face['intercept']-shift) for alpha, face, shift in zip(oracle.alpha, minor_faces, minor_shifts)))
        np.testing.assert_allclose(linear_gradient@current+linear_intercept, oracle.alpha@two, atol=1e-9, rtol=1e-9)
        step = {'major': major, 'input_weights': current_actual.tolist(), 'input_objective': current_value,
                'support_gradient':support_gradient.tolist(), 'support_intercept':support_intercept,
                'minor_faces': minor_faces, 'minor_cap_shifts': minor_shifts, 'linear_gradient': linear_gradient.tolist(), 'linear_intercept': linear_intercept,
                'subproblem_trace': []}
        success = False; fit = None; reason = 'subproblem_iteration_budget_exhausted'
        for iteration in range(max_iter):
            fit, matrix, b, objective = master(linear_gradient, support_gradient)
            if not fit.success:
                reason = 'subproblem_master_lp_failed'; break
            candidate = np.maximum(fit.x[:p], 0.); candidate /= candidate.sum()
            actual, repair_trace, blend = repair_cell_metric(C, rhs, cell, candidate*mass, anchor)
            candidate = actual / mass
            three, faces = oracle.top_sum(candidate, 3)
            upper = float(oracle.alpha@three-linear_gradient@candidate-linear_intercept+reg*abs(candidate-initial).sum()+sparsity_lambda*(support_gradient@candidate+support_intercept))
            lower = float(fit.fun-linear_intercept+sparsity_lambda*support_intercept); gap = upper-lower
            step['subproblem_trace'].append({'iteration': iteration, 'cuts': len(cuts), 'upper': upper, 'lower': lower,
                                           'gap': gap, 'canonical_repair_trace': repair_trace, 'anchor_blend': blend, 'class_cell_evaluation': evaluate_class_cell(cell, actual)})
            if -1e-6 <= gap <= gap_tolerance and evaluate_class_cell(cell, actual)['feasible'] and (not len(C) or np.all(C@actual >= rhs)):
                success = True; reason = 'numerical_convex_subproblem_gap_met'; break
            before = len(cuts); add(candidate, faces)
            if len(cuts) == before:
                reason = 'subproblem_duplicate_cut_numerical_stall'; break
        if fit is not None and fit.success:
            step['last_master'] = {'x': fit.x.tolist(), 'objective': float(fit.fun),
                'inequality_marginals': fit.ineqlin.marginals.tolist(), 'equality_marginals': fit.eqlin.marginals.tolist(),
                'lower_marginals': fit.lower.marginals.tolist(), 'upper_marginals': fit.upper.marginals.tolist(),
                'matrix_sha256': hashlib.sha256(matrix.data.tobytes()+matrix.indices.tobytes()+matrix.indptr.tobytes()).hexdigest(),
                'inequality_rows': len(b)}
            step['candidate_weights'] = actual.tolist(); step['candidate_objective'] = actual_objective(candidate)
        step.update({'subproblem_success': success, 'status': reason, 'accepted_descent': False})
        if not success:
            solver_failures += 1; status = reason; steps.append(step); break
        candidate_value = actual_objective(candidate)
        # A tiny increasing numerical solve is never silently accepted.
        if candidate_value >= current_value-1e-10:
            status = 'no_verified_strict_descent_not_global_stationarity'; steps.append(step); break
        assert candidate_value <= upper+1e-8 and upper <= current_value+gap_tolerance
        step['accepted_descent'] = True; steps.append(step)
        current_actual, current, current_value = actual.copy(), candidate.copy(), candidate_value
        _, faces = oracle.top_sum(current, 3); add(current, faces)
    assert actual_objective(current_actual/mass) <= start+1e-10
    assert not len(C) or np.all(C@current_actual >= rhs)
    assert evaluate_class_cell(cell, current_actual)['feasible']
    info = {'features': fs, 'weights': current_actual.tolist(), 'mass': mass, 'constrained': constrained,
        'margin': margin, 'reg': reg, 'sparsity_lambda':sparsity_lambda, 'sparsity_delta':sparsity_delta, 'max_major': max_major, 'max_iter': max_iter, 'gap_tolerance': gap_tolerance,
        'initialization_scope': initialization_scope, 'requested_initial_weights': requested_initial.tolist(),
        'initialization_accepted': use_requested, 'optimization_start_weights': start_point.tolist(),
        'optimization_start_objective': start_point_value, 'requested_initial_objective': initial_value,
        'solver_primal_feasibility_tolerance':1e-9,'solver_dual_feasibility_tolerance':1e-9,
        'class_cell': cell, 'reference_scope': reference_scope,
        'class_cell_final': evaluate_class_cell(cell, current_actual), 'polytope': polytope, 'training_witnesses': oracle.export(), 'cuts': cuts, 'cut_points': points, 'steps': steps,
        'optimizer_success': solver_failures == 0, 'subproblem_failures': solver_failures, 'status': status,
        'training_objective_initial': start, 'training_objective_final': current_value,
        'training_error_upper_bound_initial': float(oracle.alpha@np.minimum(1., oracle.losses(initial))),
        'training_error_upper_bound_final': float(oracle.alpha@np.minimum(1., oracle.losses(current))),
        'scope': 'capped_loss_plus_concave_support_penalty_global_FRS_region_prototype_not_prediction'}
    return current_actual, info
