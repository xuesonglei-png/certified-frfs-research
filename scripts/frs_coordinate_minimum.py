"""Training-only diagnostic: which constraints prevent an attribute reaching zero?

This is a coordinate-wise LP lower bound within a sufficient fixed cell,
not a minimum-cardinality reduct and not a classifier. Individual feasible
deletions cannot be combined without rechecking the common constraints.
"""
import hashlib
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from soft_topk_predictive_metric import anchor_witness_polytope
from frs_pair_transition_cell import build_class_cell, append_class_cell_lp as pair_lp
from frs_one_sided_cell import append_class_cell_lp as one_sided_lp
from frs_pair_transition_cell import evaluate_class_cell as pair_value
from frs_one_sided_cell import evaluate_class_cell as one_sided_value

MODES = ('majority_only', 'stable_cell_only', 'one_sided', 'pair_transition')


def sparse_digest(matrix):
    return hashlib.sha256(matrix.data.tobytes()+matrix.indices.tobytes()+matrix.indptr.tobytes()).hexdigest()


def geometry(A, y, features, anchor):
    A, y, anchor = np.asarray(A, float), np.asarray(y), np.asarray(anchor, float)
    if not features or features != sorted(set(features)) or anchor.shape != (len(features),):
        raise ValueError('valid nonempty ordered support required')
    mass = float(np.sqrt(A.shape[1]))
    if not np.isfinite(anchor).all() or min(anchor) < 0 or abs(anchor.sum()-mass) > 1e-8:
        raise ValueError('original nonnegative mass required')
    full = np.ones(A.shape[1])/mass
    C, rhs, polytope = anchor_witness_polytope(A, y, features, anchor)
    stable_C, stable_rhs, cell = build_class_cell(A, y, features, anchor, full)
    if not cell['anchor_strictly_feasible']:
        raise ValueError('strict original-full cell required for this diagnostic')
    return {'features':list(features), 'anchor':anchor.tolist(), 'mass':mass,
            'polytope':polytope, 'cell':cell}, C, rhs, stable_C, stable_rhs


def problem(info, C, rhs, stable_C, stable_rhs, mode):
    if mode not in MODES:
        raise ValueError('invalid diagnostic regime')
    p, mass = len(info['features']), info['mass']; rows, cols, vals, b = [], [], [], []
    def append(c, v, bound):
        rows.extend([len(b)]*len(c)); cols.extend(c); vals.extend(v); b.append(float(bound))
    initial = np.asarray(info['anchor'])/mass
    for j in range(p):
        append([j, p+j], [1., -1.], initial[j])
        append([j, p+j], [-1., -1.], -initial[j])
    for row, floor in zip(C, rhs):
        append(list(range(p)), (-mass*row).tolist(), -floor)
    if mode != 'majority_only':
        for row, floor in zip(stable_C, stable_rhs):
            append(list(range(p)), (-mass*row).tolist(), -floor)
    size, extras = 2*p, []
    if mode in ('one_sided', 'pair_transition'):
        function = one_sided_lp if mode == 'one_sided' else pair_lp
        size, extras = function(info['cell'], mass, p, 0, append)
    matrix = coo_matrix((vals, (rows, cols)), shape=(len(b), size)).tocsr()
    eq = coo_matrix((np.ones(p), (np.zeros(p, int), np.arange(p))), shape=(1, size)).tocsr()
    bounds = [(0., 1.)]*p+[(0., None)]*p+extras
    return matrix, np.asarray(b), eq, bounds


def solve_regime(info, C, rhs, stable_C, stable_rhs, mode):
    matrix, b, eq, bounds = problem(info, C, rhs, stable_C, stable_rhs, mode)
    p, mass = len(info['features']), info['mass']; minima = []
    for j in range(p):
        objective = np.zeros(matrix.shape[1]); objective[j] = 1.
        result = linprog(objective, A_ub=matrix, b_ub=b, A_eq=eq, b_eq=[1.],
            bounds=bounds, method='highs', options={'primal_feasibility_tolerance':1e-9,
                                                  'dual_feasibility_tolerance':1e-9})
        row = {'feature':info['features'][j], 'coordinate':j,
            'success':bool(result.success), 'status':int(result.status), 'message':result.message}
        if result.success:
            actual = np.asarray(result.x[:p])*mass
            original_majority = bool(np.all(C@actual >= rhs))
            original_stable = bool(np.all(stable_C@actual >= stable_rhs))
            report = (one_sided_value if mode == 'one_sided' else pair_value)(info['cell'], actual) if mode in ('one_sided', 'pair_transition') else None
            zero = bool(actual[j] == 0.)
            direct = bool(actual.min() >= 0 and abs(actual.sum()-mass) < 1e-8 and original_majority
                and (mode == 'majority_only' or original_stable)
                and (report is None or report['feasible']))
            row.update({'actual_minimum_weight':float(actual[j]), 'candidate_weights':actual.tolist(),
                'exact_zero_candidate':zero, 'original_constraints_passed_at_actual_coordinates':direct,
                'accepted_individual_zero_witness':bool(zero and direct), 'cell_evaluation':report,
                'objective':float(result.fun), 'x':result.x.tolist(),
                'inequality_marginals':result.ineqlin.marginals.tolist(),
                'equality_marginals':result.eqlin.marginals.tolist(),
                'lower_marginals':result.lower.marginals.tolist(),
                'upper_marginals':result.upper.marginals.tolist()})
        minima.append(row)
    return {'mode':mode, 'matrix_sha256':sparse_digest(matrix), 'inequality_rows':len(b),
        'variables':matrix.shape[1], 'coordinates':minima, 'solver_primal_tolerance':1e-9,
        'solver_dual_tolerance':1e-9, 'failures':sum(not r['success'] for r in minima)}


def solve_diagnostic(A, y, features, anchor):
    info, C, rhs, SC, SR = geometry(A, y, features, anchor)
    return {'geometry':info, 'regimes':{m:solve_regime(info, C, rhs, SC, SR, m) for m in MODES},
        'classification_scored':False, 'goal_achieved':False,
        'scope':'per_coordinate_fixed_cell_LP_diagnostic_not_joint_reduct_or_general_necessity'}
