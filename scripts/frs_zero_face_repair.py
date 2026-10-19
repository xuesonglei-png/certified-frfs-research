"""Conservative interior repair while keeping the requested attribute exactly zero.

No rescaling, clipping or relaxed class/margin constraints. The first LP
maximizes common scaled slack; the second stays near the original candidate
on a stricter feasible zero face. All failed attempts remain explicit.
"""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, coo_matrix, hstack, vstack
from frs_coordinate_minimum import problem, sparse_digest
from soft_topk_predictive_metric import anchor_witness_polytope
from frs_pair_transition_cell import build_class_cell, evaluate_class_cell as pair_value
from frs_one_sided_cell import evaluate_class_cell as one_value


def pack(fit, matrix):
    record = {'success':bool(fit.success), 'status':int(fit.status), 'message':fit.message,
        'matrix_sha256':sparse_digest(matrix), 'inequality_rows':matrix.shape[0], 'variables':matrix.shape[1]}
    if fit.success:
        record.update({'x':fit.x.tolist(), 'objective':float(fit.fun),
            'inequality_marginals':fit.ineqlin.marginals.tolist(),
            'equality_marginals':fit.eqlin.marginals.tolist(),
            'lower_marginals':fit.lower.marginals.tolist(), 'upper_marginals':fit.upper.marginals.tolist()})
    return record


def direct(info, C, rhs, SC, SR, mode, point):
    w = np.asarray(point, float); mass = info['mass']
    if (w.shape != (len(info['features']),) or not np.isfinite(w).all()
        or w.min() < 0 or abs(w.sum()-mass) >= 1e-8 or not np.all(C@w >= rhs)):
        return False
    if mode != 'majority_only' and not np.all(SC@w >= SR):
        return False
    if mode in ('one_sided', 'pair_transition'):
        return bool((one_value if mode == 'one_sided' else pair_value)(info['cell'], w)['feasible'])
    return True


def repair_seed(info, C, rhs, SC, SR, mode, row):
    p, j = len(info['features']), row['coordinate']
    assert row['success'] and row['exact_zero_candidate'] and row['candidate_weights'][j] == 0
    out = {'mode':mode, 'feature':row['feature'], 'coordinate':j,
        'original_candidate_weights':row['candidate_weights'], 'originally_accepted':bool(row['accepted_individual_zero_witness']),
        'reused_accepted':False, 'repaired_accepted':False, 'target':None,
        'scope':'individual_original_constraints_zero_face_repair_not_prediction_or_joint_deletions'}
    if row['accepted_individual_zero_witness']:
        assert direct(info, C, rhs, SC, SR, mode, row['candidate_weights'])
        out.update({'reused_accepted':True, 'target':row['candidate_weights'], 'status':'original_audited_zero_witness_reused'})
        return out
    matrix, b, eq, bounds = problem(info, C, rhs, SC, SR, mode)
    q = matrix.shape[1]; scale = 1+abs(b)+np.asarray(abs(matrix).sum(axis=1)).ravel()
    zero_bounds = list(bounds); zero_bounds[j] = (0., 0.)
    slack_matrix = hstack((matrix, csr_matrix(scale[:, None])), format='csr')
    slack_eq = hstack((eq, csr_matrix((1, 1))), format='csr')
    cost = np.zeros(q+1); cost[-1] = -1.
    settings = {'primal_feasibility_tolerance':1e-9, 'dual_feasibility_tolerance':1e-9}
    margin = linprog(cost, A_ub=slack_matrix, b_ub=b, A_eq=slack_eq, b_eq=[1.],
        bounds=zero_bounds+[(0., 1.)], method='highs', options=settings)
    out['margin_master'] = pack(margin, slack_matrix)
    if not margin.success:
        out['status'] = 'zero_face_margin_lp_failed_no_witness'; return out
    gamma = float(margin.x[-1]); out['maximum_scaled_slack'] = gamma
    if gamma <= 1e-12:
        out['status'] = 'no_verified_positive_zero_face_slack_no_witness'; return out
    tau = float(min(1e-9, gamma/2)); out['conservative_buffer'] = tau
    original = np.asarray(row['candidate_weights'])/info['mass']
    r, c, v, extra_b = [], [], [], []
    for a in range(p):
        for sign in (1., -1.):
            r.extend([len(extra_b)]*2); c.extend([a, q+a]); v.extend([sign, -1.]); extra_b.append(sign*original[a])
    near_matrix = vstack((hstack((matrix, csr_matrix((len(b), p))), format='csr'),
        coo_matrix((v, (r, c)), shape=(2*p, q+p)).tocsr()), format='csr')
    near_b = np.r_[b-scale*tau, extra_b]
    near_eq = hstack((eq, csr_matrix((1, p))), format='csr')
    cost = np.r_[np.zeros(q), np.ones(p)]
    projection = linprog(cost, A_ub=near_matrix, b_ub=near_b, A_eq=near_eq, b_eq=[1.],
        bounds=zero_bounds+[(0., None)]*p, method='highs', options=settings)
    out['projection_master'] = pack(projection, near_matrix)
    if not projection.success:
        out['status'] = 'strict_zero_face_projection_lp_failed_no_witness'; return out
    target = projection.x[:p]*info['mass']; out['projection_candidate_weights'] = target.tolist()
    accepted = bool(target[j] == 0 and direct(info, C, rhs, SC, SR, mode, target))
    out['repaired_accepted'] = accepted
    out['normalized_L1_change_from_original_candidate'] = float(abs(target/info['mass']-original).sum())
    if accepted:
        out['target'] = target.tolist()
    out['status'] = 'strict_actual_coordinate_zero_witness' if accepted else 'strict_projection_actual_coordinate_recheck_failed_no_witness'
    return out


def repair_diagnostic(A, y, diagnostic):
    info = diagnostic['geometry']; fs, anchor = info['features'], np.asarray(info['anchor'])
    C, rhs, _ = anchor_witness_polytope(A, y, fs, anchor)
    SC, SR, check = build_class_cell(A, y, fs, anchor, np.asarray(info['cell']['full_weights']))
    assert check == info['cell']
    result = {'geometry':info, 'modes':{}, 'classification_scored':False, 'goal_achieved':False}
    for mode, value in diagnostic['regimes'].items():
        rows = [r for r in value['coordinates'] if r['success'] and r['exact_zero_candidate']]
        result['modes'][mode] = [repair_seed(info, C, rhs, SC, SR, mode, row) for row in rows]
    return result
