"""Conservative joint feature elimination in an unchanged sufficient FRS region.

Fix every previously deleted coordinate together with the newly requested one.
No union of independently feasible solutions is ever treated as a certificate.
Training-only deterministic greedy order, not a minimum-cardinality algorithm.
"""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, coo_matrix, hstack, vstack
from frs_coordinate_minimum import problem
from frs_zero_face_repair import pack, direct


def solve_face(info, C, rhs, SC, SR, mode, zero_indices, previous):
    p, mass = len(info['features']), info['mass']
    zeros = sorted(set(zero_indices)); prev = np.asarray(previous, float)
    if not zeros or any(int(j) != j or not 0 <= j < p for j in zeros):
        raise ValueError('Nonempty valid joint zero coordinate set required')
    if prev.shape != (p,) or not direct(info, C, rhs, SC, SR, mode, prev):
        raise ValueError('Previously verified actual-coordinate feasible point required')
    out = {'mode':mode, 'zero_indices':zeros, 'zero_features':[info['features'][j] for j in zeros],
        'previous_weights':prev.tolist(), 'accepted':False, 'target':None}
    matrix, b, eq, bounds = problem(info, C, rhs, SC, SR, mode)
    q = matrix.shape[1]; scale = 1+abs(b)+np.asarray(abs(matrix).sum(axis=1)).ravel()
    bounds = list(bounds)
    for j in zeros:
        bounds[j] = (0., 0.)
    settings = {'primal_feasibility_tolerance':1e-9, 'dual_feasibility_tolerance':1e-9}
    slack_matrix = hstack((matrix, csr_matrix(scale[:, None])), format='csr')
    slack_eq = hstack((eq, csr_matrix((1, 1))), format='csr')
    cost = np.r_[np.zeros(q), -1.]
    margin = linprog(cost, A_ub=slack_matrix, b_ub=b, A_eq=slack_eq, b_eq=[1.],
        bounds=bounds+[(0., 1.)], method='highs', options=settings)
    out['margin_master'] = pack(margin, slack_matrix)
    if not margin.success:
        out['status'] = 'joint_zero_face_margin_lp_failed'; return out
    gamma = float(margin.x[-1]); out['maximum_scaled_slack'] = gamma
    if gamma <= 1e-12:
        out['status'] = 'no_verified_positive_joint_zero_face_slack'; return out
    tau = min(1e-9, gamma/2); out['conservative_buffer'] = tau
    rows, cols, vals, bb = [], [], [], []
    for j in range(p):
        for s in (1., -1.):
            rows.extend([len(bb)]*2); cols.extend([j, q+j]); vals.extend([s, -1.]); bb.append(s*prev[j]/mass)
    M = vstack((hstack((matrix, csr_matrix((len(b), p))), format='csr'),
        coo_matrix((vals, (rows, cols)), shape=(2*p, q+p)).tocsr()), format='csr')
    ub = np.r_[b-scale*tau, bb]; equality = hstack((eq, csr_matrix((1, p))), format='csr')
    projection = linprog(np.r_[np.zeros(q), np.ones(p)], A_ub=M, b_ub=ub,
        A_eq=equality, b_eq=[1.], bounds=bounds+[(0., None)]*p, method='highs', options=settings)
    out['projection_master'] = pack(projection, M)
    if not projection.success:
        out['status'] = 'strict_joint_zero_face_projection_lp_failed'; return out
    point = projection.x[:p]*mass; out['projection_candidate_weights'] = point.tolist()
    ok = bool(np.all(point[zeros] == 0.) and direct(info, C, rhs, SC, SR, mode, point))
    out['accepted'] = ok
    out['status'] = 'original_constraints_verified_joint_zero_face' if ok else 'joint_projection_actual_coordinate_recheck_failed'
    if ok:
        out['target'] = point.tolist()
    return out


def eliminate(info, C, rhs, SC, SR, mode):
    current = np.asarray(info['anchor']); p = len(current)
    if not direct(info, C, rhs, SC, SR, mode, current):
        raise ValueError('Original feasible anchor required')
    zeros = list(np.flatnonzero(current == 0)); order = sorted(range(p), key=lambda j:(current[j], info['features'][j]))
    history = []
    for j in order:
        if j in zeros:
            continue
        request = sorted(zeros+[j]); outcome = solve_face(info, C, rhs, SC, SR, mode, request, current)
        outcome['requested_coordinate'] = j; outcome['requested_feature'] = info['features'][j]
        history.append(outcome)
        if outcome['accepted']:
            current = np.asarray(outcome['target']); zeros = np.flatnonzero(current == 0).tolist()
    assert direct(info, C, rhs, SC, SR, mode, current)
    return {'mode':mode, 'order':order, 'steps':history, 'target':current.tolist(),
        'exact_zero_indices':zeros, 'exact_support_features':[info['features'][j] for j in range(p) if current[j] != 0],
        'scope':'greedy_training_joint_zero_faces_not_minimum_reduct_or_prediction',
        'classification_scored':False, 'goal_achieved':False}
