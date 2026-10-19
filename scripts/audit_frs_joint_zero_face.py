"""Independent joint-zero-face matrices, terminal duals and greedy deployment."""
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, hstack, vstack
from audit_frs_coordinate_minimum import reconstruct_problem
from audit_frs_zero_face_repair import master
from independent_frs_pair_transition_cell import scalar_cell_evaluation as pair_value
from independent_frs_one_sided_stored import scalar_cell_evaluation as one_value
from audit_forest_compact_transfer import equivalent


def direct(info, C, rhs, SC, SR, mode, point):
    p = np.asarray(point, float)
    return bool(p.shape == (len(info['features']),) and np.isfinite(p).all() and p.min() >= 0
        and abs(sum(p)-info['mass']) < 1e-8 and np.all(C@p >= rhs)
        and (mode == 'majority_only' or np.all(SC@p >= SR))
        and (mode not in ('one_sided', 'pair_transition') or
            (one_value if mode == 'one_sided' else pair_value)(info['cell'], p)['feasible']))


def check_face(info, C, rhs, SC, SR, mode, zeros, previous, saved, counts):
    p, mass = len(info['features']), info['mass']; zeros = sorted(set(zeros))
    assert saved['mode'] == mode and saved['zero_indices'] == zeros
    assert saved['zero_features'] == [info['features'][j] for j in zeros]
    equivalent(saved['previous_weights'], previous); assert direct(info, C, rhs, SC, SR, mode, previous)
    M, b, eq, lower, upper = reconstruct_problem(info, C, rhs, SC, SR, mode)
    q = M.shape[1]; lower[zeros] = upper[zeros] = 0.
    scale = 1+abs(b)+np.asarray(abs(M).sum(axis=1)).ravel()
    S = hstack((M, csr_matrix(scale[:, None])), format='csr')
    x = master(saved['margin_master'], S, b, np.r_[eq, 0.], np.r_[lower, 0.], np.r_[upper, 1.],
        np.r_[np.zeros(q), -1.], counts)
    if x is None:
        assert not saved['accepted'] and saved['target'] is None and saved['status'] == 'joint_zero_face_margin_lp_failed'
        return
    gamma = float(x[-1]); equivalent(gamma, saved['maximum_scaled_slack'])
    if gamma <= 1e-12:
        assert not saved['accepted'] and saved['target'] is None
        assert saved['status'] == 'no_verified_positive_joint_zero_face_slack'; return
    tau = min(1e-9, gamma/2); equivalent(tau, saved['conservative_buffer'])
    r, c, v, bb = [], [], [], []
    for j in range(p):
        for sign in (1., -1.):
            r.extend([len(bb)]*2); c.extend([j, q+j]); v.extend([sign, -1.]); bb.append(sign*previous[j]/mass)
    P = vstack((hstack((M, csr_matrix((len(b), p))), format='csr'),
        coo_matrix((v, (r, c)), shape=(2*p, q+p)).tocsr()), format='csr')
    point = master(saved['projection_master'], P, np.r_[b-scale*tau, bb], np.r_[eq, np.zeros(p)],
        np.r_[lower, np.zeros(p)], np.r_[upper, np.full(p, np.inf)], np.r_[np.zeros(q), np.ones(p)], counts)
    if point is None:
        assert not saved['accepted'] and saved['target'] is None
        assert saved['status'] == 'strict_joint_zero_face_projection_lp_failed'; return
    target = point[:p]*mass; equivalent(target.tolist(), saved['projection_candidate_weights'])
    accepted = bool(np.all(target[zeros] == 0.) and direct(info, C, rhs, SC, SR, mode, target))
    assert saved['accepted'] == accepted
    if accepted:
        equivalent(saved['target'], target.tolist()); assert saved['status'] == 'original_constraints_verified_joint_zero_face'
        counts['accepted_joint_zero_faces'] += 1
    else:
        assert saved['target'] is None and saved['status'] == 'joint_projection_actual_coordinate_recheck_failed'


def check_elimination(info, C, rhs, SC, SR, mode, saved, counts):
    current = list(info['anchor']); p = len(current); zeros = [j for j,w in enumerate(current) if w == 0]
    order = sorted(range(p), key=lambda j:(current[j], info['features'][j])); assert saved['order'] == order
    i = 0
    for j in order:
        if j in zeros:
            continue
        row = saved['steps'][i]; i += 1
        assert row['requested_coordinate'] == j and row['requested_feature'] == info['features'][j]
        check_face(info, C, rhs, SC, SR, mode, sorted(zeros+[j]), current, row, counts)
        if row['accepted']:
            current = row['target']; zeros = [k for k,w in enumerate(current) if w == 0]
    assert i == len(saved['steps']); equivalent(current, saved['target'])
    assert zeros == saved['exact_zero_indices']
    assert saved['exact_support_features'] == [info['features'][j] for j in range(p) if current[j] != 0]
    assert direct(info, C, rhs, SC, SR, mode, current)
    assert not saved['classification_scored'] and not saved['goal_achieved']
    counts['joint_elimination_regimes'] += 1
