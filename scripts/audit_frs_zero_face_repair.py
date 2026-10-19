"""Independent reconstruction of conservative zero-face LPs and original checks."""
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, hstack, vstack
from audit_frs_coordinate_minimum import reconstruct_geometry, reconstruct_problem
from independent_frs_pair_transition_cell import scalar_cell_evaluation as pair_value
from independent_frs_one_sided_stored import scalar_cell_evaluation as one_value
from frs_coordinate_minimum import sparse_digest
from audit_forest_compact_transfer import equivalent


def master(record, matrix, b, eq, lower, upper, objective, counts):
    assert record['matrix_sha256'] == sparse_digest(matrix)
    assert record['inequality_rows'] == len(b) and record['variables'] == matrix.shape[1]
    if not record['success']:
        assert record['status'] != 0; counts['zero_repair_solver_failures'] += 1; return None
    assert record['status'] == 0
    x, lam, lo, hi = [np.asarray(record[k]) for k in ('x', 'inequality_marginals', 'lower_marginals', 'upper_marginals')]
    sigma = record['equality_marginals'][0]
    assert np.max(matrix@x-b, initial=-np.inf) < 1e-6 and abs(eq@x-1) < 1e-7
    assert min(x-lower) > -1e-7 and np.all(x[np.isfinite(upper)] <= upper[np.isfinite(upper)]+1e-7)
    assert np.max(lam, initial=-np.inf) <= 1e-7 and min(lo) >= -1e-7 and max(hi) <= 1e-7
    assert np.all(abs(lo[~np.isfinite(lower)]) < 1e-9) and np.all(abs(hi[~np.isfinite(upper)]) < 1e-9)
    np.testing.assert_allclose(objective-matrix.T@lam-eq*sigma-lo-hi, 0., atol=1e-6, rtol=0)
    np.testing.assert_allclose(lam*(b-matrix@x), 0., atol=1e-6, rtol=0)
    dual = float(b@lam+sigma+lower[np.isfinite(lower)]@lo[np.isfinite(lower)]+upper[np.isfinite(upper)]@hi[np.isfinite(upper)])
    np.testing.assert_allclose(objective@x, record['objective'], atol=1e-6, rtol=0)
    np.testing.assert_allclose(dual, record['objective'], atol=1e-6, rtol=0)
    counts['independent_zero_repair_LP_duals'] += 1
    return x


def check_seed(info, C, rhs, SC, SR, mode, original_row, saved, counts):
    p, j, mass = len(info['features']), original_row['coordinate'], info['mass']
    assert saved['mode'] == mode and saved['coordinate'] == j and saved['feature'] == original_row['feature']
    equivalent(saved['original_candidate_weights'], original_row['candidate_weights'])
    assert saved['originally_accepted'] == original_row['accepted_individual_zero_witness']
    def direct(point):
        v = np.asarray(point)
        return bool(v.shape == (p,) and np.isfinite(v).all() and v.min() >= 0
            and abs(v.sum()-mass) < 1e-8 and np.all(C@v >= rhs)
            and (mode == 'majority_only' or np.all(SC@v >= SR))
            and (mode not in ('one_sided', 'pair_transition') or
                 (one_value if mode == 'one_sided' else pair_value)(info['cell'], v)['feasible']))
    if original_row['accepted_individual_zero_witness']:
        assert saved['reused_accepted'] and not saved['repaired_accepted']
        equivalent(saved['target'], original_row['candidate_weights'])
        assert direct(saved['target']) and saved['target'][j] == 0
        assert 'margin_master' not in saved; counts['previous_audited_zero_witnesses_reused'] += 1; return
    assert not saved['reused_accepted']
    matrix, b, eq, lower, upper = reconstruct_problem(info, C, rhs, SC, SR, mode)
    lower[j] = upper[j] = 0.; q = matrix.shape[1]
    scale = 1+abs(b)+np.asarray(abs(matrix).sum(axis=1)).ravel()
    M = hstack((matrix, csr_matrix(scale[:, None])), format='csr')
    c = np.zeros(q+1); c[-1] = -1.
    x = master(saved['margin_master'], M, b, np.r_[eq, 0.], np.r_[lower, 0.], np.r_[upper, 1.], c, counts)
    if x is None:
        assert not saved['repaired_accepted'] and saved['target'] is None
        assert saved['status'] == 'zero_face_margin_lp_failed_no_witness'; return
    gamma = float(x[-1]); equivalent(gamma, saved['maximum_scaled_slack'])
    if gamma <= 1e-12:
        assert not saved['repaired_accepted'] and saved['target'] is None
        assert saved['status'] == 'no_verified_positive_zero_face_slack_no_witness'; return
    tau = min(1e-9, gamma/2); equivalent(tau, saved['conservative_buffer'])
    base = np.asarray(original_row['candidate_weights'])/mass
    rows, cols, values, bb = [], [], [], []
    for a in range(p):
        for sign in (1., -1.):
            rows.extend([len(bb)]*2); cols.extend([a, q+a]); values.extend([sign, -1.]); bb.append(sign*base[a])
    P = vstack((hstack((matrix, csr_matrix((len(b), p))), format='csr'),
        coo_matrix((values, (rows, cols)), shape=(2*p, q+p)).tocsr()), format='csr')
    pb = np.r_[b-scale*tau, bb]
    px = master(saved['projection_master'], P, pb, np.r_[eq, np.zeros(p)],
        np.r_[lower, np.zeros(p)], np.r_[upper, np.full(p, np.inf)], np.r_[np.zeros(q), np.ones(p)], counts)
    if px is None:
        assert not saved['repaired_accepted'] and saved['target'] is None
        assert saved['status'] == 'strict_zero_face_projection_lp_failed_no_witness'; return
    point = px[:p]*mass; equivalent(point.tolist(), saved['projection_candidate_weights'])
    accepted = bool(point[j] == 0 and direct(point)); assert saved['repaired_accepted'] == accepted
    equivalent(float(abs(point/mass-base).sum()), saved['normalized_L1_change_from_original_candidate'])
    if accepted:
        equivalent(saved['target'], point.tolist())
        assert saved['status'] == 'strict_actual_coordinate_zero_witness'; counts['new_repaired_zero_witnesses'] += 1
    else:
        assert saved['target'] is None and saved['status'] == 'strict_projection_actual_coordinate_recheck_failed_no_witness'


def check_diagnostic(A, y, original, saved, counts):
    equivalent(original['geometry'], saved['geometry'])
    assert not saved['classification_scored'] and not saved['goal_achieved']
    info = saved['geometry']; C, rhs, SC, SR = reconstruct_geometry(A, y, info, counts)
    assert tuple(saved['modes']) == tuple(original['regimes'])
    for mode, old in original['regimes'].items():
        expected = [r for r in old['coordinates'] if r['success'] and r['exact_zero_candidate']]
        assert len(expected) == len(saved['modes'][mode])
        for row, outcome in zip(expected, saved['modes'][mode], strict=True):
            check_seed(info, C, rhs, SC, SR, mode, row, outcome, counts)
    counts['zero_repair_training_folds'] += 1
