"""Scalar geometry and independent LP primal/dual diagnostic verification."""
import numpy as np
from scipy.sparse import coo_matrix
from independent_canonical_polytope import independent_polytope
from audit_flexible_majority_repair_admission import independent_protected
from independent_canonical_majority import friend_enemy_bounds
from independent_frs_pair_transition_cell import (reconstruct_cell,
    independent_append_cell_lp as pair_lp, scalar_cell_evaluation as pair_value)
from independent_frs_one_sided_cell import independent_append_cell_lp as one_lp
from independent_frs_one_sided_stored import scalar_cell_evaluation as one_value
from audit_forest_compact_transfer import equivalent
from frs_coordinate_minimum import sparse_digest

MODES = ('majority_only', 'stable_cell_only', 'one_sided', 'pair_transition')


def reconstruct_geometry(A, y, info, counts):
    fs, anchor, mass = info['features'], np.asarray(info['anchor']), float(np.sqrt(A.shape[1]))
    equivalent(info['mass'], mass)
    full = np.ones(A.shape[1])/mass; protected = independent_protected(A, y, full)
    own = np.zeros(len(A), bool)
    for i in range(len(A)):
        v = friend_enemy_bounds(A[:, fs], y, anchor, i)
        if v is not None:
            raw = np.partition(v, 2)[2]
            own[i] = raw-1e-10*(1+abs(raw)) > 1e-10
    original_ok = all(sum((y == c)&protected&own) >= int(np.ceil(.9*sum((y == c)&protected)-1e-12)) for c in np.unique(y))
    assert info['polytope']['protection_reference'] == ('original_full' if original_ok else 'own_anchor')
    C, rhs = independent_polytope(A[:, fs], y, anchor, protected if original_ok else own, info['polytope'])
    SC, SR = reconstruct_cell(A, y, info['cell'], counts)
    equivalent(info['cell']['full_weights'], full.tolist())
    assert info['cell']['anchor_strictly_feasible']
    return C, rhs, SC, SR


def reconstruct_problem(info, C, rhs, SC, SR, mode):
    p, mass = len(info['features']), info['mass']; rows, cols, values, rhs_rows = [], [], [], []
    def add(c, v, b):
        rows.extend([len(rhs_rows)]*len(c)); cols.extend(c); values.extend(v); rhs_rows.append(float(b))
    for j, v in enumerate(info['anchor']):
        add([j, p+j], [1., -1.], v/mass); add([j, p+j], [-1., -1.], -v/mass)
    for row, floor in zip(C, rhs):
        add(list(range(p)), [-mass*v for v in row], -floor)
    if mode != 'majority_only':
        for row, floor in zip(SC, SR):
            add(list(range(p)), [-mass*v for v in row], -floor)
    size, extras = 2*p, []
    if mode in ('one_sided', 'pair_transition'):
        size, extras = (one_lp if mode == 'one_sided' else pair_lp)(info['cell'], mass, p, 0, add)
    matrix = coo_matrix((values, (rows, cols)), shape=(len(rhs_rows), size)).tocsr()
    lower = np.r_[np.zeros(2*p), [-np.inf if a is None else a for a, _ in extras]]
    upper = np.r_[np.ones(p), np.full(p, np.inf), [np.inf if b is None else b for _, b in extras]]
    equality = np.r_[np.ones(p), np.zeros(size-p)]
    return matrix, np.asarray(rhs_rows), equality, lower, upper


def audit_diagnostic(A, y, saved, counts):
    assert not saved['classification_scored'] and not saved['goal_achieved']
    info = saved['geometry']; C, rhs, SC, SR = reconstruct_geometry(A, y, info, counts)
    p, mass = len(info['features']), info['mass']; assert tuple(saved['regimes']) == MODES
    for mode in MODES:
        r = saved['regimes'][mode]; matrix, b, eq, lower, upper = reconstruct_problem(info, C, rhs, SC, SR, mode)
        assert r['mode'] == mode and r['matrix_sha256'] == sparse_digest(matrix)
        assert r['inequality_rows'] == len(b) and r['variables'] == matrix.shape[1]
        assert r['solver_primal_tolerance'] == r['solver_dual_tolerance'] == 1e-9
        assert len(r['coordinates']) == p; failures = 0
        for j, row in enumerate(r['coordinates']):
            assert row['feature'] == info['features'][j] and row['coordinate'] == j
            if not row['success']:
                assert row['status'] != 0; failures += 1; counts['LP_failures'] += 1; continue
            assert row['status'] == 0
            x, lam, low, high = [np.asarray(row[k]) for k in ('x', 'inequality_marginals', 'lower_marginals', 'upper_marginals')]
            sigma = row['equality_marginals'][0]; c = np.zeros(matrix.shape[1]); c[j] = 1.
            assert np.max(matrix@x-b, initial=-np.inf) < 1e-6 and abs(eq@x-1) < 1e-7
            assert min(x-lower) > -1e-7 and np.all(x[np.isfinite(upper)] <= upper[np.isfinite(upper)]+1e-7)
            assert np.max(lam, initial=-np.inf) <= 1e-7 and min(low) >= -1e-7 and max(high) <= 1e-7
            assert np.all(abs(low[~np.isfinite(lower)]) <= 1e-9) and np.all(abs(high[~np.isfinite(upper)]) <= 1e-9)
            np.testing.assert_allclose(c-matrix.T@lam-eq*sigma-low-high, 0., atol=1e-6, rtol=0)
            np.testing.assert_allclose(lam*(b-matrix@x), 0., atol=1e-6, rtol=0)
            dual = float(b@lam+sigma+lower[np.isfinite(lower)]@low[np.isfinite(lower)]+upper[np.isfinite(upper)]@high[np.isfinite(upper)])
            np.testing.assert_allclose(c@x, row['objective'], atol=1e-6, rtol=0)
            np.testing.assert_allclose(dual, row['objective'], atol=1e-6, rtol=0)
            actual = mass*x[:p]; equivalent(actual.tolist(), row['candidate_weights'])
            equivalent(float(actual[j]), row['actual_minimum_weight'])
            zero = bool(actual[j] == 0.); assert row['exact_zero_candidate'] == zero
            report = (one_value if mode == 'one_sided' else pair_value)(info['cell'], actual) if mode in ('one_sided', 'pair_transition') else None
            equivalent(report, row['cell_evaluation'])
            direct = bool(actual.min() >= 0 and abs(actual.sum()-mass) < 1e-8 and np.all(C@actual >= rhs)
                and (mode == 'majority_only' or np.all(SC@actual >= SR)) and (report is None or report['feasible']))
            assert row['original_constraints_passed_at_actual_coordinates'] == direct
            assert row['accepted_individual_zero_witness'] == bool(zero and direct)
            counts['independent_coordinate_LP_primal_dual_replays'] += 1
        assert r['failures'] == failures
    counts['diagnostic_training_folds'] += 1
