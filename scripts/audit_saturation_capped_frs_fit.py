"""Class-cell geometry, free epigraph duals and CCCP replayed independently."""
import hashlib
import numpy as np
from scipy.sparse import coo_matrix
from independent_majority_hinge import coefficient
from independent_crossgroup_soft5 import independent_groups
from independent_canonical_polytope import independent_polytope
from independent_canonical_majority import friend_enemy_bounds
from audit_flexible_majority_repair_admission import independent_protected
from audit_forest_compact_transfer import equivalent
from independent_frs_saturation_transition import reconstruct_cell, scalar_cell_evaluation, independent_append_cell_lp


class ScalarRankValidator:
    def __init__(self, A, info):
        self.A, self.info, self.cache = np.asarray(A, float), info, {}

    def coefficients(self, i):
        if i not in self.cache:
            r = self.info['witnesses'][i]
            self.cache[i] = np.asarray([[coefficient(self.A, i, e, f, self.info['eps']) for f in r['friends']] for e in r['enemies']])
        return self.cache[i]

    def hinges(self, i, w):
        C = self.coefficients(i)
        return [max(0., 1-min(float(c@w) for c in row)/self.info['margin']) for row in C]

    def losses(self, w):
        return np.asarray([r['constant_loss'] if r['constant_loss'] is not None else sorted(self.hinges(r['query'], w), reverse=True)[2]
                           for r in self.info['witnesses']])

    def top_sum(self, w, k):
        return np.asarray([(r['constant_loss'] if k == 3 else 0.) if r['constant_loss'] is not None
                           else sum(sorted(self.hinges(r['query'], w), reverse=True)[:k]) for r in self.info['witnesses']])

    def validate(self, point, face):
        i, k = face['query'], face['k']; row = self.info['witnesses'][i]
        assert k in (2, 3)
        if row['constant_loss'] is not None:
            assert not face['enemy_indices'] and not face['worst_friend_indices'] and not face['active_enemy_indices']
            equivalent(face['gradient'], np.zeros(len(point)).tolist())
            equivalent(face['intercept'], row['constant_loss'] if k == 3 else 0.)
            return
        ids, friends, active = face['enemy_indices'], face['worst_friend_indices'], face['active_enemy_indices']
        assert len(ids) == len(set(ids)) == len(friends) == k and len(active) == len(set(active))
        assert set(active) <= set(ids) and all(int(j) == j and 0 <= j < len(row['enemies']) for j in ids)
        assert all(int(f) == f and 0 <= f < 3 for f in friends)
        C = self.coefficients(i); gradient = np.zeros(len(point))
        for j, f in zip(ids, friends):
            if j in active:
                gradient -= C[j, f]/self.info['margin']
        equivalent(gradient.tolist(), face['gradient']); equivalent(float(len(active)), face['intercept'])
        # Every selected affine sum is a global lower face of S_k. At ties
        # the implementation's actual face, rather than a unique face, is checked.
        expected = sum(sorted(self.hinges(i, point), reverse=True)[:k])
        equivalent(float(gradient@point+face['intercept']), expected)


def check_saturation_capped_fit(A, y, record, counts):
    fs, anchor, target, fit = record['features'], np.asarray(record['anchor']), np.asarray(record['target']), record['fit']
    mass = A.shape[1]**.5; initial = anchor/mass; p, n = len(fs), len(A); info = fit['training_witnesses']
    assert fit['margin'] == info['margin'] == .02 and info['eps'] == .01 and fit['reg'] == .01
    assert fit['max_major'] == 8 and fit['max_iter'] == 200 and fit['gap_tolerance'] == 1e-7
    assert target.min() >= 0 and abs(target.sum()-mass) < 1e-8
    equivalent(target.tolist(), fit['weights']); group = independent_groups(y)
    assert group.tolist() == info['group_assignments']
    alpha = np.asarray([1/(len(np.unique(y))*sum(y == c)) for c in y]); equivalent(alpha.tolist(), info['alpha'])
    for i, row in enumerate(info['witnesses']):
        assert row['query'] == i
        refs = [j for j in range(n) if group[j] != group[i]]
        friends = [j for j in refs if y[j] == y[i]]; enemies = [j for j in refs if y[j] != y[i]]
        distance = lambda j: sum(abs(A[j, a]*w-A[i, a]*w) for a, w in zip(fs, initial))
        friends = sorted(friends, key=lambda j: (distance(j), j))[:3]
        assert row['references'] == refs and row['friends'] == friends and row['enemies'] == enemies
        constant = 1. if len(refs) < 5 or len(friends) < 3 else 0. if len(enemies) < 3 else None
        assert row['constant_loss'] == constant
    validator = ScalarRankValidator(A[:, fs], info)
    objective = lambda v: float(alpha@np.minimum(1.,validator.losses(v))+.01*abs(v-initial).sum())
    equivalent(objective(initial), fit['training_objective_initial'])
    equivalent(objective(target/mass), fit['training_objective_final'])
    equivalent(float(alpha@np.minimum(1., validator.losses(initial))), fit['training_error_upper_bound_initial'])
    equivalent(float(alpha@np.minimum(1., validator.losses(target/mass))), fit['training_error_upper_bound_final'])
    C, rhs = np.empty((0, p)), np.empty(0)
    if fit['constrained']:
        original = independent_protected(A, y, np.ones(A.shape[1])/mass); own = np.zeros(n, bool)
        for i in range(n):
            v = friend_enemy_bounds(A[:, fs], y, anchor, i)
            if v is not None:
                raw = np.partition(v, 2)[2]; own[i] = raw-1e-10*(1+abs(raw)) > 1e-10
        ok = all(sum((y == c)&original&own) >= int(np.ceil(.9*sum((y == c)&original)-1e-12)) for c in np.unique(y))
        assert fit['polytope']['protection_reference'] == ('original_full' if ok else 'own_anchor')
        C, rhs = independent_polytope(A[:, fs], y, anchor, original if ok else own, fit['polytope'])
    else:
        assert fit['polytope'] is None
    cell_C, cell_rhs = reconstruct_cell(A, y, fit['class_cell'], counts)
    assert fit['class_cell']['anchor_strictly_feasible']
    C = np.concatenate((C, cell_C)); rhs = np.r_[rhs, cell_rhs]
    if fit['reference_scope'] == 'original_full':
        equivalent(fit['class_cell']['full_weights'], (np.ones(A.shape[1])/mass).tolist())
    else:
        assert fit['reference_scope'] == 'own_anchor'
        wide = np.zeros(A.shape[1]); wide[fs] = anchor
        equivalent(fit['class_cell']['full_weights'], wide.tolist())
    equivalent(scalar_cell_evaluation(fit['class_cell'], target), fit['class_cell_final'])
    assert fit['class_cell_final']['feasible']
    for cut in fit['cuts']:
        assert cut['k'] == 3; validator.validate(np.asarray(fit['cut_points'][cut['point_id']]), cut)
        counts['independent_dc_support_faces'] += 1
    requested = np.asarray(fit['requested_initial_weights'])
    assert requested.shape == anchor.shape and requested.min() >= 0 and abs(requested.sum()-mass) < 1e-8
    assert not len(C) or np.all(C@requested >= rhs)
    assert scalar_cell_evaluation(fit['class_cell'], requested)['feasible']
    equivalent(objective(requested/mass), fit['requested_initial_objective'])
    accepted = bool(fit['requested_initial_objective'] <= fit['training_objective_initial'])
    assert fit['initialization_accepted'] == accepted
    expected_start = requested if accepted else anchor
    equivalent(expected_start.tolist(), fit['optimization_start_weights'])
    current = np.asarray(fit['optimization_start_weights']); failure_count = 0
    equivalent(objective(current/mass), fit['optimization_start_objective'])
    for step in fit['steps']:
        equivalent(current.tolist(), step['input_weights']); point = current/mass
        equivalent(objective(point), step['input_objective'])
        assert len(step['minor_faces']) == n and len(step['minor_cap_shifts']) == n
        two_values, three_values = validator.top_sum(point,2), validator.top_sum(point,3)
        assert all(shift in (0,1) for shift in step['minor_cap_shifts'])
        chosen_values = np.where(np.asarray(step['minor_cap_shifts'],bool), three_values-1., two_values)
        # At machine-precision ties, scalar and batched sums can select
        # different equally active branches. Each global support face and
        # tangency are independently checked, not a bitwise branch identity.
        np.testing.assert_allclose(chosen_values, np.maximum(two_values,three_values-1.), atol=1e-10,rtol=1e-10)
        for i, face in enumerate(step['minor_faces']):
            assert face['query'] == i and face['k'] == (3 if step['minor_cap_shifts'][i] else 2); validator.validate(point, face)
            counts['independent_dc_minor_faces'] += 1
        gradient = sum((q*np.asarray(f['gradient']) for q, f in zip(alpha, step['minor_faces'])), start=np.zeros(p))
        intercept = float(sum(q*(f['intercept']-shift) for q, f, shift in zip(alpha, step['minor_faces'],step['minor_cap_shifts'])))
        equivalent(gradient.tolist(), step['linear_gradient']); equivalent(intercept, step['linear_intercept'])
        if 'last_master' in step:
            master = step['last_master']; rid, columns, values, b = [], [], [], []
            def append(cols, vals, bound):
                rid.extend([len(b)]*len(cols)); columns.extend(cols); values.extend(vals); b.append(bound)
            for j in range(p):
                append([j, p+n+j], [1, -1], initial[j]); append([j, p+n+j], [-1, -1], -initial[j])
            for row, bound in zip(C, rhs):
                append(list(range(p)), (-row*mass).tolist(), -bound)
            size, extras = independent_append_cell_lp(fit['class_cell'], mass, p, n, append)
            used = step['subproblem_trace'][-1]['cuts']
            for cut in fit['cuts'][:used]:
                append(list(range(p))+[p+cut['query']], cut['gradient']+[-1], -cut['intercept'])
            matrix = coo_matrix((values, (rid, columns)), shape=(len(b), size)).tocsr(); b = np.asarray(b)
            assert hashlib.sha256(matrix.data.tobytes()+matrix.indices.tobytes()+matrix.indptr.tobytes()).hexdigest() == master['matrix_sha256']
            assert len(b) == master['inequality_rows']
            x, lam = np.asarray(master['x']), np.asarray(master['inequality_marginals']); sigma = master['equality_marginals'][0]
            low, high = np.asarray(master['lower_marginals']), np.asarray(master['upper_marginals'])
            c = np.r_[-gradient, alpha, np.full(p, .01), np.zeros(len(extras))]; eq = np.r_[np.ones(p), np.zeros(n+p+len(extras))]
            lower = np.r_[np.zeros(p), [r['constant_loss'] if r['constant_loss'] is not None else 0. for r in info['witnesses']], np.zeros(p), [(-np.inf if lower is None else lower) for lower, _ in extras]]
            upper = np.r_[np.ones(p), [r['constant_loss'] if r['constant_loss'] is not None else np.inf for r in info['witnesses']], np.full(p+len(extras), np.inf)]
            assert np.max(matrix@x-b) < 1e-6 and abs(eq@x-1) < 1e-7
            assert (x-lower).min() > -1e-7 and np.all(x[np.isfinite(upper)] <= upper[np.isfinite(upper)]+1e-7)
            assert lam.max() <= 1e-7 and low.min() >= -1e-7 and high.max() <= 1e-7
            assert np.all(abs(low[~np.isfinite(lower)]) <= 1e-9) and np.all(abs(high[~np.isfinite(upper)]) <= 1e-9)
            np.testing.assert_allclose(c-matrix.T@lam-eq*sigma-low-high, 0, atol=1e-6, rtol=0)
            np.testing.assert_allclose(lam*(b-matrix@x), 0, atol=1e-6, rtol=0)
            dual = float(b@lam+sigma+lower[np.isfinite(lower)]@low[np.isfinite(lower)]+upper[np.isfinite(upper)]@high[np.isfinite(upper)])
            np.testing.assert_allclose(c@x, master['objective'], atol=1e-6, rtol=0)
            np.testing.assert_allclose(dual, master['objective'], atol=1e-6, rtol=0)
            v = np.asarray(step['candidate_weights'])/mass
            equivalent(objective(v), step['candidate_objective'])
            trace = step['subproblem_trace'][-1]
            surrogate = float(alpha@validator.top_sum(v, 3)-gradient@v-intercept+.01*abs(v-initial).sum())
            equivalent(surrogate, trace['upper']); equivalent(master['objective']-intercept, trace['lower'])
            equivalent(trace['upper']-trace['lower'], trace['gap'])
            equivalent(scalar_cell_evaluation(fit['class_cell'], step['candidate_weights']), trace['class_cell_evaluation'])
            assert objective(v) <= surrogate+1e-8
            counts['independent_master_primal_dual_replays'] += 1
        if step['subproblem_success']:
            assert -1e-6 <= step['subproblem_trace'][-1]['gap'] <= 1e-7
            assert scalar_cell_evaluation(fit['class_cell'], step['candidate_weights'])['feasible']
            assert not len(C) or np.all(C@np.asarray(step['candidate_weights']) >= rhs)
            if step['accepted_descent']:
                assert step['candidate_objective'] < step['input_objective']-1e-10
                current = np.asarray(step['candidate_weights'])
        else:
            assert not step['accepted_descent']; failure_count += 1
        counts['independent_cccp_steps'] += 1
    equivalent(current.tolist(), target.tolist()); assert objective(target/mass) <= objective(initial)+1e-10
    assert failure_count == fit['subproblem_failures'] and fit['optimizer_success'] == (failure_count == 0)
    counts['target_optimizer_failures'] += failure_count; counts['independent_dc_fit_replays'] += 1
