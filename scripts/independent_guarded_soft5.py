"""Independent objective/reference/witness/transition replay, not search refit."""
import numpy as np
from certified_frfs.feasible_transport import weight_certificate
from audit_flexible_majority_repair_admission import independent_protected
from audit_classwise_repair_admission import independent_class_certificate
from audit_forest_compact_transfer import equivalent
from independent_canonical_majority import friend_enemy_bounds
from independent_canonical_polytope import independent_polytope
from independent_soft_topk_objective import independent_objective


def check_guarded_learner(A, y, record, counts):
    fs, anchor, target, fit = record['features'], np.array(record['anchor']), np.array(record['target']), record['fit']
    d, mass = A.shape[1], np.sqrt(A.shape[1]); full = np.ones(d)/mass
    assert fit['features'] == fs and fit['mass'] == mass and fit['cap_normalized'] == 1
    assert fit['temperature'] == .05 and fit['reg'] == .01 and fit['neighborhood_k'] == 5
    assert fit['max_iter'] == 4 and fit['max_guard_checks'] == 16
    equivalent(fit['weights'], target.tolist())
    for w in (anchor, target):
        np.testing.assert_allclose(w.sum(), mass, atol=1e-10)
        assert w.min() >= 0 and w.max()/mass <= 1+1e-10
    original_cert = weight_certificate(A, y, fs, full, anchor)
    original_cls = independent_class_certificate(y, original_cert)
    original_initial = {'certificate': original_cert, 'class_certificate': original_cls,
                        'accepted': bool(original_cert['feasible'] and original_cls['feasible_all_classes'])}
    equivalent(original_initial, fit['original_initial_guard'])
    reference = 'original_full' if original_initial['accepted'] else 'own_anchor'
    assert fit['guard_reference'] == reference

    def guard(w, saved):
        fresh = (weight_certificate(A, y, fs, full, w) if reference == 'original_full'
                 else weight_certificate(A[:, fs], y, list(range(len(fs))), anchor, w))
        classes = independent_class_certificate(y, fresh)
        equivalent(fresh, saved['certificate']); equivalent(classes, saved['class_certificate'])
        accepted = bool(fresh['feasible'] and classes['feasible_all_classes'])
        assert accepted == saved['accepted']; counts['independent_joint_guard_checks'] += 1
        return accepted

    assert guard(anchor, fit['initial_guard'])
    protected = independent_protected(A, y, full); own = np.zeros(len(A), bool)
    for i in range(len(A)):
        values = friend_enemy_bounds(A[:, fs], y, anchor, i)
        if values is not None:
            raw = np.partition(values, 2)[2]; own[i] = raw-1e-10*(1+abs(raw)) > 1e-10
    original_majority_ok = all(sum((y == c) & protected & own) >= int(np.ceil(.9*sum((y == c) & protected)-1e-12)) for c in np.unique(y))
    witness_reference = 'original_full' if original_majority_ok else 'own_anchor'
    assert fit['polytope']['protection_reference'] == witness_reference
    if not original_majority_ok: protected = own
    C, rhs = independent_polytope(A[:, fs], y, anchor, protected, fit['polytope'])
    counts['independent_fixed_witness_rows'] += len(C)
    initial = independent_objective(A[:, fs], y, anchor/mass)
    equivalent(initial, fit['training_objective_initial'])
    assert len(fit['checks']) <= 16
    for row in fit['checks']:
        w = np.array(row['weights']); np.testing.assert_allclose(w.sum(), mass, atol=1e-10)
        assert w.min() >= 0 and w.max()/mass <= 1+1e-10
        assert not len(C) or np.all(C@w >= rhs)
        assert row['receiver'] in fs and row['donor'] in fs and row['receiver'] != row['donor']
        assert row['normalized_step'] > 0 and 0 <= row['iteration'] < 4
        equivalent(independent_objective(A[:, fs], y, w/mass), row['training_objective'])
        guard(w, row); counts['independent_joint_search_objectives'] += 1
    steps = fit['accepted_step_ids']; assert steps == sorted(set(steps)) and len(steps) <= 4
    previous_value, previous_weights = initial, anchor
    for iteration, rid in enumerate(steps):
        row = fit['checks'][rid]; assert row['iteration'] == iteration and row['accepted']
        assert row['training_objective'] < previous_value-1e-8
        assert row['training_objective'] == min(q['training_objective'] for q in fit['checks'] if q['iteration'] == iteration and q['accepted'])
        expected = np.zeros(len(fs))
        expected[fs.index(row['receiver'])] = row['normalized_step']*mass
        expected[fs.index(row['donor'])] = -row['normalized_step']*mass
        np.testing.assert_allclose(np.array(row['weights'])-previous_weights, expected, atol=1e-10)
        previous_value, previous_weights = row['training_objective'], np.array(row['weights'])
    equivalent(previous_weights.tolist(), target.tolist()); equivalent(previous_value, fit['training_objective_final'])
    equivalent(independent_objective(A[:, fs], y, target/mass), fit['training_objective_final'])
    assert guard(target, fit['endpoint_guard'])
    assert fit['status'] in ('iteration_budget_exhausted', 'guard_budget_exhausted', 'no_improving_checked_pair')
    counts['joint_guarded_learners'] += 1; counts['joint_guarded_accepted_steps'] += len(steps)
    counts['joint_guarded_search_checks'] += len(fit['checks'])
    counts['joint_guard_reference_'+reference] += 1
    counts['joint_guarded_status_'+fit['status']] += 1
