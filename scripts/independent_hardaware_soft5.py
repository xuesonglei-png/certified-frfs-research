"""Independent hard votes, objectives, seed selection and guarded transitions."""
import numpy as np
from certified_frfs.feasible_transport import weight_certificate
from audit_flexible_majority_repair_admission import independent_protected
from audit_classwise_repair_admission import independent_class_certificate
from audit_forest_compact_transfer import equivalent
from independent_canonical_majority import friend_enemy_bounds
from independent_canonical_polytope import independent_polytope
from independent_soft_topk_objective import independent_objective
from audit_hardaware_guarded_soft5_admission import independent_hard_BA, improves


def check_hardaware_learner(A, y, record, counts):
    fs, anchor, target, fit = record['features'], np.array(record['anchor']), np.array(record['target']), record['fit']
    d, mass = A.shape[1], np.sqrt(A.shape[1]); full = np.ones(d)/mass
    assert fit['features'] == fs and fit['mass'] == mass and fit['cap_normalized'] == 1
    assert fit['temperature'] == .05 and fit['reg'] == .01 and fit['neighborhood_k'] == 5
    assert fit['max_iter'] == 4 and fit['max_guard_checks'] == 16
    assert fit['criterion'] == 'lexicographic_training_hard5_LOO_BA_then_soft5_loss'
    np.testing.assert_array_equal(fit['weights'], target)
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
        assert accepted == saved['accepted']; counts['independent_hardaware_guard_checks'] += 1
        return accepted

    def criterion(w, row=None):
        loss = independent_objective(A[:, fs], y, np.array(w)/mass)
        ba = independent_hard_BA(A[:, fs], y, w)
        if row is not None:
            equivalent(loss, row['training_objective']); equivalent(ba, row['hard_LOO_BA'])
        counts['independent_hardaware_criterion_replays'] += 1
        return ba, loss

    assert guard(anchor, fit['initial_guard'])
    initial_ba, initial_loss = criterion(anchor)
    equivalent(initial_ba, fit['initial_hard_LOO_BA']); equivalent(initial_loss, fit['training_objective_initial'])
    candidates = np.asarray(fit['candidate_targets']); assert candidates.shape in ((1, len(fs)), (2, len(fs)))
    assert candidates.min() >= 0
    np.testing.assert_allclose(candidates.sum(axis=1), mass, atol=1e-10)
    grid = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
    equivalent(grid, fit['seed_interpolations']); assert len(fit['seed_checks']) == 6*len(candidates)
    selected, seed = None, anchor
    seed_ba, seed_loss, seed_guard = initial_ba, fit['training_objective_initial'], fit['initial_guard']
    for sid, row in enumerate(fit['seed_checks']):
        source, t = sid//6, grid[sid%6]
        assert row['source_target_index'] == source and row['interpolation'] == t
        weights = (1-t)*anchor+t*candidates[source]
        np.testing.assert_array_equal(weights, row['weights'])
        ba, loss = criterion(weights, row); guard(weights, row)
        assert row['hard_nonworsening_from_anchor'] == (ba >= initial_ba-1e-12)
        if row['accepted'] and improves(ba, row['training_objective'], seed_ba, seed_loss):
            selected, seed, seed_ba, seed_loss, seed_guard = sid, weights, ba, row['training_objective'], row
    assert selected == fit['selected_seed_id']
    np.testing.assert_array_equal(seed, fit['search_initial_weights'])
    equivalent(seed_ba, fit['search_initial_hard_LOO_BA']); equivalent(seed_loss, fit['search_initial_objective'])
    for key in ('certificate', 'class_certificate', 'accepted'): equivalent(fit['seed_guard'][key], seed_guard[key])

    # Witness reference depends on the ACTUAL seed; guard reference never resets.
    protected = independent_protected(A, y, full); own = np.zeros(len(A), bool)
    for i in range(len(A)):
        values = friend_enemy_bounds(A[:, fs], y, seed, i)
        if values is not None:
            raw = np.partition(values, 2)[2]; own[i] = raw-1e-10*(1+abs(raw)) > 1e-10
    original_ok = all(sum((y == c) & protected & own) >= int(np.ceil(.9*sum((y == c) & protected)-1e-12)) for c in np.unique(y))
    assert fit['polytope']['protection_reference'] == ('original_full' if original_ok else 'own_anchor')
    if not original_ok: protected = own
    C, rhs = independent_polytope(A[:, fs], y, seed, protected, fit['polytope'])
    counts['independent_fixed_witness_rows'] += len(C)
    assert len(fit['checks']) <= 16
    for row in fit['checks']:
        w = np.array(row['weights']); np.testing.assert_allclose(w.sum(), mass, atol=1e-10)
        assert w.min() >= 0 and w.max()/mass <= 1+1e-10
        assert not len(C) or np.all(C@w >= rhs)
        assert row['receiver'] in fs and row['donor'] in fs and row['receiver'] != row['donor']
        assert row['normalized_step'] > 0 and 0 <= row['iteration'] < 4
        criterion(w, row); guard(w, row)
    steps = fit['accepted_step_ids']; assert steps == sorted(set(steps)) and len(steps) <= 4
    previous_ba, previous_loss, previous_weights = seed_ba, seed_loss, seed
    iterations = sorted(set(q['iteration'] for q in fit['checks']))
    assert iterations == list(range(1+max(iterations, default=-1)))
    for iteration in iterations:
        best = None
        for rid, row in enumerate(fit['checks']):
            if row['iteration'] != iteration: continue
            expected = np.zeros(len(fs))
            expected[fs.index(row['receiver'])] = row['normalized_step']*mass
            expected[fs.index(row['donor'])] = -row['normalized_step']*mass
            np.testing.assert_allclose(np.array(row['weights'])-previous_weights, expected, atol=1e-10)
            admissible = bool(row['accepted'] and improves(row['hard_LOO_BA'], row['training_objective'], previous_ba, previous_loss))
            assert row['update_admissible'] == admissible
            if admissible and (best is None or improves(row['hard_LOO_BA'], row['training_objective'], best[1]['hard_LOO_BA'], best[1]['training_objective'])):
                best = (rid, row)
        selected_ids = [rid for rid in steps if fit['checks'][rid]['iteration'] == iteration]
        if best is None:
            assert not selected_ids
            assert iteration == max(iterations)
            continue
        rid, row = best; assert selected_ids == [rid]
        expected = np.zeros(len(fs))
        expected[fs.index(row['receiver'])] = row['normalized_step']*mass
        expected[fs.index(row['donor'])] = -row['normalized_step']*mass
        np.testing.assert_allclose(np.array(row['weights'])-previous_weights, expected, atol=1e-10)
        previous_ba, previous_loss, previous_weights = row['hard_LOO_BA'], row['training_objective'], np.array(row['weights'])
    np.testing.assert_array_equal(previous_weights, target)
    equivalent(previous_ba, fit['final_hard_LOO_BA']); equivalent(previous_loss, fit['training_objective_final'])
    ba, loss = criterion(target)
    equivalent(ba, previous_ba); equivalent(loss, previous_loss)
    assert ba >= initial_ba-1e-12 and guard(target, fit['endpoint_guard'])
    assert fit['status'] in ('iteration_budget_exhausted', 'guard_budget_exhausted', 'no_improving_checked_pair')
    counts['hardaware_learners'] += 1
    counts['hardaware_search_checks'] += len(fit['checks'])
    counts['hardaware_seed_checks'] += len(fit['seed_checks'])
    counts['hardaware_accepted_steps'] += len(steps)
    counts['hardaware_guard_reference_'+reference] += 1
    counts['hardaware_status_'+fit['status']] += 1
