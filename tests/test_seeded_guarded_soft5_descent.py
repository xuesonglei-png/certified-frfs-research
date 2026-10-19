"""Seed guards preserve their reference; replay actual seed and all updates."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from seeded_guarded_soft5_descent import fit_seeded_guarded_soft5
from simplex_soft_topk_targets import fit_simplex_target
from independent_soft_topk_objective import independent_objective
from certified_frfs.feasible_transport import weight_certificate
from audit_classwise_repair_admission import independent_class_certificate


def test_certified_seed_loss_and_all_actual_transitions_are_replayed():
    rng = np.random.default_rng(135); y = np.repeat([0, 1], 12); d = 12
    A = rng.uniform(0, 1, (24, d)); A[:, 0] = .1+.8*y
    anchor = np.ones(d)/np.sqrt(d); fs = list(range(d))
    candidate, _ = fit_simplex_target(A, y, fs, anchor)
    target, fit = fit_seeded_guarded_soft5(A, y, fs, anchor, [candidate, anchor])
    assert fit['guard_reference'] == 'original_full' and fit['selected_seed_id'] is not None
    assert len(fit['seed_checks']) == 12 and len(fit['checks']) <= 64
    previous = independent_objective(A, y, np.array(fit['search_initial_weights'])/np.sqrt(d))
    assert previous < independent_objective(A, y, anchor/np.sqrt(d))-1e-8
    for row in fit['seed_checks']+fit['checks']:
        w = np.array(row['weights']); cert = weight_certificate(A, y, fs, anchor, w)
        accepted = bool(cert['feasible'] and independent_class_certificate(y, cert)['feasible_all_classes'])
        assert accepted == row['accepted']
        np.testing.assert_allclose(independent_objective(A, y, w/np.sqrt(d)), row['training_objective'], atol=1e-9)
    for rid in fit['accepted_step_ids']:
        value = fit['checks'][rid]['training_objective']; assert value < previous-1e-8; previous = value
    np.testing.assert_allclose(independent_objective(A, y, target/np.sqrt(d)), fit['training_objective_final'], atol=1e-9)


def test_own_reference_stays_original_control_anchor_and_bad_seeds_fail():
    rng = np.random.default_rng(135); y = np.repeat([0, 1], 12); d = 12
    A = rng.uniform(0, 1, (24, d)); A[:, 0] = .1+.8*y
    fs = [1, 2]; anchor = np.full(2, np.sqrt(d)/2)
    candidate = np.array([np.sqrt(d), 0.])
    _, fit = fit_seeded_guarded_soft5(A, y, fs, anchor, [candidate], max_iter=4, max_guard_checks=16)
    assert fit['guard_reference'] == 'own_anchor' and not fit['original_initial_guard']['accepted']
    for row in fit['seed_checks']+fit['checks']:
        cert = weight_certificate(A[:, fs], y, [0, 1], anchor, row['weights'])
        assert row['accepted'] == bool(cert['feasible'] and independent_class_certificate(y, cert)['feasible_all_classes'])
    with pytest.raises(ValueError): fit_seeded_guarded_soft5(A, y, fs, anchor, [[1., 0.]])
