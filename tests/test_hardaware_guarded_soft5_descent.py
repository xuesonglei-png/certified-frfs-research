"""A valid FRS certificate does not imply nondecreasing actual hard accuracy."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from hardaware_guarded_soft5_descent import fit_hardaware_guarded_soft5, hard5_LOO_BA, criterion_improves
from diagnose_soft5_calibration import profile
from certified_frfs.feasible_transport import weight_certificate
from audit_classwise_repair_admission import independent_class_certificate


def test_frs_admitted_but_hard_accuracy_worsening_seed_is_not_selected():
    A = np.random.default_rng(12).uniform(0, 1, (20, 12)); y = np.repeat([0, 1], 10)
    fs = list(range(12)); anchor = np.ones(12)/np.sqrt(12)
    candidate = anchor.copy(); candidate[0] += .03*np.sqrt(12); candidate[1] -= .03*np.sqrt(12)
    cert = weight_certificate(A, y, fs, anchor, candidate)
    assert cert['feasible'] and independent_class_certificate(y, cert)['feasible_all_classes']
    assert hard5_LOO_BA(A, y, candidate) < hard5_LOO_BA(A, y, anchor)-1e-12
    target, fit = fit_hardaware_guarded_soft5(A, y, fs, anchor, [candidate], max_iter=4, max_guard_checks=16)
    last_seed = fit['seed_checks'][-1]
    assert last_seed['accepted'] and not last_seed['hard_nonworsening_from_anchor']
    assert fit['selected_seed_id'] != len(fit['seed_checks'])-1
    previous_ba, previous_loss = fit['search_initial_hard_LOO_BA'], fit['search_initial_objective']
    for row in fit['seed_checks']+fit['checks']:
        actual = profile(A, y, row['weights'])['clean_hard5_LOO_BA']
        np.testing.assert_allclose(actual, row['hard_LOO_BA'], atol=1e-12)
    for rid in fit['accepted_step_ids']:
        row = fit['checks'][rid]; assert row['accepted'] and row['update_admissible']
        assert criterion_improves(row['hard_LOO_BA'], row['training_objective'], previous_ba, previous_loss)
        previous_ba, previous_loss = row['hard_LOO_BA'], row['training_objective']
    assert profile(A, y, target)['clean_hard5_LOO_BA'] >= fit['initial_hard_LOO_BA']-1e-12


def test_own_reference_controls_keep_original_provenance_and_actual_accuracy():
    A = np.random.default_rng(135).uniform(0, 1, (24, 12)); y = np.repeat([0, 1], 12)
    A[:, 0] = .1+.8*y; fs = [1, 2]; anchor = np.full(2, np.sqrt(12)/2)
    candidate = np.array([np.sqrt(12), 0.])
    target, fit = fit_hardaware_guarded_soft5(A, y, fs, anchor, [candidate], max_iter=4, max_guard_checks=16)
    assert fit['guard_reference'] == 'own_anchor' and not fit['original_initial_guard']['accepted']
    for row in fit['seed_checks']+fit['checks']:
        cert = weight_certificate(A[:, fs], y, [0, 1], anchor, row['weights'])
        assert row['accepted'] == bool(cert['feasible'] and independent_class_certificate(y, cert)['feasible_all_classes'])
        np.testing.assert_allclose(profile(A[:, fs], y, row['weights'])['clean_hard5_LOO_BA'], row['hard_LOO_BA'], atol=1e-12)
    assert profile(A[:, fs], y, target)['clean_hard5_LOO_BA'] >= fit['initial_hard_LOO_BA']-1e-12
