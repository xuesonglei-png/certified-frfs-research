"""Controls receive a feasible own-reference search, without original claims."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from adaptive_guarded_soft5_descent import fit_adaptive_guarded_soft5
from independent_soft_topk_objective import independent_objective
from certified_frfs.feasible_transport import weight_certificate
from audit_classwise_repair_admission import independent_class_certificate


def test_adaptive_original_and_own_references_have_equal_budgets_and_fresh_checks():
    rng = np.random.default_rng(135); y = np.repeat([0, 1], 12); d = 12
    A = rng.uniform(0, 1, (24, d)); A[:, 0] = .1+.8*y
    full = np.ones(d)/np.sqrt(d)
    cases = [(list(range(d)), full, 'original_full'), ([1, 2], np.full(2, np.sqrt(d)/2), 'own_anchor')]
    for fs, anchor, expected in cases:
        target, fit = fit_adaptive_guarded_soft5(A, y, fs, anchor, max_iter=4, max_guard_checks=16)
        assert fit['guard_reference'] == expected
        assert fit['max_iter'] == 4 and fit['max_guard_checks'] == 16 and len(fit['checks']) <= 16
        if expected == 'own_anchor': assert not fit['original_initial_guard']['accepted']
        initial = independent_objective(A[:, fs], y, anchor/np.sqrt(d))
        previous = initial
        for rid in fit['accepted_step_ids']:
            row = fit['checks'][rid]; w = np.array(row['weights'])
            cert = (weight_certificate(A, y, fs, full, w) if expected == 'original_full'
                    else weight_certificate(A[:, fs], y, list(range(len(fs))), anchor, w))
            assert cert['feasible'] and independent_class_certificate(y, cert)['feasible_all_classes']
            value = independent_objective(A[:, fs], y, w/np.sqrt(d))
            assert value < previous-1e-8; previous = value
        np.testing.assert_allclose(independent_objective(A[:, fs], y, target/np.sqrt(d)), fit['training_objective_final'], atol=1e-9)
        assert fit['endpoint_guard']['accepted'] and fit['training_objective_final'] <= initial+1e-8
