"""Replay accepted transitions and refuse uncertified starting points."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from guarded_soft5_descent import fit_guarded_soft5
from independent_soft_topk_objective import independent_objective
from certified_frfs.feasible_transport import weight_certificate
from audit_classwise_repair_admission import independent_class_certificate


def table():
    rng = np.random.default_rng(135); y = np.repeat([0, 1], 12)
    A = rng.uniform(0, 1, (24, 12)); A[:, 0] = .1+.8*y
    return A, y


def test_each_accepted_update_improves_independent_loss_and_preserves_original_guards():
    A, y = table(); anchor = np.ones(12)/np.sqrt(12); fs = list(range(12))
    target, info = fit_guarded_soft5(A, y, fs, anchor, max_iter=6)
    assert len(info['accepted_step_ids']) > 0
    previous = independent_objective(A, y, anchor/np.sqrt(12))
    for rid in info['accepted_step_ids']:
        row = info['checks'][rid]; w = np.array(row['weights'])
        np.testing.assert_allclose(w.sum(), anchor.sum(), atol=1e-10)
        cert = weight_certificate(A, y, fs, anchor, w)
        assert cert['feasible'] and independent_class_certificate(y, cert)['feasible_all_classes']
        current = independent_objective(A, y, w/np.sqrt(12))
        np.testing.assert_allclose(current, row['training_objective'], atol=1e-9)
        assert current < previous-1e-8; previous = current
    np.testing.assert_allclose(independent_objective(A, y, target/np.sqrt(12)), info['training_objective_final'], atol=1e-9)
    assert info['training_objective_final'] < info['training_objective_initial']


def test_budget_and_uncertified_anchor_are_explicit():
    A, y = table(); anchor = np.ones(12)/np.sqrt(12)
    target, info = fit_guarded_soft5(A, y, list(range(12)), anchor, max_guard_checks=1)
    assert len(info['checks']) <= 1 and info['endpoint_guard']['accepted']
    cert = weight_certificate(A, y, list(range(12)), anchor, target)
    assert cert['feasible'] and independent_class_certificate(y, cert)['feasible_all_classes']
    with pytest.raises(ValueError, match='original-certified'):
        fit_guarded_soft5(A, y, [1], np.array([np.sqrt(12)]))
    with pytest.raises(ValueError): fit_guarded_soft5(A, y, list(range(12)), anchor, max_guard_checks=0)
