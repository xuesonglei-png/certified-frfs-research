from collections import Counter
import copy
import numpy as np
import pytest
from fit_frs_cell_hinge import fit_frs_cell_target
from audit_frs_cell_fit import check_cell_fit
from frs_class_cell import build_class_cell, evaluate_class_cell
from frs_cell_metric_repair import repair_cell_metric
from independent_frs_class_cell import reconstruct_cell


def test_frs_constrained_fit_replays_primal_dual_and_actual_descent():
    rng = np.random.default_rng(781); A = rng.uniform(0, 1, (18, 3)); y = np.repeat([0, 1], 9)
    anchor = np.ones(3)/np.sqrt(3)
    target, fit = fit_frs_cell_target(A, y, [0, 1, 2], anchor, full_reference=anchor, reference_scope='original_full')
    assert fit['optimizer_success'], fit['status']
    assert fit['class_cell_final']['feasible'] and fit['training_objective_final'] <= fit['training_objective_initial']
    counts = Counter()
    check_cell_fit(A, y, {'features': [0, 1, 2], 'anchor': anchor.tolist(), 'target': target.tolist(), 'fit': fit}, counts)
    assert counts['independent_master_primal_dual_replays'] >= 1


def test_star_free_epigraphs_replay_and_forged_bounds_are_rejected():
    A = np.r_[np.full((9, 3), .05), np.full((1, 3), .35), np.full((8, 3), .65)]
    y = np.repeat([0, 1], 9); anchor = np.ones(3)/np.sqrt(3)
    target, fit = fit_frs_cell_target(A, y, [0, 1, 2], anchor, full_reference=anchor, reference_scope='original_full')
    assert any(r['mode'] == 'shared_star' for r in fit['class_cell']['classes'])
    record = {'features': [0, 1, 2], 'anchor': anchor.tolist(), 'target': target.tolist(), 'fit': fit}
    check_cell_fit(A, y, record, Counter())
    altered = copy.deepcopy(fit['class_cell']); altered['classes'][0]['H'][0] += .1
    with pytest.raises(AssertionError):
        reconstruct_cell(A, y, altered, Counter())


def test_convex_class_repair_handles_large_violation_without_relaxing_eta():
    rng = np.random.default_rng(181); A = rng.uniform(0, 1, (18, 3)); y = np.repeat([0, 1], 9)
    anchor = np.ones(3)/np.sqrt(3); C, rhs, info = build_class_cell(A, y, [0, 1, 2], anchor, anchor)
    candidate = np.array([np.sqrt(3), 0., 0.])
    assert not evaluate_class_cell(info, candidate)['feasible']
    fixed, trace, blend = repair_cell_metric(C, rhs, info, candidate, anchor)
    assert evaluate_class_cell(info, fixed)['feasible'] and trace and 0 < blend <= 1
    np.testing.assert_allclose(fixed.sum(), anchor.sum(), atol=1e-14)
    assert info['eta'] == .1 and info['eps'] == .01


def test_infeasible_anchor_is_explicitly_rejected():
    A = np.r_[np.tile([.1, 0.], (9, 1)), np.tile([.8, .02], (9, 1))]
    y = np.repeat([0, 1], 9); anchor = np.array([np.sqrt(2)])
    with pytest.raises(ValueError, match='strictly feasible'):
        fit_frs_cell_target(A, y, [1], anchor, full_reference=np.ones(2)/np.sqrt(2), reference_scope='original_full')
