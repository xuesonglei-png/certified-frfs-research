from collections import Counter
import copy
import numpy as np
import pytest
from frs_global_saturation_transition import build_class_cell, evaluate_class_cell
from independent_global_frs_transition import reconstruct_cell, scalar_cell_evaluation
from frs_saturation_transition_cell import build_class_cell as fixed_cell, evaluate_class_cell as fixed_value
from soft_topk_predictive_metric import anchor_witness_polytope
from audit_forest_compact_transfer import equivalent


def test_attribute_deletion_is_blocked_by_fixed_cell_but_globally_certified():
    A = np.array([[0., 0.]]*6+[[0., 1.]]+[[.2, 1.]]*5)
    y = np.repeat([0, 1], 6); anchor = np.ones(2)/np.sqrt(2); target = np.array([0., np.sqrt(2)])
    C, rhs, fixed = fixed_cell(A, y, [0, 1], anchor, anchor)
    assert fixed['selected_objects'] and len(C)
    assert np.any(C@target < rhs) and not fixed_value(fixed, target)['feasible']
    GC, GR, global_info = build_class_cell(A, y, [0, 1], anchor, anchor)
    assert GC.shape == (0, 2) and len(GR) == 0
    reconstruct_cell(A, y, global_info, Counter())
    value = evaluate_class_cell(global_info, target); equivalent(value, scalar_cell_evaluation(global_info, target))
    assert value['feasible'] and max(value['raw_upper_bounds_with_inflation']) < 0
    majority_C, majority_floor, _ = anchor_witness_polytope(A, y, [0, 1], anchor)
    assert np.all(majority_C@target >= majority_floor)
    bad = copy.deepcopy(global_info); bad['classes'][0]['remaining_signed_anchor_sum'] -= .1
    with pytest.raises(AssertionError):
        reconstruct_cell(A, y, bad, Counter())


def test_global_signed_anchor_plus_transition_bounds_shared_class_losses_on_reduced_support():
    rng = np.random.default_rng(20119); A = rng.uniform(0, 1, (12, 3)); y = np.repeat([0, 1], 6)
    full = np.ones(3)/np.sqrt(3); fs = [0, 2]; anchor = np.repeat(np.sqrt(3)/2, 2)
    _, _, info = build_class_cell(A, y, fs, anchor, full)
    reconstruct_cell(A, y, info, Counter())
    lo, hi = np.maximum(0., A-.01), np.minimum(1., A+.01)
    for _ in range(15):
        weights = rng.dirichlet(np.ones(2))*np.sqrt(3)
        report = evaluate_class_cell(info, weights); equivalent(report, scalar_cell_evaluation(info, weights))
        wide = np.zeros(3); wide[fs] = weights
        for _ in range(6):
            Z = rng.uniform(lo, hi)
            def memberships(w):
                return np.array([min(1., min(float(abs(Z[i]-Z[e])@w) for e in range(len(Z)) if y[e] != y[i])) for i in range(len(Z))])
            difference = memberships(full)-memberships(wide)
            for label, upper in zip(np.unique(y), report['raw_upper_bounds_with_inflation'], strict=True):
                assert difference[y == label].mean() <= upper+1e-12
