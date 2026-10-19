from collections import Counter
import copy
import numpy as np
import pytest
from normalized_log_support_penalty import penalty, tangent
from fit_sparse_global_transition_frs import fit_sparse_global_transition_target
from audit_sparse_global_transition_frs import check_sparse_global_transition_fit
from fit_global_transition_capped_frs_hinge import fit_global_transition_capped_target


def test_concave_support_tangent_is_a_global_upper_majorizer_and_sparse_penalty_is_lower():
    rng = np.random.default_rng(771)
    for p in (2, 3, 7):
        for _ in range(10):
            anchor, target = rng.dirichlet(np.ones(p)), rng.dirichlet(np.ones(p))
            g, b = tangent(anchor)
            assert abs(g@anchor+b-penalty(anchor)) < 1e-12
            assert penalty(target) <= g@target+b+1e-12
    assert penalty([0., 1.]) < penalty([.5, .5])


def test_sparsity_incentive_can_keep_a_valid_deletion_that_risk_only_rejects():
    A = np.array([[0., 0.]]*6+[[0., 1.]]+[[.2, 1.]]*5)
    y = np.repeat([0, 1], 6); anchor = np.ones(2)/np.sqrt(2); zero = np.array([0., np.sqrt(2)])
    control, old = fit_global_transition_capped_target(A, y, [0, 1], anchor,
        full_reference=anchor, reference_scope='original_full', initialization=zero,
        initialization_scope='synthetic_feasible_zero')
    assert not old['initialization_accepted']
    target, fit = fit_sparse_global_transition_target(A, y, [0, 1], anchor,
        full_reference=anchor, reference_scope='original_full', initialization=zero,
        initialization_scope='synthetic_feasible_zero', sparsity_lambda=.05)
    assert fit['optimizer_success'] and fit['initialization_accepted']
    assert target[0] == 0 and fit['class_cell_final']['feasible']
    assert fit['training_error_upper_bound_final'] == old['training_error_upper_bound_final'] == 0
    assert fit['training_objective_final'] <= fit['training_objective_initial']+1e-10
    record = {'features':[0, 1], 'anchor':anchor.tolist(), 'target':target.tolist(), 'fit':fit}
    counts = Counter(); check_sparse_global_transition_fit(A, y, record, counts)
    assert counts['independent_master_primal_dual_replays'] >= 1
    changed = copy.deepcopy(record); changed['fit']['steps'][0]['support_gradient'][0] += .1
    with pytest.raises(AssertionError):
        check_sparse_global_transition_fit(A, y, changed, Counter())


def test_lambda_zero_preserves_original_fit_and_unaltered_constraints():
    A = np.random.default_rng(781).uniform(0, 1, (24, 3)); y = np.repeat([0, 1], 12)
    anchor = np.ones(3)/np.sqrt(3)
    old, original = fit_global_transition_capped_target(A, y, [0, 1, 2], anchor,
        full_reference=anchor, reference_scope='original_full')
    new, fit = fit_sparse_global_transition_target(A, y, [0, 1, 2], anchor,
        full_reference=anchor, reference_scope='original_full', sparsity_lambda=0.)
    np.testing.assert_allclose(new, old, atol=1e-10, rtol=0)
    assert fit['class_cell'] == original['class_cell']
    assert fit['polytope'] == original['polytope']
    check_sparse_global_transition_fit(A, y, {'features':[0, 1, 2], 'anchor':anchor.tolist(),
        'target':new.tolist(), 'fit':fit}, Counter())
