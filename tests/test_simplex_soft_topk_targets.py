"""Model-family extension never replaces the original certificate contracts."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from simplex_soft_topk_targets import fit_simplex_target
from independent_soft_topk_objective import independent_objective
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.classwise_dependency_certificate import classwise_dependency_certificate


def test_one_attribute_can_satisfy_original_FRS_and_majority_contracts_at_fixed_mass():
    # The old cap2/d restriction excludes this representation; the original
    # loss/margin certificates themselves are valid and nonvacuous on it.
    rng = np.random.default_rng(135); y = np.repeat([0, 1], 12); d = 12
    A = rng.uniform(0, 1, (24, d)); A[:, 0] = .1+.8*y
    full = np.ones(d)/np.sqrt(d); w = np.array([np.sqrt(d)])
    cert = weight_certificate(A, y, [0], full, w)
    classes = classwise_dependency_certificate(y, cert)
    assert cert['feasible'] and classes['feasible_all_classes']
    assert cert['dependency_certificate']['upper_bound'] < 1e-8
    target, fit = fit_simplex_target(A, y, [0], w)
    np.testing.assert_array_equal(target, w)
    assert fit['status'] == 'unique_simplex' and fit['optimizer_iterations'] == 0


def test_same_mass_larger_family_can_learn_beyond_old_cap_without_changing_guard():
    rng = np.random.default_rng(135); y = np.repeat([0, 1], 12); d = 12
    A = rng.uniform(0, 1, (24, d)); A[:, 0] = .1+.8*y
    anchor = np.ones(d)/np.sqrt(d)
    target, fit = fit_simplex_target(A, y, list(range(d)), anchor)
    assert fit['optimizer_success'] and fit['status'] == 'local_simplex_soft5_target'
    assert fit['linear_constraints_verified'] and target[0]/np.sqrt(d) > 2/d
    np.testing.assert_allclose(target.sum(), anchor.sum(), atol=1e-10)
    final = independent_objective(A, y, target/np.sqrt(d))
    np.testing.assert_allclose(final, fit['training_objective_final'], atol=1e-9)
    assert final < independent_objective(A, y, anchor/np.sqrt(d))
    cert = weight_certificate(A, y, list(range(d)), anchor, target)
    assert cert['feasible'] and classwise_dependency_certificate(y, cert)['feasible_all_classes']


def test_failure_stays_visible_and_bad_mass_is_rejected():
    rng = np.random.default_rng(136); A = rng.uniform(0, 1, (18, 8)); y = np.repeat([0, 1], 9)
    anchor = np.ones(8)/np.sqrt(8)
    target, fit = fit_simplex_target(A, y, list(range(8)), anchor, constrained=False, max_iter=1)
    np.testing.assert_array_equal(target, anchor); assert not fit['optimizer_success']
    with pytest.raises(ValueError): fit_simplex_target(A, y, [0, 1], np.ones(2))
