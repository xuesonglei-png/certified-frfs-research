"""Supervised target keeps mass/cap and requires separate certification."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from capped_predictive_metric import fit_predictive_target


def test_unique_simplex_cannot_invent_additional_learning_degrees_of_freedom():
    A = np.random.default_rng(4).uniform(0, 1, (12, 4)); y = np.repeat([0, 1], 6)
    target, info = fit_predictive_target(A, y, [0, 1], np.ones(2))
    np.testing.assert_array_equal(target, np.ones(2))
    assert info['status'] == 'unique_capped_simplex' and info['optimizer_iterations'] == 0


def test_actual_training_target_respects_cap_and_improves_soft_objective():
    rng = np.random.default_rng(7); y = np.repeat([0, 1], 12)
    A = rng.uniform(0, 1, (24, 6)); A[:, 0] = np.clip(.1+.8*y+rng.normal(0, .02, 24), 0, 1)
    anchor = np.ones(5)*np.sqrt(6)/5
    target, info = fit_predictive_target(A, y, list(range(5)), anchor)
    assert info['optimizer_success'] and info['status'] == 'local_predictive_target'
    assert info['training_objective_final'] <= info['training_objective_initial']+1e-8
    np.testing.assert_allclose(target.sum(), np.sqrt(6), atol=1e-10)
    assert np.min(target) > 0 and max(target)/np.sqrt(6) <= 2/6+1e-10
    assert target[0] > anchor[0]
    assert 'certificate' not in info


def test_singletons_do_not_create_false_same_class_targets():
    A = np.array([[.1, .2, .3], [.4, .5, .6], [.7, .8, .9]])
    anchor = np.ones(3)/np.sqrt(3)
    target, info = fit_predictive_target(A, np.arange(3), [0, 1, 2], anchor)
    np.testing.assert_array_equal(target, anchor)
    assert info['status'] == 'no_LOO_same_class_supervision'
    assert info['singleton_training_classes'] == [0, 1, 2]


def test_failed_optimizer_is_recorded_and_returns_the_exact_anchor():
    rng = np.random.default_rng(11); A = rng.uniform(0, 1, (20, 6)); y = np.repeat([0, 1], 10)
    anchor = np.ones(5)*np.sqrt(6)/5
    target, info = fit_predictive_target(A, y, list(range(5)), anchor, max_iter=1)
    np.testing.assert_array_equal(target, anchor)
    assert not info['optimizer_success'] and info['status'] == 'optimizer_failed_explicit_anchor_fallback'


def test_invalid_mass_cap_and_too_small_support_rejected():
    A = np.zeros((8, 6)); y = np.repeat([0, 1], 4)
    with pytest.raises(ValueError): fit_predictive_target(A, y, [0, 1], np.ones(2)*np.sqrt(6)/2)
    with pytest.raises(ValueError): fit_predictive_target(A, y, [0, 1, 2, 3], np.array([2, .1, .1, np.sqrt(6)-2.2]))
