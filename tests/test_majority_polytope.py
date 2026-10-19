import numpy as np
import pytest
from scipy.spatial.distance import cdist
from certified_frfs.majority_polytope import fit_majority_polytope


def test_full_metric_remains_optimal_and_targets_retained():
    rng = np.random.default_rng(1501); y = np.repeat([0, 1], 12)
    X = .7*y[:, None]+rng.uniform(0, .2, (24, 4)); w = np.ones(4)/2
    v, info = fit_majority_polytope(X, y, range(4), w)
    np.testing.assert_allclose(v, w, atol=1e-12, rtol=0)
    assert info['linear_constraints_verified']
    assert info['certificate']['feasible']
    assert info['required'] == [11, 11]
    assert info['target_rule'] == 'full'


def test_reduced_metric_has_mass_and_sampled_target_majorities():
    rng = np.random.default_rng(1502); y = np.repeat([0, 1, 2], 8)
    X = .35*y[:, None]+rng.uniform(0, .05, (24, 4)); w = np.ones(4)/2
    fs = [0, 2]; v, info = fit_majority_polytope(X, y, fs, w)
    assert v is not None and info['linear_constraints_verified']
    assert abs(v.sum()-w.sum()) < 1e-12
    for _ in range(30):
        Z = np.clip(X+rng.uniform(-.01, .01, X.shape), 0, 1)
        D = cdist(Z[:, fs]*v, Z[:, fs]*v, 'cityblock'); np.fill_diagonal(D, np.inf)
        nn = np.argsort(D, axis=1, kind='stable')[:, :5]
        correct = (y[nn] == y[:, None]).sum(axis=1) >= 3
        assert np.all(correct[info['targets']])


def test_impossible_constant_support_returns_no_feasible_metric():
    y = np.repeat([0, 1], 8); X = np.column_stack([.8*y, np.ones(16)*.5])
    v, info = fit_majority_polytope(X, y, [1], np.ones(2)/np.sqrt(2))
    assert v is None
    assert info['status'] == 'no_solver_feasible_point'


def test_target_rule_is_explicit_and_invalid_rule_rejected():
    y = np.repeat([0, 1], 8); X = np.column_stack([.8*y, np.ones(16)*.5])
    with pytest.raises(ValueError):
        fit_majority_polytope(X, y, [0], np.ones(2)/np.sqrt(2), target_rule='guess')
