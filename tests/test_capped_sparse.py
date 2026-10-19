import itertools
import numpy as np
import pytest
from scipy.optimize import minimize
from certified_frfs.capped_sparse import capped_sparse_projection, optimize_capped_sparse


@pytest.mark.parametrize('seed', [1, 7, 23])
def test_projection_matches_exhaustive_convex_supports(seed):
    z = np.random.default_rng(seed).normal(size=5); k = 3; cap = .4
    actual = capped_sparse_projection(z, k, cap)
    best = np.inf
    for ids in itertools.combinations(range(5), k):
        ids = np.array(ids)
        fit = minimize(lambda x: np.sum((x-z[ids])**2), np.ones(k)/k,
            jac=lambda x: 2*(x-z[ids]), bounds=[(0, cap)]*k,
            constraints={'type': 'eq', 'fun': lambda x: x.sum()-1, 'jac': lambda x: np.ones(k)},
            method='SLSQP', options={'ftol': 1e-12})
        assert fit.success
        candidate = np.zeros(5); candidate[ids] = fit.x
        best = min(best, np.sum((candidate-z)**2))
    assert abs(np.sum((actual-z)**2)-best) < 1e-9
    assert abs(actual.sum()-1) < 1e-12
    assert np.count_nonzero(actual) <= k
    assert actual.max() <= cap


def test_infeasible_cap_rejected_and_saturation():
    with pytest.raises(ValueError):
        capped_sparse_projection(np.arange(5), 2, .4)
    w = capped_sparse_projection(np.arange(6), 3, 1/3)
    assert np.allclose(w, [0, 0, 0, 1/3, 1/3, 1/3])


def test_sparse_descent_checks_real_objective():
    target = np.array([.4, .3, .2, .1, 0])
    def objective(w):
        return float(np.sum((w-target)**2)), 2*(w-target)
    w, fit = optimize_capped_sparse(objective, np.ones(5)/5, 3, cap=.4)
    values = [r['objective'] for r in fit['trace']]
    assert len(values) > 1
    assert np.all(np.diff(values) <= 1e-12)
    assert np.count_nonzero(w) <= 3 and w.max() <= .4
