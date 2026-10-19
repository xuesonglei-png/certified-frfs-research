import numpy as np
import pytest
from certified_frfs.feasible_query_descent import fit_feasible_query
from certified_frfs.feasible_query_descent import unconstrained_oracle


def test_feasible_descent_mass_monotone_objective_and_disjoint_queries():
    rng = np.random.default_rng(1801); y = np.repeat([0, 1], 12)
    X = rng.uniform(0, .1, (24, 4))+.7*y[:, None]; X[:, 3] = rng.uniform(0, 1, 24)
    fit = fit_feasible_query(X, y, steps=2, oracle_seconds=1.)
    assert fit['certificate']['feasible']
    assert abs(sum(fit['weights'])-2.) < 1e-12
    assert all(b['objective'] < a['objective'] for a, b in zip(fit['trace'], fit['trace'][1:]))
    for split in fit['training_splits']:
        assert not set(split['reference']) & set(split['query'])
        assert set(split['reference']) | set(split['query']) == set(range(24))
    assert fit['certificate']['dependency_certificate']['upper_bound'] <= .1


def test_unconstrained_oracle_same_linear_preference_and_mass():
    full = np.ones(4)/2
    oracle = unconstrained_oracle(full, np.array([2., 1., 1.5, 1.9]), 1.)
    np.testing.assert_allclose(oracle['weights'], [0., 2., 0., 0.], atol=1e-12)


def test_unconstrained_branch_does_not_claim_certificate():
    rng = np.random.default_rng(1901); y = np.repeat([0, 1], 12)
    X = rng.uniform(0, 1, (24, 4))
    fit = fit_feasible_query(X, y, certified=False)
    assert fit['certificate'] is None
    assert all(b['objective'] < a['objective'] for a, b in zip(fit['trace'], fit['trace'][1:]))
    assert abs(sum(fit['weights'])-2.) < 1e-12


def test_capped_oracle_prevents_single_coordinate_mass_and_ratio_one_is_uniform():
    full = np.ones(6)/np.sqrt(6); costs = np.arange(1., 7.)
    oracle = unconstrained_oracle(full, costs, 1., weight_cap_ratio=2.)
    weights = np.array(oracle['weights'])
    assert np.all(weights <= 2*full+1e-10)
    assert (weights > 0).sum() >= 3
    uniform = unconstrained_oracle(full, costs, 1., weight_cap_ratio=1.)
    np.testing.assert_allclose(uniform['weights'], full, atol=1e-12)
    with pytest.raises(ValueError): unconstrained_oracle(full, costs, 1., weight_cap_ratio=.5)


def test_capped_fit_keeps_every_trace_within_fixed_limit():
    rng = np.random.default_rng(2101); y = np.repeat([0, 1], 12); X = rng.uniform(0, 1, (24, 6))
    fit = fit_feasible_query(X, y, certified=False, weight_cap_ratio=2.)
    for node in fit['trace']:
        w = np.array(node['weights']); assert np.all(w <= 2/np.sqrt(6)+1e-10)
        assert 1/np.sum((w/w.sum())**2) >= 3-1e-10
