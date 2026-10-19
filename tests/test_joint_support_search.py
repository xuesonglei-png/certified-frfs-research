import numpy as np
import pytest
from certified_frfs.joint_support_search import support_pool, fit_joint_support


def test_pool_unique_budget_and_reproducible_nonlocal_moves():
    calls = []
    def evaluate(fs):
        calls.append(fs)
        return {'utility': float(2 in fs and 4 in fs)-.01*len(fs)}
    first = support_pool(8, evaluate, budget=48, seed=21)
    assert len(calls) == len(set(calls)) == len(first) == 48
    assert all(tuple(sorted(set(fs))) == fs for fs in calls)
    assert tuple(range(8)) in calls
    assert any(len(fs) <= 4 for fs in calls)
    assert any(r['origin'] == 'swap' for r in first)
    assert first == support_pool(8, evaluate, budget=48, seed=21)


def test_small_space_terminates_and_invalid_budget_rejected():
    rows = support_pool(1, lambda fs: {'utility': 1.}, budget=64)
    assert len(rows) == 1
    with pytest.raises(ValueError):
        support_pool(3, lambda fs: {'utility': 1.}, budget=0)


def test_shared_pool_feasible_selection_mass_and_fold_isolation():
    rng = np.random.default_rng(92)
    y = np.repeat([0, 1], 12)
    X = rng.uniform(0, .2, (24, 4)) + .7*y[:, None]
    fit = fit_joint_support(X, y, budget=12, certificate_budget=4)
    assert fit['certified']['certificate']['feasible']
    assert fit['unconstrained']['features'] == fit['pool'][0]['features']
    assert fit['evaluations'] <= 12
    assert len(fit['certificates']) <= 5
    for branch in ('certified', 'unconstrained'):
        w = np.array(fit[branch]['weights'])
        assert abs(w.sum()-2.) < 1e-12
        assert np.all(w[np.setdiff1d(np.arange(4), fit[branch]['features'])] == 0)
    for split in fit['search_splits']:
        assert not set(split['train']) & set(split['validation'])
        assert set(split['train']) | set(split['validation']) == set(range(24))
