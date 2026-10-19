import numpy as np
import pytest
from certified_frfs.guarded_support import discovery_guard_split, paired_guard_gain, fit_guarded_support


def test_classwise_split_is_disjoint_and_small_classes_stay_in_reference():
    y = np.repeat([0, 1, 2], [2, 4, 20])
    d, g = discovery_guard_split(y)
    assert not set(d) & set(g)
    assert set(d) | set(g) == set(range(len(y)))
    assert np.all(y[g] != 0)
    assert np.sum(y[d] == 1) == 3
    assert len(g) == 6


def test_paired_guard_penalty_is_balanced_and_singleton_conservative():
    result = paired_guard_gain(np.array([1., 1., .5, .5]), np.zeros(4), np.array([0, 0, 1, 1]))
    assert result['mean_gain'] == .75
    assert result['penalized_gain'] == .75
    one = paired_guard_gain(np.ones(1), np.zeros(1), np.zeros(1))
    assert one['penalized_gain'] == 0.
    with pytest.raises(ValueError):
        paired_guard_gain([], [], [])


def test_candidate_discovery_does_not_use_guard_features():
    rng = np.random.default_rng(1301)
    y = np.repeat([0, 1], 16)
    X = rng.uniform(0, .1, (32, 3))+.8*y[:, None]
    fit = fit_guarded_support(X, y, budget=8, certificate_budget=4)
    altered = X.copy(); altered[fit['guard_indices']] = rng.uniform(0, 1, (len(fit['guard_indices']), 3))
    second = fit_guarded_support(altered, y, budget=8, certificate_budget=4)
    assert fit['pool'] == second['pool']
    assert fit['certified']['certificate']['feasible']
    assert np.sum(fit['certified']['weights']) == pytest.approx(np.sqrt(3))
