import numpy as np
from certified_frfs.core import gamma
from certified_frfs.joint_milp import pairwise_loss_envelope, joint_certified_metric


def test_signed_pair_envelope_covers_dependency_losses():
    rng = np.random.default_rng(1601); X = rng.uniform(0, 1, (12, 4)); y = np.repeat([0, 1], 6)
    w = np.ones(4)/2; v = np.array([.9, 0., .3, .8])
    bound = pairwise_loss_envelope(X, y, w, v)['upper_bound']
    for _ in range(50):
        Z = np.clip(X+rng.uniform(-.01, .01, X.shape), 0, 1)
        assert gamma(Z, y, weights=w)-gamma(Z, y, weights=v) <= bound+1e-12
    assert pairwise_loss_envelope(X, y, w, w)['upper_bound'] == 0.


def test_joint_solver_can_remove_duplicate_attributes_and_verify_mass():
    y = np.repeat([0, 1], 8)
    column = .1+.7*y
    X = np.repeat(column[:, None], 4, axis=1)
    result = joint_certified_metric(X, y, time_limit=5.)
    assert result['certificate']['feasible']
    assert len(result['features']) < 4
    assert abs(sum(result['weights'])-2.) < 1e-12
    assert result['certificate']['dependency_certificate']['upper_bound'] <= .1


def test_equal_cost_relaxation_has_constant_one_lower_bound():
    y = np.repeat([0, 1], 8); X = np.repeat((.1+.7*y)[:, None], 4, axis=1)
    fit = joint_certified_metric(X, y, continuous=True, time_limit=5.)
    assert abs(fit['fit']['solver_objective']-1.) < 1e-8
    assert fit['fit']['continuous']


def test_weighted_continuous_model_can_select_lowest_cost_duplicate():
    y = np.repeat([0, 1], 8); X = np.repeat((.1+.7*y)[:, None], 4, axis=1)
    fit = joint_certified_metric(X, y, continuous=True, support_costs=[1., 2., 3., 4.], time_limit=5.)
    assert fit['features'] == [0]
    assert fit['certificate']['feasible']


def test_solver_side_slack_accepts_verified_boundary_near_metric():
    y = np.repeat([0, 1], 8); X = np.repeat((.1+.7*y)[:, None], 4, axis=1)
    fit = joint_certified_metric(X, y, eps=.02, continuous=True,
        support_costs=[1., 2., 3., 4.], loss_slack=1e-8, time_limit=5.)
    assert not fit['full_fallback']
    assert fit['fit']['incumbent_linear_verified']
    assert fit['fit']['incumbent_loss_bound'] <= .1-1e-10


def test_certified_joint_milp_accepts_fixed_weight_cap():
    y = np.repeat([0, 1], 8); X = np.repeat((.1+.7*y)[:, None], 6, axis=1)
    fit = joint_certified_metric(X, y, continuous=True, support_costs=np.arange(1., 7.),
        support_cap_ratio=2., loss_slack=1e-8, time_limit=5.)
    assert fit['certificate']['feasible']
    assert np.all(np.array(fit['weights']) <= 2/np.sqrt(6)+1e-10)
