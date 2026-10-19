"""Shared-anchor, witness-mixture and full-certified-object margin checks."""
from types import SimpleNamespace

import numpy as np
import pytest

import certified_frfs.margin as margin
from certified_frfs.margin_verification import exact_negative_loo_margin_milp


def _table_from_box(lower, upper):
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    return (lower+upper)/2, (upper-lower)/2


def _assert_bound_vs_milp(lower, upper, y, target, friends, enemy, weights, expected):
    X, eps = _table_from_box(lower, upper)
    bound = margin.mixture_enemy_bound(lower, upper, target, enemy, friends,
                                      list(range(X.shape[1])), weights)
    exact = exact_negative_loo_margin_milp(X, y, target, weights=weights, eps=eps,
                                          clip=False)
    assert exact["optimal"], exact["message"]
    assert -exact["lower_bound"] == pytest.approx(expected, abs=2e-6)
    assert bound["lower_bound"] <= -exact["upper_bound"] + 2e-6
    return bound, exact


def test_shared_lambda_strictly_improves_onehot_and_changes_robust_decision():
    lower = [[0, 0], [0, 0], [1, 0], [0, 0.75]]
    upper = [[1, 0], [0, 0], [1, 0], [1, 0.75]]
    bound, _ = _assert_bound_vs_milp(lower, upper, [0, 0, 0, 1], 0, [1, 2], 3, [1, 1], 0.25)
    assert bound["onehot_bound"] == pytest.approx(-0.25)
    assert bound["raw_feasible_mixture_value"] == pytest.approx(0.25)
    assert bound["lower_bound"] > 0
    assert bound["mixture"] == pytest.approx([0.5, 0.5])
    assert bound["midpoint_separation"]


def test_same_mixture_must_be_shared_across_attributes():
    lower = upper = [[0, 0], [0, 1], [1, 0], [0.4, 0.4]]
    bound, _ = _assert_bound_vs_milp(lower, upper, [0, 0, 0, 1], 0, [1, 2], 3,
                                    [1, 1], -0.2)
    assert bound["lower_bound"] == pytest.approx(-0.2, abs=1e-9)
    # Independently selecting a different friend per attribute would give
    # +.8 instead of -.2; a nearest reference must be one actual object.
    assert bound["raw_feasible_mixture_value"] < 0


def test_midpoint_crossing_can_leave_a_strict_minimax_gap():
    lower = [[0, 0], [0, 0], [0.5, 0], [0, 0.6]]
    upper = [[1, 0], [0.5, 0], [1, 0], [1, 0.6]]
    bound, _ = _assert_bound_vs_milp(lower, upper, [0, 0, 0, 1], 0, [1, 2], 3, [1, 1], 0.1)
    assert not bound["midpoint_separation"]
    assert bound["raw_feasible_mixture_value"] == pytest.approx(-0.15)
    assert bound["mixture"] == pytest.approx([0.5, 0.5])


def test_triplet_broadcast_matches_every_scalar_piecewise_minimum():
    rng = np.random.default_rng(10400)
    lower = rng.uniform(-1, 1, (5, 2))
    upper = lower+rng.uniform(0, 1, (5, 2))
    friends = np.array([[1, 2, 4], [0, 2, 4], [0, 1, 4], [0, 1, 2], [0, 1, 2]])
    for k in range(2):
        actual = margin.triplet_interval_min(
            lower[:, k, None, None], upper[:, k, None, None],
            lower[None, :, k, None], upper[None, :, k, None],
            lower[friends, k][:, None, :], upper[friends, k][:, None, :])
        assert actual.shape == (5, 5, 3)
        for i in range(5):
            for j in range(5):
                for t, h in enumerate(friends[i]):
                    points = np.clip([lower[i,k], upper[i,k], lower[j,k], upper[j,k],
                                      (lower[h,k]+upper[h,k])/2], lower[i,k], upper[i,k])
                    expected = min(max(lower[j,k]-a, a-upper[j,k], 0)
                                   -max(abs(a-lower[h,k]), abs(a-upper[h,k])) for a in points)
                    assert actual[i,j,t] == pytest.approx(expected)


def test_workspace_excludes_self_and_never_certifies_singleton_classes():
    X = np.array([[0], [0.2], [0.1], [1]])
    y = np.array([0, 0, 1, 1])
    workspace = margin.MarginWorkspace(X, y, weights=[1], eps=0, n_friends=3)
    assert all(i not in workspace.friends[i] for i in range(4))
    bounds = workspace.bounds([0])
    assert bounds[0] == pytest.approx(-0.1)
    assert bounds[0] < 0  # Including self would incorrectly make this positive.
    singleton = margin.MarginWorkspace([[0], [0.2], [1]], [0, 0, 1], weights=[1], eps=0)
    assert not singleton.applicable[2]
    assert np.isneginf(singleton.bounds([0])[2])
    with pytest.raises(ValueError, match="protected objects"):
        singleton.select([0], protected=[False, False, True])


def test_selector_guarantee_is_restricted_to_full_certified_objects():
    X = np.array([[0,0], [0.1,0.1], [0.2,0.2], [0.9,0.9], [1,1], [0.8,0.8], [0.5,0.5]])
    y = np.array([0,0,1,1,1,0,2])
    workspace = margin.MarginWorkspace(X, y, weights=[1,1], eps=0.005, n_friends=2)
    full = workspace.bounds([0,1])
    selected = workspace.select([0,1])
    expected = np.flatnonzero(workspace.applicable & (full > 1e-10)).tolist()
    assert selected["protected_indices"] == expected
    assert len(expected) > 0 and len(expected) < len(X)
    assert 6 not in expected
    assert not selected["minimality_proven"]
    for i in expected:
        exact = exact_negative_loo_margin_milp(X,y,i,features=selected["features"],
                                               weights=[1,1],eps=0.005)
        assert exact["upper_bound"] < 0
        assert selected["final_margin_lower"][i] <= -exact["upper_bound"]+2e-6
    assert sum(row["protected"] for row in selected["coverage"]) == len(expected)


def test_no_protected_object_does_not_trigger_vacuous_compression():
    workspace = margin.MarginWorkspace([[0,0],[0.2,0.2],[0.1,0.1],[0.3,0.3]],
                                       [0,0,1,1], weights=[1,1], eps=0.01)
    selected = workspace.select([0,1])
    assert selected["protected_indices"] == []
    assert selected["features"] == [0,1]
    assert selected["history"] == []


def test_timeout_mixture_projects_and_recomputes_a_feasible_simplex_witness(monkeypatch):
    lower = [[0,0],[0,0],[1,0],[0,.75]]
    upper = [[1,0],[0,0],[1,0],[1,.75]]
    monkeypatch.setattr(margin, "linprog", lambda *args, **kwargs: SimpleNamespace(
        x=np.array([2.,2.,1e6,1e6]), fun=-1e12, success=False, status=1))
    bound = margin.mixture_enemy_bound(lower,upper,0,3,[1,2],[0,1],[1,1])
    assert bound["mixture"] == pytest.approx([0.5,0.5])
    assert bound["raw_feasible_mixture_value"] == pytest.approx(.25)
    assert bound["lower_bound"] < .25
    assert not bound["solver_success"]


def test_missing_mixture_incumbent_keeps_a_safe_onehot_witness(monkeypatch):
    monkeypatch.setattr(margin,"linprog",lambda *args,**kwargs: SimpleNamespace(
        x=None, fun=-1e12, success=False, status=1))
    bound=margin.mixture_enemy_bound([[0,0],[0,0],[1,0],[0,.75]],
        [[1,0],[0,0],[1,0],[1,.75]],0,3,[1,2],[0,1],[1,1])
    assert bound["raw_feasible_mixture_value"] == pytest.approx(-.25)
    assert sum(bound["mixture"]) == pytest.approx(1)
    assert all(value >= 0 for value in bound["mixture"])


def test_finite_huge_solver_coordinates_cannot_overflow_simplex_normalization(monkeypatch):
    monkeypatch.setattr(margin,"linprog",lambda *args,**kwargs: SimpleNamespace(
        x=np.array([1e308,1e308,0.,0.]), fun=-1e308, success=False, status=1))
    bound=margin.mixture_enemy_bound([[0,0],[0,0],[1,0],[0,.75]],
        [[1,0],[0,0],[1,0],[1,.75]],0,3,[1,2],[0,1],[1,1])
    assert sum(bound["mixture"]) == pytest.approx(1)
    assert all(np.isfinite(value) and value >= 0 for value in bound["mixture"])
    assert bound["lower_bound"] <= .25+1e-9


@pytest.mark.parametrize("seed", range(12))
def test_random_workspace_and_all_friend_mixtures_against_exact_object_margins(seed):
    rng=np.random.default_rng(10600+seed)
    X=rng.uniform(.1,.9,(5,2))
    y=np.array([0,0,0,1,1])
    eps=rng.uniform(.01,.2,(5,2))
    weights=rng.uniform(.1,1.5,2)
    workspace=margin.MarginWorkspace(X,y,weights=weights,eps=eps,n_friends=3)
    features=[0] if seed%2 else [0,1]
    triplet_bounds=workspace.bounds(features)
    for i in range(len(X)):
        exact=exact_negative_loo_margin_milp(X,y,i,features=features,weights=weights,eps=eps)
        assert exact["optimal"],exact["message"]
        true_lower=-exact["upper_bound"]
        assert triplet_bounds[i] <= true_lower+2e-6
        friends=np.flatnonzero((y==y[i]) & (np.arange(len(y))!=i))
        enemy_bounds=[]
        for j in np.flatnonzero(y!=y[i]):
            bound=margin.mixture_enemy_bound(workspace.lower,workspace.upper,i,j,friends,features,weights)
            assert np.isfinite(bound["lower_bound"])
            assert min(bound["mixture"]) >= 0
            assert sum(bound["mixture"]) == pytest.approx(1)
            assert bound["raw_feasible_mixture_value"]+1e-10 >= bound["onehot_bound"]
            enemy_bounds.append(bound["lower_bound"])
        assert min(enemy_bounds) <= true_lower+2e-6
        # With every friend included, mixture is at least as strong as the
        # fixed-friend triplet bound, apart from outward floating safeguards.
        assert min(enemy_bounds)+1e-8 >= triplet_bounds[i]
