"""Single-object joint-box LOO objectives and numerical-verifier semantics."""
from types import SimpleNamespace

import numpy as np
import pytest

from certified_frfs.core import interval_bounds
from certified_frfs.margin_verification import exact_negative_loo_margin_milp
from certified_frfs.verification import _MILP


def _negative_margin(X, y, target, weights):
    X, y = np.asarray(X), np.asarray(y)
    distances = abs(X-X[target]) @ weights
    same = (y == y[target]) & (np.arange(len(y)) != target)
    enemy = y != y[target]
    return float(distances[same].min()-distances[enemy].min())


@pytest.mark.parametrize("normalize,expected", [(False, -2.5), (True, -0.5)])
def test_two_classes_two_objects_known_uncapped_worst_margin(normalize, expected):
    result = exact_negative_loo_margin_milp([[0], [0.2], [0.8], [1]], [0, 0, 1, 1],
                                            0, weights=[5], eps=0.05, normalize=normalize)
    assert result["optimal"], result["message"]
    assert result["lower_bound"] == pytest.approx(expected)
    assert result["upper_bound"] == pytest.approx(expected, abs=6e-8)
    assert result["upper_bound"] < 0
    assert result["nearest_same_at_witness"] == 1
    assert result["nearest_enemy_at_witness"] == 2


def test_maximum_can_be_strictly_inside_target_box():
    X = np.array([[0.2], [0], [1], [0.5]])
    eps = [[0.8], [0], [0], [0]]
    y = [0, 0, 0, 1]
    result = exact_negative_loo_margin_milp(X, y, 0, weights=[1], eps=eps)
    assert result["optimal"]
    assert result["lower_bound"] == pytest.approx(0.5)
    assert result["upper_bound"] == pytest.approx(0.5, abs=1.1e-8)
    assert result["worst_case"][0, 0] == pytest.approx(0.5)
    for endpoint in (0, 1):
        z = X.copy()
        z[0, 0] = endpoint
        assert _negative_margin(z, y, 0, [1]) == pytest.approx(-0.5)


def test_shared_target_cannot_independently_maximize_same_and_minimize_enemy():
    X = [[0.5], [0], [1], [0.9]]
    y = [0, 0, 1, 1]
    result = exact_negative_loo_margin_milp(X, y, 0, weights=[1], eps=[[0.2], [0], [0], [0]])
    assert result["optimal"]
    assert result["lower_bound"] == pytest.approx(0.5)
    assert result["worst_case"][0, 0] == pytest.approx(0.7)


@pytest.mark.parametrize("y,target", [([0, 1, 1], 0), ([0, 0, 0], 1), ([0], 0)])
def test_inapplicable_neighborhoods_are_explicit(y, target):
    result = exact_negative_loo_margin_milp(np.zeros((len(y), 1)), y, target)
    assert not result["applicable"]
    assert result["status"] == "not_applicable"
    assert result["lower_bound"] is None and result["upper_bound"] is None
    assert not result["optimal"]


@pytest.mark.parametrize("target", [-1, 4, 0.0, [0], True])
def test_target_must_be_a_valid_integer_index(target):
    with pytest.raises(ValueError, match="target"):
        exact_negative_loo_margin_milp([[0], [0.2], [0.8], [1]], [0, 0, 1, 1], target)


def test_zero_weight_and_singleton_box_allow_ties_without_claiming_strict_correctness():
    result = exact_negative_loo_margin_milp([[0], [0.2], [0.8], [1]], [0, 0, 1, 1],
                                            0, weights=[0])
    assert result["optimal"]
    assert result["lower_bound"] == result["upper_bound"] == 0
    assert "certified" not in result
    with pytest.raises(ValueError, match="positive selected weight"):
        exact_negative_loo_margin_milp([[0], [0.2], [0.8], [1]], [0, 0, 1, 1],
                                        0, weights=[0], normalize=True)
    singleton = exact_negative_loo_margin_milp([[0], [0.2], [0.8], [1]], [0, 0, 1, 1],
                                               0, weights=[1], eps=0)
    assert singleton["upper_bound"] == pytest.approx(-0.6)


def test_separate_target_and_reference_coordinate_boxes_and_feature_mask():
    X = [[0, 100], [2, 0], [5, 0], [6, 0]]
    y = [0, 0, 1, 1]
    eps = [[0, 0], [0.5, 0], [0.1, 0], [0.2, 0]]
    result = exact_negative_loo_margin_milp(X, y, 0, features=[0], weights=[2, 3],
                                            eps=eps, clip=False)
    assert result["optimal"]
    assert result["lower_bound"] == pytest.approx(2*(2.5-4.9))


def test_normalization_handles_tiny_weights_without_reciprocal_overflow():
    result = exact_negative_loo_margin_milp([[0], [0.2], [0.8], [1]], [0, 0, 1, 1],
                                            0, weights=[1e-310], normalize=True, eps=0.05)
    assert result["optimal"]
    assert result["upper_bound"] == pytest.approx(-0.5, abs=1.1e-8)


def test_timeout_without_primal_or_dual_uses_attained_nominal_and_safe_interval(monkeypatch):
    monkeypatch.setattr(_MILP, "solve", lambda *args: SimpleNamespace(
        x=None, status=1, message="Time limit reached"))
    result = exact_negative_loo_margin_milp([[0], [0.2], [0.8], [1]], [0, 0, 1, 1],
                                            0, weights=[1], eps=0.05)
    assert result["lower_bound"] == pytest.approx(-0.6)
    assert result["upper_bound"] == pytest.approx(-0.45)
    assert not result["optimal"]
    assert not result["solver_has_dual_bound"]
    assert not result["solver_has_incumbent"]


def test_timeout_with_finite_dual_tightens_bound_without_claiming_optimality(monkeypatch):
    monkeypatch.setattr(_MILP, "solve", lambda *args: SimpleNamespace(
        x=None, mip_dual_bound=0.5, status=1, message="Time limit reached"))
    result = exact_negative_loo_margin_milp([[0], [0.2], [0.8], [1]], [0, 0, 1, 1],
                                            0, weights=[1], eps=0.05)
    assert result["lower_bound"] == pytest.approx(-0.6)
    assert result["upper_bound"] == pytest.approx(-0.5, abs=1.1e-8)
    assert result["used_solver_dual_bound"]
    assert not result["optimal"]


def test_inconsistent_solver_objective_and_dual_are_not_trusted(monkeypatch):
    def fake(self, *args):
        x = np.zeros(len(self.lower))
        x[:4] = [100, 100, -100, -100]
        return SimpleNamespace(x=x, fun=-1000, mip_dual_bound=1000,
                               status=1, message="Invalid numerical dual")
    monkeypatch.setattr(_MILP, "solve", fake)
    X, y = np.array([[0], [0.2], [0.8], [1]]), [0, 0, 1, 1]
    result = exact_negative_loo_margin_milp(X, y, 0, weights=[1], eps=0.05)
    lo, hi = interval_bounds(X, 0.05)
    assert np.all(result["worst_case"] >= lo) and np.all(result["worst_case"] <= hi)
    assert result["lower_bound"] == pytest.approx(_negative_margin(result["worst_case"], y, 0, [1]))
    assert result["numerical_bound_conflict"]
    assert not result["used_solver_dual_bound"]
    assert not result["optimal"]


@pytest.mark.parametrize("seed", range(8))
def test_random_multiclass_small_tables_against_internal_attacks(seed):
    rng = np.random.default_rng(9300+seed)
    X, y = rng.uniform(0.05, 0.95, (5, 2)), np.array([0, 0, 1, 1, 2])
    eps, weights = rng.uniform(0.01, 0.2, (5, 2)), rng.uniform(0.1, 2, 2)
    target, normalize = seed % 4, bool(seed % 2)
    result = exact_negative_loo_margin_milp(X, y, target, weights=weights, eps=eps,
                                            normalize=normalize)
    assert result["optimal"], result["message"]
    local_weights = np.asarray(result["effective_weights"])
    lo, hi = interval_bounds(X, eps)
    for _ in range(80):
        z = rng.uniform(lo, hi)
        actual = _negative_margin(z, y, target, local_weights)
        assert actual <= result["upper_bound"]+1e-7
        assert actual >= result["analytic_lower_bound"]-1e-9
    assert result["lower_bound"] == pytest.approx(
        _negative_margin(result["worst_case"], y, target, local_weights))
    # HiGHS may declare numerical optimality at its primal feasibility
    # tolerance; the recomputed legal witness can expose a nonzero gap.
    # Preserve that gap rather than replacing the attack by -solver.fun.
    assert result["gap"] < 2e-6 * max(1.0, float(local_weights.sum()))


@pytest.mark.parametrize("seed", range(20))
def test_shared_anchor_three_point_endpoint_formula_matches_exact_milp(seed):
    rng = np.random.default_rng(9800+seed)
    lower = rng.uniform(-1.0, 0.5, (3, 2))
    upper = lower+rng.uniform(0.0, 1.5, (3, 2))
    X, eps = (lower+upper)/2, (upper-lower)/2
    weights = rng.uniform(0.1, 1.5, 2)
    # Rows are anchor, same-class friend h, opposite-class enemy j.
    scalar_minima = []
    for k in range(2):
        def objective(anchor):
            near_enemy = max(lower[2, k]-anchor, anchor-upper[2, k], 0)
            far_friend = max(abs(anchor-lower[1, k]), abs(anchor-upper[1, k]))
            return near_enemy-far_friend
        endpoint_min = min(objective(lower[0, k]), objective(upper[0, k]))
        breakpoints = np.clip([lower[0, k], upper[0, k], lower[2, k], upper[2, k],
                              (lower[1, k]+upper[1, k])/2], lower[0, k], upper[0, k])
        assert endpoint_min == pytest.approx(min(map(objective, breakpoints)))
        scalar_minima.append(endpoint_min)
    exact = exact_negative_loo_margin_milp(X, [0, 0, 1], 0, weights=weights,
                                          eps=eps, clip=False)
    formula_negative_margin = -float(weights @ scalar_minima)
    assert exact["optimal"], exact["message"]
    assert exact["lower_bound"] <= formula_negative_margin + 1e-7
    assert exact["upper_bound"] >= formula_negative_margin - 1e-7
    assert exact["lower_bound"] == pytest.approx(formula_negative_margin, abs=2e-6)
