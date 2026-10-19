"""Semantic and adversarial tests; none treats box vertices as exact E."""
import itertools

import numpy as np
import pytest

from certified_frfs.core import (
    certificate, forest_maximum, gamma, interval_bounds, positive_region,
    _shared_anchor_difference_upper,
)
from certified_frfs.verification import exact_loss_milp, sampled_attack


def test_fixed_weights_and_empty_reduct_semantics():
    X = np.array([[0.0, 0.0], [0.4, 0.8]])
    y = [0, 1]
    weights = [0.5, 0.5]
    assert gamma(X, y, [0], weights) == pytest.approx(0.2)
    assert gamma(X, y, [0, 1], weights) == pytest.approx(0.6)
    assert gamma(X, y, [], weights) == 0.0
    assert gamma(X, [0, 0], [], weights) == 1.0
    assert certificate(X, y, [0, 1], weights, 0.1)["upper_bound"] == 0.0
    assert certificate(X, [0, 0], [], weights, 0.1)["upper_bound"] == 0.0


def test_clipping_and_bad_inputs():
    X = np.array([[0.0, 0.5], [0.9, 1.0]])
    lower, upper = interval_bounds(X, [0.2, 0.1])
    np.testing.assert_allclose(lower, [[0, 0.4], [0.7, 0.9]])
    np.testing.assert_allclose(upper, [[0.2, 0.6], [1, 1]])
    with pytest.raises(ValueError):
        interval_bounds(X, -0.1)
    with pytest.raises(ValueError):
        gamma(X, [0, 1], [0, 0])
    with pytest.raises(ValueError):
        interval_bounds(X + 1, 0.1)


def test_forest_dynamic_program_against_all_endpoint_assignments():
    rng = np.random.default_rng(902)
    for n in range(2, 8):
        lower = rng.uniform(-0.5, 0.5, (n, 2))
        upper = lower + rng.uniform(0, 1, (n, 2))
        edges = [(i, int(rng.integers(i)), float(rng.uniform(0.1, 2))) for i in range(1, n)]
        weights = np.array([0.7, 0.2])
        expected = 0.0
        for k in range(2):
            maximum = -np.inf
            for bits in itertools.product((0, 1), repeat=n):
                point = np.where(bits, upper[:, k], lower[:, k])
                maximum = max(maximum, sum(c * abs(point[i] - point[j]) for i, j, c in edges))
            expected += weights[k] * maximum
        assert forest_maximum(lower, upper, edges, weights) == pytest.approx(expected)
    with pytest.raises(ValueError, match="cycle"):
        forest_maximum(np.zeros((3, 1)), np.ones((3, 1)), [(0, 1, 1), (1, 2, 1), (2, 0, 1)])


def test_exact_stable_case_coupling_strictly_improves_independent_bound():
    # All B and C nearest neighbors stay fixed, and C distances stay <1.
    # Two stable arcs use edge (0,1), one uses (0,2). Shared z_02 cannot
    # simultaneously maximize its distance from both 0 and 1.
    X = np.array([[0.0, 0.5], [0.1, 0.0], [0.7, 1.0]])
    y, weights = np.array([0, 1, 1]), np.array([1.0, 0.2])
    eps = np.zeros_like(X)
    eps[0, 1] = 0.5
    result = certificate(X, y, [0], weights, eps)
    assert result["stable_fraction"] == 1.0
    assert result["upper_bound"] == pytest.approx(0.4 / 3)
    assert result["independent_bound"] == pytest.approx(0.2)
    assert result["upper_bound"] < result["independent_bound"] - 0.05
    exact = exact_loss_milp(X, y, [0], weights, eps)
    assert exact["optimal"]
    assert exact["lower_bound"] == pytest.approx(0.4 / 3)
    assert exact["upper_bound"] == pytest.approx(0.4 / 3, abs=1e-7)


def test_partial_stability_hybrid_is_sound_and_ties_are_not_stable():
    X = np.array([[0.5, 0.5], [0.0, 0.0], [1.0, 1.0]])
    eps = np.array([[0.15, 0.2], [0, 0], [0, 0]])
    y, weights = np.array([0, 1, 1]), np.array([0.7, 0.3])
    result = certificate(X, y, [0], weights, eps)
    assert result["stable_count"] == 2
    assert result["stable_neighbors"][0] == -1
    exact = exact_loss_milp(X, y, [0], weights, eps)
    assert exact["optimal"]
    assert exact["lower_bound"] <= result["upper_bound"] + 1e-8


def test_loss_maximum_can_be_strictly_inside_the_box():
    # This is a counterexample to "checking all vertices verifies E".
    X = np.array([[0.5], [0.0], [1.0]])
    y, eps = [0, 1, 1], np.array([[0.5], [0.0], [0.0]])
    for vertex in (0.0, 1.0):
        z = X.copy()
        z[0, 0] = vertex
        assert gamma(z, y, [], [1.0]) == 0.0
        assert gamma(z, y, weights=[1.0]) == pytest.approx(1 / 3)
    exact = exact_loss_milp(X, y, [], [1.0], eps)
    assert exact["optimal"]
    assert exact["lower_bound"] == pytest.approx(0.5)
    assert exact["upper_bound"] == pytest.approx(0.5, abs=1e-7)
    assert exact["adversarial_X"][0, 0] == pytest.approx(0.5)


@pytest.mark.parametrize("seed", range(6))
def test_random_small_tables_grid_attacks_below_all_bounds(seed):
    rng = np.random.default_rng(seed)
    X, y = rng.uniform(0.0, 1.0, (3, 2)), np.array([0, 1, 1])
    weights = rng.uniform(0.2, 0.8, 2)
    eps = rng.uniform(0.02, 0.3, (3, 2))
    lower, upper = interval_bounds(X, eps)
    results = {method: certificate(X, y, [0], weights, eps, method)
               for method in ("hybrid", "independent", "lipschitz")}
    attack_max = 0.0
    for choices in itertools.product((0.0, 0.5, 1.0), repeat=X.size):
        z = lower + np.reshape(choices, X.shape) * (upper - lower)
        loss = gamma(z, y, weights=weights) - gamma(z, y, [0], weights)
        assert loss >= -1e-12
        attack_max = max(attack_max, loss)
    for result in results.values():
        assert 0 <= result["upper_bound"] <= 1
        assert attack_max <= result["upper_bound"] + 1e-10
    assert results["hybrid"]["upper_bound"] <= min(
        results["independent"]["upper_bound"], results["lipschitz"]["upper_bound"]) + 1e-10
    exact = exact_loss_milp(X, y, [0], weights, eps, time_limit=10)
    assert exact["optimal"]
    assert attack_max <= exact["upper_bound"] + 1e-7
    assert exact["lower_bound"] <= results["hybrid"]["upper_bound"] + 1e-8
    assert exact["gap"] <= 1e-6


def test_zero_radius_certificate_is_exact():
    X = np.array([[0.1, 0.2], [0.4, 0.9], [0.9, 0.1]])
    for method in ("hybrid", "independent", "lipschitz"):
        result = certificate(X, [0, 1, 1], [1], [0.3, 0.7], 0.0, method)
        assert result["upper_bound"] == pytest.approx(result["nominal_loss"], abs=2e-12)


def test_sampled_attack_reports_an_attained_lower_bound():
    X = np.array([[0.2, 0.6], [0.7, 0.2], [0.4, 0.9]])
    y, weights = [0, 1, 1], [0.6, 0.4]
    attack = sampled_attack(X, y, [0], weights, 0.1, n_samples=25)
    z = attack["adversarial_X"]
    assert attack["lower_bound"] == pytest.approx(gamma(z, y, weights=weights) - gamma(z, y, [0], weights))
    assert attack["lower_bound"] <= certificate(X, y, [0], weights, 0.1)["upper_bound"] + 1e-10


def test_shared_anchor_comparison_against_all_piecewise_breakpoints():
    rng = np.random.default_rng(321)
    for _ in range(200):
        lower = rng.uniform(-1, 1, 3)
        upper = lower + rng.uniform(0, 2, 3)
        la, lb, lc = lower
        ua, ub, uc = upper
        # Breakpoints of max_b |a-b| - min_c |a-c|, plus box endpoints.
        anchors = np.clip([la, ua, (lb + ub) / 2, lc, uc], la, ua)
        brute = max(abs(a - b) - abs(a - c)
                    for a in anchors for b in (lb, ub)
                    for c in (lc, uc, np.clip(a, lc, uc)))
        exact = _shared_anchor_difference_upper(la, ua, lb, ub, lc, uc)
        assert exact == pytest.approx(brute)


def test_exact_stability_strictly_improves_interval_check_and_bound():
    X = np.array([[0.3, 0.5], [0.8, 0.8], [1.0, 0.0]])
    y, weights = [0, 1, 1], [0.5, 0.08]
    eps = np.array([[0.3, 0.5], [0.0, 0.0], [0.0, 0.0]])
    basic = certificate(X, y, [0], weights, eps)
    improved = certificate(X, y, [0], weights, eps, stability="exact")
    assert basic["stable_count"] == 2
    assert improved["stable_count"] == 3
    assert improved["upper_bound"] < basic["upper_bound"] - 0.005
    exact = exact_loss_milp(X, y, [0], weights, eps)
    assert exact["optimal"]
    assert improved["upper_bound"] == pytest.approx(exact["lower_bound"], abs=1e-8)
    assert improved["upper_bound"] == pytest.approx(0.08 * 1.6 / 3)


@pytest.mark.parametrize("seed", range(8))
def test_exact_stability_certificate_is_sound_on_random_boxes(seed):
    rng = np.random.default_rng(800 + seed)
    X, y = rng.uniform(0, 1, (4, 3)), np.array([0, 1, 0, 1])
    weights, eps = rng.uniform(0.1, 1, 3), rng.uniform(0.0, 0.3, (4, 3))
    basic = certificate(X, y, [0, 1], weights, eps)
    improved = certificate(X, y, [0, 1], weights, eps, stability="exact")
    assert improved["stable_count"] >= basic["stable_count"]
    assert improved["upper_bound"] <= basic["upper_bound"] + 1e-10
    attack = sampled_attack(X, y, [0, 1], weights, eps, n_samples=200, seed=seed)
    assert attack["lower_bound"] <= improved["upper_bound"] + 1e-10
    exact = exact_loss_milp(X, y, [0, 1], weights, eps, time_limit=5)
    assert exact["lower_bound"] <= improved["upper_bound"] + 1e-8
