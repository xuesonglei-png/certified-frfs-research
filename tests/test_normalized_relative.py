"""Semantic, nonmonotonicity and attack checks for the distinct normalized model."""
import itertools

import numpy as np
import pytest

from certified_frfs.core import interval_bounds
from certified_frfs.normalized_relative import (
    normalized_relative_certificate, select_normalized_relative,
)
from certified_frfs.normalized_verification import exact_normalized_margin_milp


def _normalized_memberships(table, y, features, weights):
    # Independent direct implementation for witnesses, not a certificate.
    out = []
    weight_sum = sum(weights[k] for k in features)
    for i in range(len(table)):
        distances = [sum(weights[k] * abs(table[i, k] - table[j, k]) for k in features)
                     / weight_sum for j in range(len(table)) if y[i] != y[j]]
        out.append(min(distances) if distances else 1.0)
    return np.asarray(out)


def _assert_attack_below_bounds(cert, table, y, features, weights):
    pc = _normalized_memberships(table, y, list(range(table.shape[1])), weights)
    pb = _normalized_memberships(table, y, features, weights)
    violation = cert["retention"] * pc - pb
    for group in cert["groups"]:
        if cert["scope"] == "global":
            indices = np.ones(len(y), bool)
        elif cert["scope"] == "objectwise":
            indices = np.arange(len(y)) == group["group"]
        else:
            indices = y == group["group"]
        assert violation[indices].mean() <= group["upper_bound"] + 2e-9


def test_hand_calculated_zero_radius_and_negative_loss():
    X = np.array([[0.0, 0.0], [1.0, 0.2]])
    y = np.array([0, 1])
    strong = normalized_relative_certificate(X, y, [0], [1, 1], eps=0, retention=0.9)
    weak = normalized_relative_certificate(X, y, [1], [1, 1], eps=0, retention=0.9)
    assert strong["max_upper_bound"] == pytest.approx(0.9*0.6 - 1.0)
    assert weak["max_upper_bound"] == pytest.approx(0.9*0.6 - 0.2)
    assert strong["certified"] and not weak["certified"]
    assert strong["attribute_inclusion_monotone"] is False
    assert strong["groups"][0]["retained_dependency_nominal"] == pytest.approx(1)


def test_duplicate_columns_nominal_normalized_equality_and_selection():
    X = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    y = np.array([0, 1])
    cert = normalized_relative_certificate(X, y, [1], eps=0, retention=1)
    assert cert["max_upper_bound"] == pytest.approx(0)
    assert cert["certified"]
    chosen = select_normalized_relative(X, y, eps=0, retention=0.9, order=[0, 1, 2])
    assert chosen["features"] == [2]
    assert chosen["certificate"]["certified"]
    assert chosen["minimality_proven"] is False
    # A valid two-object table must not fail in the default MI-order helper.
    default_order = select_normalized_relative(X, y, eps=0, retention=0.9)
    assert default_order["features"] == chosen["features"]


def test_subset_inclusion_is_not_monotone_in_either_direction():
    X = np.array([[0.0, 0.0, 0.0], [1.0, 0.1, 0.9]])
    y, weights = np.array([0, 1]), np.ones(3)
    low = _normalized_memberships(X, y, [1], weights)
    middle = _normalized_memberships(X, y, [0, 1], weights)
    high = _normalized_memberships(X, y, [0], weights)
    assert np.all(low < middle)
    assert np.all(high > middle)
    # Both [0] and [1] are subsets of [0,1], but their dependencies move in
    # opposite directions; an additive-model inclusion proof is invalid.
    np.testing.assert_allclose(middle, 0.55)


@pytest.mark.parametrize("scope", ["classwise", "global", "objectwise"])
@pytest.mark.parametrize("stability", ["interval", "exact"])
def test_independent_and_signed_bounds_dominate_random_attacks(scope, stability):
    rng = np.random.default_rng(550 + (scope == "global"))
    for _ in range(8):
        X = rng.uniform(0.05, 0.95, (5, 3))
        y = np.array([0, 0, 0, 1, 1])
        weights = rng.uniform(0.2, 2, 3)
        features = [0, 2]
        lower, upper = interval_bounds(X, 0.04)
        independent = normalized_relative_certificate(X, y, features, weights,
                                                      eps=0.04, retention=0.8,
                                                      scope=scope, method="independent")
        hybrid = normalized_relative_certificate(X, y, features, weights,
                                                 eps=0.04, retention=0.8,
                                                 scope=scope, method="hybrid", stability=stability)
        for old, new in zip(independent["groups"], hybrid["groups"]):
            assert new["upper_bound"] <= old["upper_bound"] + 1e-12
        for point in (X, lower, upper, *(rng.uniform(lower, upper) for _ in range(30))):
            _assert_attack_below_bounds(independent, point, y, features, weights)
            _assert_attack_below_bounds(hybrid, point, y, features, weights)


def test_unique_neighbors_two_objects_have_exact_signed_extrema():
    # Both full and retained nearest neighbors are unique. No clipping of the
    # normalized distances occurs, so opposite-signed attribute extrema give
    # the exact worst margin: .8/2*.3 - (1-.8/2)*.5 = -.18.
    X = np.array([[0.2, 0.4], [0.8, 0.6]])
    y, weights = np.array([0, 1]), np.array([1.0, 1.0])
    lower, upper = interval_bounds(X, 0.05)
    expected = 0.8 / 2 * 0.3 - (1 - 0.8 / 2) * 0.5
    for scope in ("classwise", "global"):
        cert = normalized_relative_certificate(X, y, [0], weights, eps=0.05,
                                               retention=0.8, method="hybrid", scope=scope)
        assert cert["max_upper_bound"] == pytest.approx(expected, abs=1e-8)
        assert cert["stable_fraction"] == 1
        for bits in itertools.product((0, 1), repeat=4):
            point = np.where(np.array(bits).reshape(2, 2), upper, lower)
            _assert_attack_below_bounds(cert, point, y, [0], weights)


def test_bounds_against_independent_small_table_milp():
    rng = np.random.default_rng(8871)
    y = np.array([0, 0, 1, 1])
    for _ in range(4):
        X = rng.uniform(0.1, 0.9, (4, 2))
        weights = rng.uniform(0.3, 1.4, 2)
        for scope in ("classwise", "global"):
            independent = normalized_relative_certificate(
                X, y, [0], weights, eps=0.025, retention=0.85,
                scope=scope, method="independent")
            hybrid = normalized_relative_certificate(
                X, y, [0], weights, eps=0.025, retention=0.85,
                scope=scope, method="hybrid", stability="exact")
            for base, coupled in zip(independent["groups"], hybrid["groups"]):
                object_weights = (y == base["group"]).astype(float) if scope == "classwise" else None
                exact = exact_normalized_margin_milp(
                    X, y, [0], weights, eps=0.025, retention=0.85,
                    object_weights=object_weights, time_limit=8)
                assert exact["optimal"]
                assert exact["gap"] < 1e-6
                assert exact["lower_bound"] <= coupled["upper_bound"] + 2e-8
                assert coupled["upper_bound"] <= base["upper_bound"] + 1e-12
                _assert_attack_below_bounds(hybrid, exact["worst_case"], y, [0], weights)


def test_full_subset_single_class_and_zero_attribute_weight():
    X = np.array([[0.0, 0.2], [1.0, 0.8]])
    full = normalized_relative_certificate(X, [0, 1], eps=0.1, retention=1)
    assert full["certified"] and full["max_upper_bound"] <= 0
    equivalent = normalized_relative_certificate(X, [0, 1], [0], weights=[1, 0], retention=1)
    assert equivalent["certified"] and equivalent["max_upper_bound"] <= 0
    single = normalized_relative_certificate(X, [1, 1], [1], eps=0.1, retention=0.7,
                                             method="hybrid")
    assert single["max_upper_bound"] == pytest.approx(-0.3)
    assert single["certified"]
    chosen = select_normalized_relative(X, [0, 1], weights=[1, 0], order=[0, 1], eps=0)
    assert 0 in chosen["features"]
    assert chosen["certificate"]["retained_weight"] > 0


def test_each_accepted_deletion_and_final_subset_are_rechecked():
    X = np.array([[0.0, 0.0, 0.0], [0.9, 0.3, 0.8],
                  [0.1, 0.0, 0.1], [1.0, 0.4, 0.9]])
    y = np.array([0, 1, 0, 1])
    selected = select_normalized_relative(X, y, order=[1, 2, 0], method="hybrid",
                                         eps=0.01, retention=0.8)
    current = [0, 1, 2]
    for step in selected["history"]:
        if step["accepted"]:
            current.remove(step["candidate_removed"])
            cert = normalized_relative_certificate(X, y, current, method="hybrid",
                                                   eps=0.01, retention=0.8)
            assert cert["certified"]
    assert current == selected["features"]
    assert selected["certificate"]["certified"]


@pytest.mark.parametrize("kwargs", [
    {"features": []}, {"features": [1], "weights": [1, 0]},
    {"weights": [0, 0]}, {"weights": [-1, 1]}, {"retention": 0},
    {"retention": 1.1}, {"scope": "unknown"}, {"method": "unknown"},
    {"stability": "unknown"}, {"clip": False, "eps": 0.2},
])
def test_reject_invalid_normalized_models(kwargs):
    with pytest.raises(ValueError):
        normalized_relative_certificate(np.array([[0.1, 0.2], [0.9, 0.8]]), [0, 1], **kwargs)


def test_reject_outside_unit_domain_and_bad_orders():
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        normalized_relative_certificate(np.array([[-0.1, 0.2], [0.9, 0.8]]), [0, 1])
    with pytest.raises(ValueError, match="permutation"):
        select_normalized_relative(np.array([[0.1, 0.2], [0.9, 0.8]]), [0, 1], order=[0, 0])
