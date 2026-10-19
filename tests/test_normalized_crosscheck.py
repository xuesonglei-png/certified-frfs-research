"""Cross-check normalized certificates against an independent exact MILP."""
import numpy as np
import pytest

from certified_frfs.core import interval_bounds, positive_region
from certified_frfs.normalized_relative import (
    normalized_relative_certificate, select_normalized_relative,
)
from certified_frfs.normalized_verification import exact_normalized_margin_milp


def _memberships(X, y, features, weights):
    weights = np.asarray(weights, dtype=float)
    wc = weights / weights.sum()
    wb = np.zeros_like(weights)
    wb[features] = weights[features] / weights[features].sum()
    return positive_region(X, y, weights=wc), positive_region(X, y, features, wb)


@pytest.mark.parametrize("seed", range(24))
def test_multiclass_normalized_bounds_against_exact_class_margins(seed):
    rng = np.random.default_rng(8200 + seed)
    n, d = 3 + seed % 3, 2 + seed % 2
    X = rng.uniform(0.02, 0.98, (n, d))
    y = np.arange(n) % 3
    eps = rng.uniform(0.001, 0.14, (n, d))
    if seed % 4 == 0:
        X[0, :] = 0
        X[-1, :] = 1
    if seed % 6 == 0:
        eps[1:] = 0
    weights = rng.uniform(0.1, 2, d)
    if seed % 5 == 0:
        weights[-1] = 0
    if seed % 4 == 1:
        weights[0] = 1e-14
    weights *= (1.0, 1e-16, 1e-250, 1e100)[seed % 4]
    features = list(range(d)) if seed % 7 == 0 else [0]
    q = 1.0 if seed % 8 == 0 else float(rng.uniform(0.25, 0.95))
    certificates = [normalized_relative_certificate(X, y, features, weights, eps,
                       retention=q, scope="classwise", method=method, stability=stability)
                    for method, stability in (("independent", "interval"),
                                               ("hybrid", "interval"), ("hybrid", "exact"))]
    lo, hi = interval_bounds(X, eps)
    attacks = []
    for _ in range(30):
        z = rng.uniform(lo, hi)
        pc, pb = _memberships(z, y, features, weights)
        attacks.append(q * pc - pb)
    attacks = np.asarray(attacks)
    for group_index, group in enumerate(certificates[0]["groups"]):
        v = (y == group["group"]).astype(float)
        v /= v.sum()
        exact = exact_normalized_margin_milp(X, y, features, weights, eps,
                                            retention=q, object_weights=v)
        assert exact["optimal"], exact["message"]
        bounds = []
        for result in certificates:
            bound = result["groups"][group_index]["upper_bound"]
            bounds.append(bound)
            assert np.isfinite(bound)
            assert bound + 2e-7 >= exact["upper_bound"]
            assert bound + 2e-9 >= np.max(attacks @ v)
        assert bounds[1] <= bounds[0] + 1e-9
        assert bounds[2] <= bounds[1] + 1e-9


@pytest.mark.parametrize("scope", ["global", "classwise", "objectwise"])
@pytest.mark.parametrize("method", ["independent", "hybrid"])
def test_extremely_small_common_weight_scale_is_finite_and_safe_for_full_set(scope, method):
    # Positive finite weight scaling must not introduce reciprocal overflow.
    X, y, weights, features = [[0, 0], [1, 1]], [0, 1], [1e-310, 1e-310], [0, 1]
    result = normalized_relative_certificate(X, y, features, weights, eps=0.01,
                    retention=0.9, scope=scope, method=method, stability="exact")
    assert np.isfinite(result["max_upper_bound"])
    assert result["certified"]
    assert result["max_upper_bound"] >= -0.098 - 1e-9
    assert result["max_upper_bound"] == pytest.approx(-0.098, abs=1e-8)


@pytest.mark.parametrize("scope", ["global", "classwise", "objectwise"])
@pytest.mark.parametrize("method", ["independent", "hybrid"])
def test_extreme_imbalance_cannot_turn_nominal_margin_into_an_unsafe_upper_bound(scope, method):
    X, y, features, weights = [[0, 0], [1, 1]], [0, 1], [0], [1e-310, 1]
    result = normalized_relative_certificate(X, y, features, weights, eps=0.01,
                    retention=0.9, scope=scope, method=method, stability="exact")
    # A legal table has retained distance .98 and full distance effectively
    # 1, hence margin -.08, which exceeds the nominal margin -.1.
    assert np.isfinite(result["max_upper_bound"])
    assert result["max_upper_bound"] >= -0.08 - 1e-9
    exact = exact_normalized_margin_milp(X, y, features, weights, eps=0.01, retention=0.9)
    assert exact["lower_bound"] == pytest.approx(-0.08)
    assert result["max_upper_bound"] + 1e-7 >= exact["upper_bound"]


def test_nonmonotone_negative_margin_is_not_clamped_to_q_minus_one():
    X, y, weights = [[0, 0], [0.8, 0.1]], [0, 1], [1, 1]
    eps = [[0, 0], [0.1, 0.2]]
    exact = exact_normalized_margin_milp(X, y, [0], weights, eps, retention=0.9)
    assert exact["lower_bound"] == pytest.approx(-0.25)
    for method in ("independent", "hybrid"):
        result = normalized_relative_certificate(X, y, [0], weights, eps,
                                                  retention=0.9, method=method)
        assert result["max_upper_bound"] == pytest.approx(-0.25, abs=1e-8)
        assert result["max_upper_bound"] < -0.1


@pytest.mark.parametrize("method", ["independent", "hybrid"])
def test_extreme_weight_imbalance_cannot_falsely_certify_a_positive_actual_margin(method):
    X, y, weights, features = [[0, 0], [1, 1]], [0, 1], [1e-310, 1], [0]
    result = normalized_relative_certificate(X, y, features, weights, eps=0.01,
                    retention=0.99, scope="global", method=method, stability="exact")
    exact = exact_normalized_margin_milp(X, y, features, weights, eps=0.01, retention=0.99)
    assert exact["lower_bound"] == pytest.approx(0.01)
    assert result["max_upper_bound"] >= 0.01 - 1e-9
    assert not result["certified"]


def test_twenty_duplicate_columns_selector_verifies_against_original_full_reference():
    X, y = np.vstack([np.zeros(20), np.ones(20)]), [0, 1]
    for method in ("independent", "hybrid"):
        result = select_normalized_relative(X, y, weights=np.ones(20), eps=0.01,
                       retention=0.9, scope="classwise", method=method, order=list(range(20)))
        assert result["features"] == [19]
        assert not result["minimality_proven"]
        assert result["attribute_inclusion_monotone"] is False
        assert result["certificate"]["certified"]
        for step in result["history"]:
            if step["accepted"]:
                assert step["max_upper_bound"] <= -1e-10
        assert result["certificate"]["max_upper_bound"] >= -0.0809 - 1e-9
        assert result["certificate"]["max_upper_bound"] == pytest.approx(-0.0809, abs=1e-8)


def test_normalized_scopes_aggregate_nominal_signed_margins_without_monotonicity():
    X = np.array([[0, 0.1], [0.8, 0.2], [0.2, 1], [1, 0.9]])
    y, features, weights = np.array([0, 1, 2, 2]), [0], [0.1, 1]
    outputs = {scope: normalized_relative_certificate(X, y, features, weights, eps=0,
                    retention=0.9, scope=scope) for scope in ("global", "classwise", "objectwise")}
    signed = np.array([group["upper_bound"] for group in outputs["objectwise"]["groups"]])
    assert outputs["global"]["max_upper_bound"] == pytest.approx(signed.mean())
    for group in outputs["classwise"]["groups"]:
        assert group["upper_bound"] == pytest.approx(signed[y == group["group"]].mean())
