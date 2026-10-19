"""Normalized distances, nonmonotone margins and solver fallback semantics."""
import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from certified_frfs.core import interval_bounds, positive_region
from certified_frfs.normalized_verification import exact_normalized_margin_milp
from certified_frfs.verification import _MILP


def _margin(X, y, features, weights, q, v=None):
    weights = np.asarray(weights, dtype=float)
    retained = np.zeros_like(weights)
    retained[features] = weights[features] / weights[features].sum()
    v = np.ones(len(X)) if v is None else np.asarray(v, dtype=float)
    v = v / v.sum()
    return float(v @ (q * positive_region(X, y, weights=weights / weights.sum())
                     - positive_region(X, y, features, retained)))


def test_twenty_repeated_columns_allow_95_percent_deletion_with_negative_worst_margin():
    X = np.vstack([np.zeros(20), np.ones(20)])
    result = exact_normalized_margin_milp(X, [0, 1], [7], eps=0.01, retention=0.9)
    expected = 0.9 * (19 + 0.98) / 20 - 0.98
    assert result["optimal"], result["message"]
    assert 1-len(result["features"])/20 == pytest.approx(0.95)
    assert result["lower_bound"] == pytest.approx(expected)
    assert result["upper_bound"] == pytest.approx(-0.0809, abs=1.1e-8)
    assert result["upper_bound"] < 0
    assert result["analytic_upper_bound"] == pytest.approx(-0.08)
    # The retained column is reduced to its minimum separation while the
    # 19 discarded columns attain maximum separation in the same table.
    distances = abs(result["worst_case"][0] - result["worst_case"][1])
    assert distances[7] == pytest.approx(0.98)
    assert np.sum(distances) == pytest.approx(19.98)


def test_normalization_allows_reduced_dependency_above_full_and_margin_below_q_minus_one():
    X = [[0, 0], [1, 0]]
    result = exact_normalized_margin_milp(X, [0, 1], [0], [1, 1], eps=0, retention=0.9)
    assert result["lower_bound"] == pytest.approx(-0.55)
    assert result["upper_bound"] == pytest.approx(-0.55)
    assert result["upper_bound"] < result["retention"] - 1


@pytest.mark.parametrize("features,expected", [([0], -0.25), ([1], 0.405)])
def test_two_point_box_known_signed_extrema_without_monotonicity(features, expected):
    X = [[0, 0], [0.8, 0.1]]
    result = exact_normalized_margin_milp(X, [0, 1], features, [1, 1],
                    eps=[[0, 0], [0.1, 0.2]], retention=0.9)
    assert result["optimal"]
    assert result["lower_bound"] == pytest.approx(expected)
    assert result["upper_bound"] == pytest.approx(expected, abs=1.1e-8)


@pytest.mark.parametrize("features,expected", [([0], 0.005), ([1], 0.25)])
def test_unequal_weights_use_separate_denominators_and_are_scale_invariant(features, expected):
    X, y, eps = [[0, 0], [0.5, 0.4]], [0, 1], [[0, 0], [0.1, 0.2]]
    for scale in (1, 10):
        result = exact_normalized_margin_milp(X, y, features, np.array([3, 1])*scale,
                                             eps=eps, retention=0.9)
        assert result["optimal"]
        assert result["lower_bound"] == pytest.approx(expected)
        assert result["upper_bound"] == pytest.approx(expected, abs=1.1e-8)
        assert sum(result["full_normalized_weights"]) == pytest.approx(1)
        assert sum(result["retained_normalized_weights"]) == pytest.approx(1)


def test_full_set_zero_omitted_weights_and_single_class_shortcuts():
    for features in ([0], [0, 1]):
        result = exact_normalized_margin_milp([[0, 0], [1, 1]], [0, 1], features,
                                             [1, 0], retention=1)
        assert result["lower_bound"] == result["upper_bound"] == 0
        assert result["optimal"]
    full = exact_normalized_margin_milp([[0, 0], [1, 1]], [0, 1], [0, 1],
                                        [1, 1], retention=0.9, eps=0.01)
    assert full["optimal"]
    assert full["lower_bound"] == pytest.approx(-0.098)
    assert full["upper_bound"] == pytest.approx(-0.098, abs=1.1e-8)
    single = exact_normalized_margin_milp([[0, 0], [1, 1]], [0, 0], [0], retention=0.9)
    assert single["upper_bound"] == pytest.approx(-0.1)


@pytest.mark.parametrize("features,weights", [([], [1, 1]), ([0], [0, 1]),
                                               ([0, 1], [0, 0])])
def test_empty_or_zero_mass_attribute_sets_are_rejected(features, weights):
    with pytest.raises(ValueError, match="weight sums"):
        exact_normalized_margin_milp([[0, 0], [1, 1]], [0, 1], features, weights)


@pytest.mark.parametrize("X,eps,clip", [([[-0.1], [0.5]], 0, False),
                                       ([[0], [1]], 0.01, False),
                                       ([[0.2], [0.5]], 0.3, False)])
def test_nominal_and_box_must_stay_in_unit_domain(X, eps, clip):
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        exact_normalized_margin_milp(X, [0, 1], [0], eps=eps, clip=clip)


def test_clip_false_is_supported_for_an_entirely_internal_box():
    a = exact_normalized_margin_milp([[0.2, 0.3], [0.7, 0.8]], [0, 1], [0], eps=0.01)
    b = exact_normalized_margin_milp([[0.2, 0.3], [0.7, 0.8]], [0, 1], [0], eps=0.01,
                                    clip=False)
    assert a["lower_bound"] == pytest.approx(b["lower_bound"])
    assert a["upper_bound"] == pytest.approx(b["upper_bound"])


def test_missing_solver_data_keep_signed_analytic_bounds_and_nominal_witness(monkeypatch):
    monkeypatch.setattr(_MILP, "solve", lambda *args: SimpleNamespace(
        x=None, status=1, message="Time limit reached"))
    result = exact_normalized_margin_milp([[0, 0], [0.8, 0.1]], [0, 1], [0], [1, 1],
                                          eps=[[0, 0], [0.1, 0.2]], retention=0.9)
    assert result["lower_bound"] == pytest.approx(0.9*0.45-0.8)
    assert result["analytic_lower_bound"] == pytest.approx(0.9*0.35-0.9)
    assert result["upper_bound"] == pytest.approx(0.9*0.6-0.7)
    assert result["upper_bound"] < -0.1
    assert not result["optimal"]
    assert not result["solver_has_incumbent"]
    assert not result["solver_has_dual_bound"]


def test_dual_conflict_is_discarded_and_attack_is_recomputed_after_projection(monkeypatch):
    def fake(self, *args):
        x = np.zeros(len(self.lower))
        x[:4] = [-10, -10, 10, 10]
        return SimpleNamespace(x=x, fun=-1000, mip_dual_bound=1000,
                               status=1, message="Numerically invalid dual")
    monkeypatch.setattr(_MILP, "solve", fake)
    X, y, weights, features = np.array([[0.2, 0.3], [0.7, 0.8]]), [0, 1], [1, 1], [0]
    result = exact_normalized_margin_milp(X, y, features, weights, eps=0.1)
    lo, hi = interval_bounds(X, 0.1)
    assert np.all(result["worst_case"] >= lo) and np.all(result["worst_case"] <= hi)
    assert result["lower_bound"] == pytest.approx(_margin(result["worst_case"], y, features,
                                                          weights, 0.9))
    assert result["numerical_bound_conflict"]
    assert not result["used_solver_dual_bound"]
    assert not result["optimal"]


@pytest.mark.parametrize("seed", range(12))
def test_random_normalized_milps_dominate_endpoint_and_interior_attacks(seed):
    rng = np.random.default_rng(5800 + seed)
    X, y = rng.uniform(0.05, 0.95, (3, 2)), np.array([0, 1, 1])
    weights, eps = rng.uniform(0.1, 3, 2), rng.uniform(0.01, 0.25, (3, 2))
    retention = float(rng.uniform(0.3, 1))
    v = np.array([1, 0, 0]) if seed % 2 else rng.uniform(0, 1, 3)
    features = [seed % 2]
    result = exact_normalized_margin_milp(X, y, features, weights, eps,
                                          object_weights=v, retention=retention)
    assert result["optimal"], result["message"]
    lo, hi = interval_bounds(X, eps)
    attacks = [np.where(np.reshape(bits, X.shape), hi, lo)
               for bits in itertools.product((False, True), repeat=X.size)]
    attacks += [rng.uniform(lo, hi) for _ in range(100)]
    for z in attacks:
        margin = _margin(z, y, features, weights, retention, v)
        assert margin <= result["upper_bound"] + 1e-7
        assert margin >= result["analytic_lower_bound"] - 1e-9
    assert result["lower_bound"] == pytest.approx(_margin(result["worst_case"], y, features,
                                                          weights, retention, v))
    assert result["gap"] <= 1e-7
