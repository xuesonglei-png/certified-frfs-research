"""Signed relative-retention semantics and independent small-instance checks."""
import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from certified_frfs.core import interval_bounds, positive_region
from certified_frfs.relative_verification import exact_relative_margin_milp
from certified_frfs.verification import _MILP


def _margin(z, y, features, weights, retention, v=None):
    v = np.ones(len(z)) if v is None else np.asarray(v, dtype=float)
    v = v / v.sum()
    return float(v @ (retention * positive_region(z, y, weights=weights)
                     - positive_region(z, y, features, weights)))


@pytest.mark.parametrize("features,expected", [([0], -0.05), ([1], 0.35), ([], 0.4)])
def test_singleton_box_preserves_positive_and_negative_margins(features, expected):
    result = exact_relative_margin_milp([[0, 0], [0.9, 0.1]], [0, 1], features,
                                        [0.5, 0.5], eps=0, retention=0.8)
    assert result["optimal"]
    assert result["lower_bound"] == pytest.approx(expected)
    assert result["upper_bound"] == pytest.approx(expected)
    assert result["gap"] == 0


@pytest.mark.parametrize("features,expected", [([0], 0.24), ([], 0.52), ([0, 1], -0.07)])
def test_two_point_box_has_known_signed_optimum(features, expected):
    result = exact_relative_margin_milp([[0, 0], [0.5, 0.5]], [0, 1], features,
                                        [0.5, 0.5], eps=[[0, 0], [0.1, 0.2]],
                                        retention=0.8)
    assert result["optimal"], result["message"]
    assert result["lower_bound"] == pytest.approx(expected)
    assert result["upper_bound"] == pytest.approx(expected, abs=1.1e-8)
    if features == [0, 1]:
        assert result["upper_bound"] < 0


def test_class_supported_weights_and_interior_witness():
    X = np.array([[0.5], [0], [1]])
    y = [0, 1, 1]
    result = exact_relative_margin_milp(X, y, [], [1], eps=[[0.5], [0], [0]],
                                        retention=0.8, object_weights=[2, 0, 0])
    assert result["optimal"]
    assert result["lower_bound"] == pytest.approx(0.4)
    assert result["worst_case"][0, 0] == pytest.approx(0.5)
    assert result["object_weights"] == [1, 0, 0]
    for endpoint in (0, 1):
        z = X.copy()
        z[0, 0] = endpoint
        assert _margin(z, y, [], [1], 0.8, [1, 0, 0]) == 0


def test_zero_information_is_vacuously_retained_and_one_class_is_distinct():
    zero = exact_relative_margin_milp([[0.5], [0.5]], [0, 1], [], eps=0)
    assert zero["lower_bound"] == zero["upper_bound"] == 0
    full = exact_relative_margin_milp([[0.1], [0.8]], [0, 1], [0], retention=1)
    assert full["lower_bound"] == full["upper_bound"] == 0
    no_weight = exact_relative_margin_milp([[0.1], [0.8]], [0, 1], [], weights=[0])
    assert no_weight["upper_bound"] == 0
    single_class = exact_relative_margin_milp([[0.1], [0.8]], [1, 1], [], retention=0.9)
    assert single_class["lower_bound"] == pytest.approx(-0.1)
    assert single_class["upper_bound"] == pytest.approx(-0.1)


@pytest.mark.parametrize("retention", [0, -0.1, 1.1, np.nan, [0.9]])
def test_bad_retention_is_rejected(retention):
    with pytest.raises(ValueError, match="retention"):
        exact_relative_margin_milp([[0], [1]], [0, 1], [], retention=retention)


@pytest.mark.parametrize("object_weights", [[0, 0], [-1, 2], [1], [np.inf, 1], [np.nan, 1]])
def test_bad_object_weights_are_rejected(object_weights):
    with pytest.raises(ValueError, match="object_weights"):
        exact_relative_margin_milp([[0], [1]], [0, 1], [], object_weights=object_weights)


def test_timeout_without_incumbent_or_dual_keeps_a_signed_safe_interval(monkeypatch):
    monkeypatch.setattr(_MILP, "solve", lambda *args: SimpleNamespace(
        x=None, status=1, message="Time limit reached"))
    result = exact_relative_margin_milp([[0], [0.8]], [0, 1], [0],
                                        weights=[1], eps=0.1, retention=0.9)
    # Minimal separation is 0.6, so the exact worst margin is -0.06.
    assert result["lower_bound"] == pytest.approx(-0.08)
    assert result["upper_bound"] == pytest.approx(-0.06)
    assert not result["optimal"]
    assert not result["solver_has_incumbent"]
    assert not result["solver_has_dual_bound"]
    assert "nominal witness" in result["message"]
    assert "analytic upper bound" in result["message"]


def test_timeout_with_dual_but_no_incumbent_does_not_claim_attainment(monkeypatch):
    monkeypatch.setattr(_MILP, "solve", lambda *args: SimpleNamespace(
        x=None, mip_dual_bound=-0.21, status=1, message="Time limit reached"))
    result = exact_relative_margin_milp([[0, 0], [0.5, 0.5]], [0, 1], [0],
                                        [0.5, 0.5], eps=0.1, retention=0.8)
    assert result["lower_bound"] == pytest.approx(0.15)
    assert result["upper_bound"] == pytest.approx(0.21, abs=1.1e-8)
    assert result["used_solver_dual_bound"]
    assert not result["optimal"]


def test_solver_objective_is_not_used_as_attack_and_witness_is_projected(monkeypatch):
    def fake_solve(self, *args):
        x = np.zeros(len(self.lower))
        x[:2] = [-100, 100]
        return SimpleNamespace(x=x, fun=-1000, status=1, message="Time limit reached")
    monkeypatch.setattr(_MILP, "solve", fake_solve)
    X = np.array([[0.2], [0.7]])
    result = exact_relative_margin_milp(X, [0, 1], [], [1], eps=0.1, retention=0.8)
    lo, hi = interval_bounds(X, 0.1)
    assert np.all(result["worst_case"] >= lo)
    assert np.all(result["worst_case"] <= hi)
    assert result["lower_bound"] == pytest.approx(0.56)
    assert result["witness_projection_max"] > 99
    assert not result["optimal"]


def test_inconsistent_dual_is_discarded(monkeypatch):
    monkeypatch.setattr(_MILP, "solve", lambda *args: SimpleNamespace(
        x=None, mip_dual_bound=10, status=1, message="Bad numerical dual"))
    result = exact_relative_margin_milp([[0], [0.8]], [0, 1], [], [1], retention=0.9)
    assert result["numerical_bound_conflict"]
    assert not result["used_solver_dual_bound"]
    assert result["upper_bound"] >= result["lower_bound"]


@pytest.mark.parametrize("seed", range(4))
def test_random_three_object_instances_dominate_grid_attacks(seed):
    rng = np.random.default_rng(1100 + seed)
    X, y = rng.uniform(0.1, 0.9, (3, 2)), [0, 1, 1]
    eps, weights = rng.uniform(0.02, 0.2, (3, 2)), rng.uniform(0.2, 1.5, 2)
    v, retention = rng.uniform(0, 1, 3), float(rng.uniform(0.3, 1))
    if seed % 2:
        v[1:] = 0
    result = exact_relative_margin_milp(X, y, [0], weights, eps,
                                        object_weights=v, retention=retention)
    assert result["optimal"], result["message"]
    lo, hi = interval_bounds(X, eps)
    grid_max = -np.inf
    for proportions in itertools.product((0, 0.5, 1), repeat=X.size):
        z = lo + np.reshape(proportions, X.shape) * (hi - lo)
        grid_max = max(grid_max, _margin(z, y, [0], weights, retention, v))
    assert grid_max <= result["upper_bound"] + 1e-7
    assert result["lower_bound"] == pytest.approx(
        _margin(result["worst_case"], y, [0], weights, retention, v))
    assert result["gap"] <= 1e-7
