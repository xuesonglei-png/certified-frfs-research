"""Independent adversarial checks of the exploratory relative certificates."""
import itertools
from types import SimpleNamespace

import numpy as np
import pytest

import certified_frfs.relative as relative
from certified_frfs.core import interval_bounds, positive_region
from certified_frfs.relative_verification import exact_relative_margin_milp


def _finite_breakpoint_graph_minimum(lower, upper, edges):
    # Scalar absolute-distance objectives are piecewise linear in the order
    # cells. One minimizing cell vertex places every constant-value block
    # at an interval endpoint; all endpoints, not just each node's own two,
    # therefore give an independent finite oracle for these tiny graphs.
    endpoints = np.unique(np.r_[lower, upper])
    choices = [endpoints[(endpoints >= lo) & (endpoints <= hi)]
               for lo, hi in zip(lower, upper)]
    return min(sum(c * abs(z[i] - z[j]) for i, j, c in edges)
               for z in itertools.product(*choices))


@pytest.mark.parametrize("seed", range(8))
def test_general_graph_minimum_dual_against_piecewise_linear_breakpoints(seed):
    rng = np.random.default_rng(2300 + seed)
    lo = rng.uniform(-1.5, 0.5, 4)
    hi = lo + rng.uniform(0.0, 1.3, 4)
    edges = [(i, j, float(rng.uniform(0.1, 1.5)))
             for i in range(4) for j in range(i + 1, 4)]
    # Check aggregation, reversed orientations, self edges and zero weights.
    edges += [(1, 0, 0.4), (0, 0, 3), (0, 2, 0)]
    exact = _finite_breakpoint_graph_minimum(lo, hi, edges)
    result = relative.graph_minimum_lower(lo, hi, edges)
    assert result["lower_bound"] <= exact + 1e-9
    assert result["lower_bound"] == pytest.approx(exact, abs=2e-8)
    assert result["primal_value"] == pytest.approx(exact, abs=2e-8)


def test_graph_minimum_can_be_strictly_above_independent_edge_minima():
    # The shared center cannot simultaneously meet both fixed leaves.
    result = relative.graph_minimum_lower([0, 0, 1], [1, 0, 1],
                                          [(0, 1, 1), (0, 2, 1)])
    assert result["independent_lower"] == 0
    assert result["lower_bound"] == pytest.approx(1, abs=1e-8)
    assert result["dual_value"] == pytest.approx(1)


def test_lp_primal_alone_is_not_used_as_lower_bound(monkeypatch):
    monkeypatch.setattr(relative, "linprog", lambda *args, **kwargs: SimpleNamespace(
        x=np.array([0.5, 0, 1, 0.5, 0.5]), status=1,
        ineqlin=SimpleNamespace(marginals=None)))
    result = relative.graph_minimum_lower([0, 0, 1], [1, 0, 1],
                                          [(0, 1, 1), (0, 2, 1)])
    assert result["primal_value"] == 1
    assert result["lower_bound"] == 0
    assert result["status"] == 1


def test_arbitrary_finite_lp_marginals_are_projected_to_a_feasible_flow(monkeypatch):
    # Even adversarial/inaccurate marginals must yield a sound dual after
    # projection. No claim of optimality should be needed for this property.
    monkeypatch.setattr(relative, "linprog", lambda *args, **kwargs: SimpleNamespace(
        x=None, status=1,
        ineqlin=SimpleNamespace(marginals=np.array([-200, 30, 40, -300, -50, 60]))))
    lo, hi = np.array([0, 0.2, 1]), np.array([0.5, 0.8, 1.1])
    edges = [(0, 1, 0.1), (1, 2, 0.4), (0, 2, 0.6)]
    result = relative.graph_minimum_lower(lo, hi, edges)
    assert np.all(np.abs(result["flow"]) <= [0.1, 0.4, 0.6])
    assert result["lower_bound"] <= _finite_breakpoint_graph_minimum(lo, hi, edges) + 1e-9
    assert result["primal_value"] is None


@pytest.mark.parametrize("seed", range(24))
def test_relative_bounds_against_milp_and_shared_box_attacks(seed):
    rng = np.random.default_rng(2900 + seed)
    n, d = 3 + seed % 2, 2 + (seed % 3 == 0)
    X = rng.uniform(0.03, 0.97, (n, d))
    y = np.array([0, 1, 1] if n == 3 else [0, 1, 1, 2 if seed % 3 == 0 else 0])
    weights = rng.uniform(0.05, 1.5, d)
    eps = rng.uniform(0.0, 0.18, (n, d))
    if seed % 4 == 0:
        eps[1:] = 0
    if seed % 6 == 0:
        weights[-1] = 0
    if seed % 8 == 0:
        features = []
    elif seed % 8 == 1:
        features = list(range(d))
    else:
        features = [0]
    retention = 1.0 if seed % 7 == 0 else float(rng.uniform(0.4, 0.95))
    lower, upper = interval_bounds(X, eps)
    attacks = []
    for _ in range(20):
        z = rng.uniform(lower, upper)
        attacks.append(retention * positive_region(z, y, weights=weights)
                       - positive_region(z, y, features, weights))
    attacks = np.array(attacks)
    for scope in ("global", "classwise"):
        results = [relative.relative_certificate(X, y, features, weights, eps,
                    retention=retention, scope=scope, method=method, stability="exact")
                   for method in ("independent", "hybrid")]
        for group_index, group in enumerate(results[0]["groups"]):
            v = np.ones(n) if scope == "global" else (y == group["group"]).astype(float)
            v /= v.sum()
            exact = exact_relative_margin_milp(X, y, features, weights, eps,
                                               retention=retention, object_weights=v)
            assert exact["optimal"], exact["message"]
            for result in results:
                certificate_upper = result["groups"][group_index]["upper_bound"]
                assert certificate_upper + 2e-7 >= exact["upper_bound"]
                assert certificate_upper + 2e-9 >= np.max(attacks @ v)
            assert (results[1]["groups"][group_index]["upper_bound"] <=
                    results[0]["groups"][group_index]["upper_bound"] + 1e-10)


def test_scopes_preserve_aggregation_and_reveal_a_bad_minority_class():
    # The global average can satisfy retention while a class violates it.
    X = np.array([[0, 0], [0.05, 0.8], [0.8, 0], [1, 0]])
    y = np.array([0, 1, 1, 1])
    weights, features, retention = [1, 0.3], [0], 0.7
    results = {scope: relative.relative_certificate(X, y, features, weights, eps=0,
                retention=retention, scope=scope) for scope in ("global", "classwise", "objectwise")}
    objects = np.array([g["upper_bound"] for g in results["objectwise"]["groups"]])
    assert results["global"]["groups"][0]["upper_bound"] == pytest.approx(objects.mean())
    for group in results["classwise"]["groups"]:
        assert group["upper_bound"] == pytest.approx(objects[y == group["group"]].mean())
    assert results["global"]["certified"]
    assert not results["classwise"]["certified"]


def test_signed_retained_graph_minimum_can_change_certification_decision():
    # Opposite leaves share the uncertain retained-coordinate center. Their
    # retained distances sum to one; independent minima sum only to 0.8.
    # The signed bound must exploit this minimization, not maximize it.
    X = np.array([[0.5, 0.5], [0, 0], [1, 0]])
    y, weights = [0, 1, 1], [1, 0.1]
    eps = np.array([[0.1, 0], [0, 0], [0, 0]])
    independent = relative.relative_certificate(X, y, [0], weights, eps,
                      retention=0.9, scope="global", method="independent")
    hybrid = relative.relative_certificate(X, y, [0], weights, eps,
                      retention=0.9, scope="global", method="hybrid")
    exact = exact_relative_margin_milp(X, y, [0], weights, eps, retention=0.9)
    assert independent["max_upper_bound"] == pytest.approx(0.005)
    assert not independent["certified"]
    assert hybrid["max_upper_bound"] == pytest.approx(-0.005 / 3, abs=2e-9)
    assert hybrid["certified"]
    assert hybrid["max_upper_bound"] == pytest.approx(exact["lower_bound"], abs=2e-9)
    groups = relative.relative_certificate(X, y, [0], weights, eps,
                      retention=0.9, scope="classwise", method="hybrid")["groups"]
    assert groups[1]["upper_bound"] == pytest.approx(-0.005, abs=2e-9)
    assert groups[0]["upper_bound"] == pytest.approx(0.005, abs=2e-9)


def test_structural_and_degenerate_scope_boundaries():
    for scope in ("global", "classwise", "objectwise"):
        result = relative.relative_certificate([[0.1], [0.9]], [0, 1], [0],
                    weights=[1], eps=0.05, retention=1, scope=scope, method="hybrid")
        assert result["certified"]
        assert result["max_upper_bound"] == 0
        single = relative.relative_certificate([[0.1], [0.9]], [1, 1], [],
                    weights=[1], eps=0.05, retention=0.8, scope=scope, method="hybrid")
        assert single["certified"]
        assert single["max_upper_bound"] == pytest.approx(-0.2)
        zero = relative.relative_certificate([[0.5], [0.5]], [0, 1], [],
                    weights=[1], eps=0, retention=0.8, scope=scope, method="hybrid")
        assert zero["certified"]
        assert zero["max_upper_bound"] == 0
