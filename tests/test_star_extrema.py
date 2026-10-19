"""Independent LP and exhaustive checks of shared-coordinate star extrema."""
import itertools

import numpy as np
import pytest
from scipy.optimize import linprog

from certified_frfs.star_extrema import star_extrema


def _objective(table, edges, leaf_weights, attribute_weights):
    return sum(
        weight * sum(ak * abs(table[leaf, k] - table[center, k])
                     for k, ak in enumerate(attribute_weights))
        for (leaf, center), weight in zip(edges, leaf_weights)
    )


def _lp_minimum(lower, upper, edges, leaf_weights, attribute_weights):
    """Independent epigraph LP; this does not use interval median formulas."""
    n, d = lower.shape
    num_coordinates = n * d
    objective = np.r_[np.zeros(num_coordinates),
                      np.outer(leaf_weights, attribute_weights).ravel()]
    rows = []
    for e, (leaf, center) in enumerate(edges):
        for k in range(d):
            row = np.zeros(len(objective))
            row[leaf * d + k] = 1
            row[center * d + k] = -1
            row[num_coordinates + e * d + k] = -1
            rows.append(row)
            reverse = row.copy()
            reverse[leaf * d + k] *= -1
            reverse[center * d + k] *= -1
            rows.append(reverse)
    bounds = list(zip(lower.ravel(), upper.ravel())) + [(0, None)] * (len(edges) * d)
    solved = linprog(objective, A_ub=np.asarray(rows), b_ub=np.zeros(len(rows)),
                     bounds=bounds, method="highs")
    assert solved.success, solved.message
    return solved.fun


def _exhaustive_maximum(lower, upper, edges, leaf_weights, attribute_weights):
    total = 0.0
    for k, ak in enumerate(attribute_weights):
        values = []
        for bits in itertools.product((0, 1), repeat=len(lower)):
            table = np.where(bits, upper[:, k], lower[:, k])
            values.append(sum(weight * abs(table[leaf] - table[center])
                              for (leaf, center), weight in zip(edges, leaf_weights)))
        total += ak * max(values)
    return total


def test_shared_center_has_interior_minimizer_and_nonadditive_pairwise_minima():
    # The center interval overlaps both fixed leaves, but their distance sum
    # cannot be zero. Unequal weights force the center into its interior.
    lower = np.array([[0.25], [0.75], [0.0]])
    upper = np.array([[0.25], [0.75], [1.0]])
    result = star_extrema(lower, upper, [(0, 2), (1, 2)], [3, 1])
    assert result["minimum"] == pytest.approx(0.5)
    assert result["minimizer"][2, 0] == pytest.approx(0.25)
    assert result["maximum"] == pytest.approx(2.5)
    assert result["maximizer"][2, 0] == 1


def test_weighted_median_uses_both_interval_endpoints_and_center_projection():
    lower = np.array([[0.0, 0.0], [4.0, 4.0], [1.5, 3.0]])
    upper = np.array([[2.0, 2.0], [6.0, 6.0], [5.0, 5.0]])
    result = star_extrema(lower, upper, [(0, 2), (1, 2)], [2, 1], [1, 2])
    # First optimum t=2, second unconstrained optimum 2 projects to t=3.
    np.testing.assert_allclose(result["minimizer"][2], [2, 3])
    np.testing.assert_allclose(result["minimum_by_attribute"], [2, 6])
    assert result["minimum"] == pytest.approx(8)


def test_random_multistar_extrema_against_lp_and_all_box_endpoints():
    rng = np.random.default_rng(20260920)
    for _ in range(35):
        n = int(rng.integers(3, 9))
        d = int(rng.integers(1, 4))
        split = int(rng.integers(1, n))
        edges = [(leaf, int(rng.integers(split, n))) for leaf in range(split)]
        lower = rng.uniform(-2, 2, (n, d))
        upper = lower + rng.uniform(0, 2, (n, d))
        fixed = rng.random((n, d)) < 0.2
        upper[fixed] = lower[fixed]
        leaf_weights = rng.uniform(0.01, 3, len(edges))
        attribute_weights = rng.uniform(0.01, 2, d)
        leaf_weights[rng.random(len(edges)) < 0.2] = 0
        attribute_weights[rng.random(d) < 0.2] = 0
        snapshot = (lower.copy(), upper.copy())

        result = star_extrema(lower, upper, edges, leaf_weights, attribute_weights)
        expected_min = _lp_minimum(lower, upper, edges, leaf_weights, attribute_weights)
        expected_max = _exhaustive_maximum(lower, upper, edges, leaf_weights, attribute_weights)
        assert result["minimum"] == pytest.approx(expected_min, abs=1e-8)
        assert result["maximum"] == pytest.approx(expected_max, abs=1e-10)
        for mode in ("minimum", "maximum"):
            witness = result["minimizer" if mode == "minimum" else "maximizer"]
            assert np.all(witness >= lower)
            assert np.all(witness <= upper)
            assert result[mode] == pytest.approx(_objective(witness, edges, leaf_weights, attribute_weights))
            assert result[mode] == pytest.approx(result[f"{mode}_by_attribute"].sum())
        np.testing.assert_array_equal(lower, snapshot[0])
        np.testing.assert_array_equal(upper, snapshot[1])


def test_no_edges_and_isolated_vertices_and_zero_weights():
    lower = np.array([[0.0, 2.0], [1.0, 1.0], [0.5, 3.0], [-3.0, -1.0]])
    upper = lower + 1
    for edges, leaf_weights, attribute_weights in (
        ([], None, None),
        ([(0, 2), (1, 2)], [0, 0], [1, 1]),
        ([(0, 2), (1, 2)], [1, 2], [0, 0]),
    ):
        result = star_extrema(lower, upper, edges, leaf_weights, attribute_weights)
        assert result["minimum"] == result["maximum"] == 0
        np.testing.assert_array_equal(result["minimizer"], lower)
        np.testing.assert_array_equal(result["maximizer"], lower)
    result = star_extrema(lower, upper, [(0, 2), (1, 2)])
    np.testing.assert_array_equal(result["minimizer"][3], lower[3])
    np.testing.assert_array_equal(result["maximizer"][3], lower[3])
    for shape in ((0, 2), (3, 0), (0, 0)):
        result = star_extrema(np.empty(shape), np.empty(shape), [])
        assert result["minimum"] == result["maximum"] == 0
        assert result["minimizer"].shape == shape


def test_fixed_intervals_and_generator_edges():
    table = np.array([[0.0, 1.0], [2.0, -1.0], [0.5, 0.25]])
    edges = [(0, 2), (1, 2)]
    result = star_extrema(table, table, iter(edges), [0.5, 2], [3, 0.25])
    expected = _objective(table, edges, [0.5, 2], [3, 0.25])
    assert result["minimum"] == result["maximum"] == pytest.approx(expected)
    np.testing.assert_array_equal(result["minimizer"], table)
    np.testing.assert_array_equal(result["maximizer"], table)


@pytest.mark.parametrize("edges", [
    [(0, 1), (0, 2)], [(0, 1), (1, 2)], [(0, 0)], [(0, 3)],
    [(-1, 1)], [(0.0, 1)], [(True, 2)], [(0, 1, 2)], [0],
])
def test_reject_invalid_star_structure(edges):
    with pytest.raises(ValueError):
        star_extrema(np.zeros((3, 2)), np.ones((3, 2)), edges)


@pytest.mark.parametrize("leaf_weights,attribute_weights", [
    ([-1], None), ([np.nan], None), ([np.inf], None), ([], None),
    (None, [-1, 1]), (None, [np.inf, 1]), (None, [1]),
])
def test_reject_invalid_weights(leaf_weights, attribute_weights):
    with pytest.raises(ValueError):
        star_extrema(np.zeros((3, 2)), np.ones((3, 2)), [(0, 2)],
                     leaf_weights, attribute_weights)


@pytest.mark.parametrize("lower,upper", [
    ([0, 1], [1, 2]), ([[0, 1]], [[1]]), ([[2]], [[1]]),
    ([[np.nan]], [[1]]), ([[0]], [[np.inf]]),
])
def test_reject_invalid_interval_arrays(lower, upper):
    with pytest.raises(ValueError):
        star_extrema(lower, upper, [])
