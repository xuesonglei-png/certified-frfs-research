import numpy as np
from scipy.spatial.distance import cdist
from certified_frfs.partition_majority import partition_majority_bounds, cell_majority_lower
from certified_frfs.core import interval_bounds


def test_refinement_cannot_worsen_cell_coverage_bound():
    rng = np.random.default_rng(1401)
    X = rng.uniform(0, 1, (20, 4)); y = np.repeat([0, 1], 10); w = np.ones(4)/2
    one, _ = partition_majority_bounds(X, y, w, max_leaves=1)
    sixteen, leaves = partition_majority_bounds(X, y, w, max_leaves=16)
    assert np.all(sixteen >= one-1e-12)
    assert np.all(leaves <= 16)


def test_every_leaf_lower_bounds_actual_order_gap():
    rng = np.random.default_rng(1402)
    X = rng.uniform(0, 1, (16, 3)); y = np.repeat([0, 1], 8); w = np.array([.4, .3, .7])
    lo, hi = interval_bounds(X, .1)
    for i in range(16):
        lower = cell_majority_lower(lo, hi, i, lo[i], hi[i], y, w)
        for _ in range(20):
            Z = rng.uniform(lo, hi)
            distance = np.abs(Z-Z[i])@w
            enemies = y != y[i]; friends = (y == y[i]) & (np.arange(16) != i)
            gap = np.partition(distance[enemies], 2)[2]-np.partition(distance[friends], 2)[2]
            assert lower <= gap+1e-12


def test_certified_majority_samples_and_protected_scope():
    rng = np.random.default_rng(1403); y = np.repeat([0, 1, 2], 8)
    X = .35*y[:, None]+rng.uniform(0, .1, (24, 4)); w = np.ones(4)/2
    mask = np.arange(24) % 2 == 0
    bounds, _ = partition_majority_bounds(X, y, w, protected=mask)
    assert np.all(bounds[mask] > 0)
    assert np.all(np.isneginf(bounds[~mask]))
    for _ in range(20):
        Z = np.clip(X+rng.uniform(-.01, .01, X.shape), 0, 1)
        D = cdist(Z*w, Z*w, 'cityblock'); np.fill_diagonal(D, np.inf)
        nn = np.argsort(D, axis=1)[:, :5]
        assert np.all(((y[nn] == y[:, None]).sum(axis=1) >= 3)[mask])
