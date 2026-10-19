"""Disjunction of fixed three-friend witnesses for robust finite-table 5NN.

Each triple independently proves three friends precede all but two enemies.
Taking their maximum strengthens the sufficient bound, not the claim scope.
"""
from itertools import combinations
import numpy as np
from scipy.spatial.distance import cdist
from .core import _inputs, interval_bounds
from .margin import triplet_interval_min


def witness_majority_bounds(X, y, weights, *, eps=.01, pool_size=5):
    X, y, fs, weights = _inputs(X, y, None, weights)
    if int(pool_size) != pool_size or pool_size < 3:
        raise ValueError('at least three friend slots required')
    n, d = X.shape
    if n < 3:
        return np.full(n, -np.inf)
    lo, hi = interval_bounds(X, eps)
    distance = cdist(X*weights, X*weights, 'cityblock')
    friends = []
    for i, label in enumerate(y):
        ids = np.flatnonzero((y == label) & (np.arange(n) != i))
        ids = ids[np.lexsort((ids, distance[i, ids]))][:pool_size]
        friends.append(np.resize(ids, pool_size) if len(ids) else np.repeat(i, pool_size))
    friends = np.array(friends)
    cross = y[:, None] != y[None, :]
    coefficients = np.zeros((n, n, pool_size))
    for a in range(d):
        if weights[a] == 0:
            continue
        coefficients += weights[a]*triplet_interval_min(
            lo[:, a, None, None], hi[:, a, None, None],
            lo[None, :, a, None], hi[None, :, a, None],
            lo[friends, a][:, None, :], hi[friends, a][:, None, :])
    best = np.full(n, -np.inf)
    for triple in combinations(range(pool_size), 3):
        ids = friends[:, triple]
        valid = ((ids[:, 0] != ids[:, 1]) & (ids[:, 0] != ids[:, 2]) &
                 (ids[:, 1] != ids[:, 2]) & np.all(ids != np.arange(n)[:, None], axis=1))
        enemy_bounds = np.where(cross, coefficients[:, :, triple].min(axis=2), np.inf)
        lower = np.partition(enemy_bounds, 2, axis=1)[:, 2]
        best = np.maximum(best, np.where(valid, lower, -np.inf))
    padding = 1e-10*(1+np.abs(np.where(np.isfinite(best), best, 0)))
    return best-padding
