import numpy as np
from scipy.spatial.distance import cdist
from certified_frfs.majority_metric import MajorityTerms
from certified_frfs.witness_majority import witness_majority_bounds


def test_three_slot_agreement_and_larger_pool_domination():
    rng = np.random.default_rng(1203)
    X = rng.uniform(0, 1, (24, 6)); y = np.repeat([0, 1, 2], 8)
    w = rng.uniform(.1, .5, 6)
    old = MajorityTerms(X, y, weights=w).bounds(range(6), allowed_enemies=2)
    three = witness_majority_bounds(X, y, w, pool_size=3)
    five = witness_majority_bounds(X, y, w, pool_size=5)
    np.testing.assert_allclose(three, old, atol=1e-12, rtol=0)
    assert np.all(five >= old-1e-12)


def test_padded_slots_never_make_three_distinct_friends():
    X = np.arange(20, dtype=float).reshape(10, 2)/20
    y = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
    result = witness_majority_bounds(X, y, np.ones(2), eps=0)
    assert np.all(np.isneginf(result[y != 1]))


def test_certified_objects_have_correct_five_neighbor_majority_under_samples():
    rng = np.random.default_rng(1204)
    y = np.repeat([0, 1, 2], 8)
    X = .35*y[:, None] + rng.uniform(0, .1, (24, 4))
    w = np.ones(4)/2
    bound = witness_majority_bounds(X, y, w)
    protected = bound > 1e-10
    assert protected.sum() == 24
    for _ in range(50):
        Z = np.clip(X+rng.uniform(-.01, .01, X.shape), 0, 1)
        D = cdist(Z*w, Z*w, 'cityblock'); np.fill_diagonal(D, np.inf)
        nn = np.argsort(D, axis=1, kind='stable')[:, :5]
        assert np.all(((y[nn] == y[:, None]).sum(axis=1) >= 3)[protected])
