from itertools import product
import numpy as np
import pytest
from certified_frfs.forest_fuzzy_classifier import forest_scores, forest_box_margins
from certified_frfs.triangular_classifier import triangular_box_margins


def test_forest_minimum_matches_exhaustive_raw_grid_and_attaining_witness():
    rng = np.random.default_rng(424)
    X = rng.random((5, 4)); y = np.array([0, 1, 2, 0, 1]); classes = np.arange(3)
    W = rng.normal(size=(3, 4, 3)); b = rng.normal(size=3)
    edges = [(0, 1), (1, 2), (1, 3)]; V = rng.normal(size=(3, 3, 3, 3))
    result = forest_box_margins(X, y, W, b, classes, edges, V, eps=.2, witnesses=True)
    for i, x in enumerate(X):
        lo, hi = np.maximum(0, x-.2), np.minimum(1, x+.2)
        grid = np.array(list(product(*[(a, z, np.clip(.5, a, z)) for a, z in zip(lo, hi)])))
        # Independent explicit hat evaluation, independent of oracle scores.
        Z = np.stack([np.maximum(0, 1-2*grid), 1-np.abs(2*grid-1), np.maximum(0, 2*grid-1)], axis=-1)
        scores = Z.reshape(len(grid), -1)@W.reshape(3, -1).T+b
        for e, (u, v) in enumerate(edges):
            for c in classes:
                scores[:, c] += np.sum(Z[:, u, :, None]*Z[:, v, None, :]*V[c, e], axis=(1, 2))
        expected = min(np.min(scores[:, y[i]]-scores[:, c]) for c in classes if c != y[i])
        np.testing.assert_allclose(result['margins'][i], expected, atol=1e-12)
    witness = result['witness_raw']
    assert np.all(witness >= np.maximum(0, X-.2)) and np.all(witness <= np.minimum(1, X+.2))
    scores = forest_scores(witness, W, b, edges, V)
    np.testing.assert_allclose(scores[np.arange(len(X)), y]-scores[np.arange(len(X)), result['rivals']], result['margins'], atol=1e-12)


def test_zero_pair_terms_reduce_to_original_additive_certificate():
    rng = np.random.default_rng(425)
    X = rng.random((7, 5)); y = np.arange(7)%2; W = rng.normal(size=(2, 5, 3)); b = rng.normal(size=2)
    edges = [(0, 1), (2, 4)]; V = np.zeros((2, 2, 3, 3))
    np.testing.assert_allclose(forest_box_margins(X, y, W, b, [0, 1], edges, V),
                               triangular_box_margins(X, y, W, b, [0, 1]), atol=1e-12)


def test_cycle_and_noncanonical_edges_are_rejected():
    X = np.zeros((3, 3)); W = np.zeros((2, 3, 3)); b = np.zeros(2)
    with pytest.raises(ValueError, match='forest'):
        forest_scores(X, W, b, [(0, 1), (1, 2), (0, 2)], np.zeros((2, 3, 3, 3)))
    with pytest.raises(ValueError, match='canonical'):
        forest_scores(X, W, b, [(1, 0)], np.zeros((2, 1, 3, 3)))
