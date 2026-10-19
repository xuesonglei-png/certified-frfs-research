import numpy as np
import pytest
from certified_frfs.centered_forest_fuzzy_logistic_2d import fit_forest_logistic


def test_stronger_pair_penalty_shrinks_interactions_on_fixed_model():
    # Mechanism check only; the zero teacher intentionally has no protected rows.
    rng = np.random.default_rng(431)
    X = rng.random((36, 2)); y = np.logical_xor(X[:, 0] > .5, X[:, 1] > .5).astype(int)
    W = np.zeros((2, 2, 3))
    fits = [fit_forest_logistic(X, y, W, [0, 0], [0, 1], [0, 1], [(0, 1)], C=1., pair_penalty=g)
            for g in (1., 10.)]
    assert all(f['feasible'] for f in fits)
    assert np.linalg.norm(fits[1]['pair_coef']) <= np.linalg.norm(fits[0]['pair_coef'])+1e-6
    assert fits[1]['pair_penalty'] == 10.
    with pytest.raises(ValueError, match='interaction penalty'):
        fit_forest_logistic(X, y, W, [0, 0], [0, 1], [0, 1], [(0, 1)], pair_penalty=0.)


def test_no_interaction_edge_remains_valid_with_separate_penalty():
    rng = np.random.default_rng(432)
    X = rng.random((20, 2)); y = (X[:, 0] > .5).astype(int)
    W = np.zeros((2, 2, 3)); W[1, 0] = [-2, 0, 2]
    fit = fit_forest_logistic(X, y, W, [0, 0], [0, 1], [0, 1], [], C=.1, pair_penalty=10.)
    assert fit['feasible'] and fit['edges'] == [] and fit['pair_coef'] == [[], []]
