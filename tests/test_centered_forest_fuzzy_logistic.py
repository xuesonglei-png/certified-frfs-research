import numpy as np
from certified_frfs.centered_forest_fuzzy_logistic import fit_forest_logistic


def test_centered_interaction_has_no_row_or_column_main_effect():
    rng = np.random.default_rng(428)
    X = rng.random((24, 2)); y = (X[:, 0] > .5).astype(int)
    W = np.zeros((2, 2, 3)); W[1, 0] = [-2, 0, 2]
    fit = fit_forest_logistic(X, y, W, [0, 0], [0, 1], [0, 1], [(0, 1)], C=.1)
    assert fit['feasible'] and fit['centered_interactions']
    V = np.array(fit['pair_coef'])
    np.testing.assert_allclose(V.sum(axis=2), 0, atol=1e-7)
    np.testing.assert_allclose(V.sum(axis=3), 0, atol=1e-7)
    original = np.array(fit['original_margins']); actual = np.array(fit['deployed_margins'])
    assert np.all(actual[original > 1e-10] >= .5*original[original > 1e-10]-1e-8)
