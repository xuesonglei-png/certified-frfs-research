import numpy as np
from certified_frfs.centered_forest_fuzzy_logistic import fit_forest_logistic as tensor_fit
from certified_frfs.centered_forest_fuzzy_logistic_2d import fit_forest_logistic as matrix_fit
from certified_frfs.forest_fuzzy_classifier import forest_scores
from certified_frfs.triangular_classifier import triangular_memberships


def test_two_dimensional_centering_preserves_objective_and_margin_feasibility():
    rng = np.random.default_rng(429)
    X = rng.random((27, 3)); y = (X[:, 0] > .5).astype(int)
    W = np.zeros((2, 3, 3)); W[1, 0] = [-2, 0, 2]; edges = [(0, 1), (1, 2)]
    fits = [f(X, y, W, [0, 0], [0, 1], [0, 1, 2], edges, C=.1) for f in (tensor_fit, matrix_fit)]
    assert all(f['feasible'] for f in fits)
    np.testing.assert_allclose(fits[0]['objective'], fits[1]['objective'], rtol=1e-6, atol=1e-7)
    for fit in fits:
        V = np.array(fit['pair_coef'])
        np.testing.assert_allclose(V.sum(axis=2), 0, atol=1e-7)
        np.testing.assert_allclose(V.sum(axis=3), 0, atol=1e-7)
        mask = np.array(fit['original_margins']) > 1e-10
        assert np.all(np.array(fit['deployed_margins'])[mask] >= .5*np.array(fit['original_margins'])[mask]-1e-8)
    scores = [forest_scores(X, f['coef'], f['intercept'], f['edges'], f['pair_coef']) for f in fits]
    np.testing.assert_allclose(scores[0], scores[1], atol=1e-4)


def test_multiclass_centering_and_gauge_are_equivalent():
    rng = np.random.default_rng(430)
    X = rng.random((24, 3)); y = triangular_memberships(X[:, 0]).argmax(axis=1)
    W = np.zeros((3, 3, 3)); W[:, 0] = 2*np.eye(3); edges = [(0, 1), (1, 2)]
    fits = [f(X, y, W, [0, 0, 0], [0, 1, 2], [0, 1, 2], edges, C=.1) for f in (tensor_fit, matrix_fit)]
    assert all(f['feasible'] for f in fits)
    np.testing.assert_allclose(fits[0]['objective'], fits[1]['objective'], rtol=1e-6, atol=1e-7)
    for fit in fits:
        V = np.array(fit['pair_coef'])
        np.testing.assert_allclose(V.sum(axis=2), 0, atol=1e-7)
        np.testing.assert_allclose(V.sum(axis=3), 0, atol=1e-7)
        original = np.array(fit['original_margins']); mask = original > 1e-10
        assert np.all(np.array(fit['deployed_margins'])[mask] >= .5*original[mask]-1e-8)
    scores = [forest_scores(X, f['coef'], f['intercept'], f['edges'], f['pair_coef']) for f in fits]
    np.testing.assert_allclose(scores[0], scores[1], atol=1e-4)
