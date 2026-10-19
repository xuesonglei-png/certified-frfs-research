import numpy as np
from certified_frfs.forest_fuzzy_logistic import fit_forest_logistic, forest_design
from certified_frfs.forest_fuzzy_classifier import forest_scores, forest_box_margins


def test_forest_constrained_fit_preserves_teacher_boxes_and_induced_support():
    rng = np.random.default_rng(426)
    X = rng.random((24, 3)); y = (X[:, 0] > .5).astype(int)
    W = np.zeros((2, 3, 3)); W[1, 0] = [-2, 0, 2]; b = np.zeros(2)
    edges = [(0, 1), (1, 2)]
    fit = fit_forest_logistic(X, y, W, b, [0, 1], [0, 1], edges, C=.1)
    assert fit['feasible']
    assert fit['edges'] == [[0, 1]]
    assert np.all(np.array(fit['coef'])[:, 2] == 0)
    actual = forest_box_margins(X, y, fit['coef'], fit['intercept'], [0, 1], fit['edges'], fit['pair_coef'])
    original = np.array(fit['original_margins']); mask = original > 1e-10
    np.testing.assert_allclose(actual, fit['deployed_margins'], atol=1e-12)
    assert np.all(actual[mask] >= .5*original[mask]-1e-8) and np.all(actual[mask] > 1e-10)
    coef = np.column_stack([np.array(fit['coef'])[:, [0, 1]].reshape(2, -1), np.array(fit['pair_coef']).reshape(2, -1)])
    np.testing.assert_allclose(forest_design(X, [0, 1], [(0, 1)])@coef.T+fit['intercept'],
                               forest_scores(X, fit['coef'], fit['intercept'], fit['edges'], fit['pair_coef']), atol=1e-12)


def test_separation_limit_is_explicit_uncertified_failure():
    rng = np.random.default_rng(427)
    X = rng.random((20, 2)); y = (X[:, 0] > .5).astype(int)
    W = np.zeros((2, 2, 3)); W[1, 0] = [-100, 0, 100]
    fit = fit_forest_logistic(X, y, W, [0, 0], [0, 1], [0, 1], [(0, 1)], C=.001, max_rounds=1)
    assert not fit['feasible'] and fit['status'] == 'separation_limit'
    assert fit['trace'][0]['violated_objects']
