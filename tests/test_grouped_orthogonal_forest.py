import numpy as np
import pytest
from certified_frfs.grouped_orthogonal_forest import fit_grouped_forest
from certified_frfs.orthogonal_forest_logistic import fit_orthogonal_forest, orthogonal_scores


@pytest.mark.parametrize('q', [2, 3])
def test_zero_group_penalty_matches_common_classifier_objective(q):
    rng = np.random.default_rng(772)
    X = rng.uniform(size=(42, 4)); y = np.arange(42) % q; edges = [[0, 1], [1, 2], [2, 3]]
    fit = fit_grouped_forest(X, y, edges, C=2., pair_penalty=3., relative_strength=0.)
    assert fit['usable']
    common = fit_orthogonal_forest(X, y, list(range(4)), edges, C=2., pair_penalty=3.)
    np.testing.assert_allclose(orthogonal_scores(X, fit), orthogonal_scores(X, common), atol=5e-5, rtol=5e-5)


@pytest.mark.parametrize('q', [2, 3])
def test_dual_allocation_threshold_is_sufficient_for_zero_solution(q):
    rng = np.random.default_rng(518)
    X = rng.uniform(size=(45, 5)); y = np.arange(45) % q; edges = [[0, 1], [0, 2], [2, 3], [2, 4]]
    fit = fit_grouped_forest(X, y, edges, relative_strength=1.01)
    assert fit['usable'] and fit['strength'] > fit['sufficient_zero_threshold'] > 0
    assert max(fit['node_group_norms']) < 1e-5
    np.testing.assert_allclose(orthogonal_scores(X, fit), 0, atol=1e-5)
