import numpy as np
import pytest
from certified_frfs.centered_forest_fuzzy_logistic_2d import fit_forest_logistic as original_fit
from certified_frfs.centered_forest_fuzzy_logistic_batched import fit_forest_logistic as batched_fit
from certified_frfs.triangular_classifier import triangular_memberships
from certified_frfs.forest_fuzzy_classifier import forest_scores


@pytest.mark.parametrize('q', [2, 3])
def test_batched_witness_constraints_preserve_objective_scores_and_guards(q):
    rng = np.random.default_rng(200+q); X = rng.random((27, 3))
    W = np.zeros((q, 3, 3)); b = np.zeros(q); edges = [(0, 1), (1, 2)]
    if q == 2:
        y = (X[:, 0] > .5).astype(int); W[1, 0] = [-2, 0, 2]
    else:
        y = triangular_memberships(X[:, 0]).argmax(axis=1); W[:, 0] = 2*np.eye(3)
    fits = [f(X, y, W, b, np.arange(q), list(range(3)), edges, C=.1, pair_penalty=3.)
            for f in (original_fit, batched_fit)]
    assert all(f['feasible'] for f in fits)
    np.testing.assert_allclose(fits[0]['objective'], fits[1]['objective'], rtol=1e-6, atol=1e-7)
    scores = [forest_scores(X, f['coef'], f['intercept'], f['edges'], f['pair_coef']) for f in fits]
    np.testing.assert_allclose(scores[0], scores[1], atol=1e-4)
    for fit in fits:
        original = np.array(fit['original_margins']); mask = original > 1e-10
        assert mask.any() and np.all(np.array(fit['deployed_margins'])[mask] >= .5*original[mask]-1e-8)
    assert len(fits[1]['trace']) > 1
