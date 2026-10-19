import numpy as np
import pytest
from scipy.special import logsumexp
from certified_frfs.orthogonal_forest_logistic import fit_orthogonal_forest, orthogonal_scores, CONTRAST
from certified_frfs.forest_teacher_guard_ablation import fit_forest_logistic
from certified_frfs.forest_fuzzy_classifier import forest_scores
from certified_frfs.triangular_classifier import triangular_memberships


@pytest.mark.parametrize('q', [2, 3])
def test_orthonormal_pair_basis_matches_centered_convex_objective(q):
    rng = np.random.default_rng(840+q); X = rng.uniform(size=(45, 3))
    y = ((X[:, 0] > .5) != (X[:, 1] > .5)).astype(int) if q == 2 else triangular_memberships(X[:, 0]).argmax(axis=1)
    args = (X, y, np.zeros((q, 3, 3)), np.zeros(q), np.arange(q), [0, 1, 2], [(0, 1), (1, 2)])
    convex = fit_forest_logistic(*args, C=.3, pair_penalty=3., enforce_teacher_guards=False)
    fit = fit_orthogonal_forest(X, y, [0, 1, 2], [(0, 1), (1, 2)], C=.3, pair_penalty=3.)
    S = orthogonal_scores(X, fit); W, V = np.array(fit['coef']), np.array(fit['pair_coef'])
    own = np.argmax(y[:, None] == np.arange(q), axis=1)
    w = np.array([len(y)/(q*sum(y == label)) for label in y])
    main = W[1] if q == 2 else W-W.mean(axis=0, keepdims=True)
    pair = V[1] if q == 2 else V-V.mean(axis=0, keepdims=True)
    objective = w@(logsumexp(S, axis=1)-S[np.arange(len(X)), own])+(np.sum(main**2)+3*np.sum(pair**2))/(2*.3)
    np.testing.assert_allclose(objective, convex['objective'], rtol=1e-6, atol=1e-7)
    target = forest_scores(X, convex['coef'], convex['intercept'], convex['edges'], convex['pair_coef'])
    np.testing.assert_allclose(S-S.mean(axis=1, keepdims=True), target-target.mean(axis=1, keepdims=True), atol=1e-4)
    np.testing.assert_allclose(V.sum(axis=2), 0, atol=1e-12)
    np.testing.assert_allclose(V.sum(axis=3), 0, atol=1e-12)
    np.testing.assert_allclose(CONTRAST.T@CONTRAST, np.eye(2), atol=1e-12)
    assert fit['optimizer_converged']


def test_no_edges_and_empty_support_produce_well_defined_scores():
    X = np.array([[.1, .2], [.9, .8]]*3); y = np.array([0, 1]*3)
    for fs in ([0], []):
        fit = fit_orthogonal_forest(X, y, fs, [[0, 1]])
        assert orthogonal_scores(X, fit).shape == (6, 2)
        assert fit['edges'] == [] and fit['teacher_guards_enforced'] is False
