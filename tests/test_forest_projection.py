import itertools
import numpy as np
import pytest
from certified_frfs.forest_projection import project_forest_parameters, forest_projection_certificate
from certified_frfs.forest_fuzzy_classifier import forest_scores, forest_box_margins


def test_omission_exactly_absorbs_pairs_into_unary_terms_and_constants():
    rng = np.random.default_rng(74)
    X = rng.uniform(size=(11, 5)); center = rng.uniform(size=5)
    W = rng.normal(size=(3, 5, 3)); b = rng.normal(size=3)
    edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
    V = rng.normal(size=(3, 4, 3, 3))
    # Every support exercises the retained, unary and constant cases.
    for mask in itertools.product((False, True), repeat=5):
        fs = list(np.flatnonzero(mask)); p = project_forest_parameters(W, b, edges, V, fs, center)
        replaced = X.copy(); replaced[:, ~np.array(mask)] = center[~np.array(mask)]
        expected = forest_scores(replaced, W, b, edges, V)
        observed = forest_scores(X, p['coef'], p['intercept'], p['edges'], p['pair_coef'])
        np.testing.assert_allclose(expected, observed, atol=1e-12)
        assert all(u in fs and v in fs for u, v in p['edges'])


def test_projected_box_margin_attains_shared_raw_vertex_minimum():
    rng = np.random.default_rng(98); X = np.array([[.48, .52, .5]])
    W = rng.normal(size=(2, 3, 3)); b = np.array([0., 12.])
    edges = [(0, 1), (1, 2)]; V = rng.normal(size=(2, 2, 3, 3))
    p = project_forest_parameters(W, b, edges, V, [0, 2], [.5]*3)
    margin = forest_box_margins(X, [1], p['coef'], p['intercept'], [0, 1], p['edges'], p['pair_coef'], eps=.05)[0]
    vertices = np.array([[u, .52, v] for u in (.43, .5, .53) for v in (.45, .5, .55)])
    scores = forest_scores(vertices, p['coef'], p['intercept'], p['edges'], p['pair_coef'])
    np.testing.assert_allclose(margin, np.min(scores[:, 1]-scores[:, 0]), atol=1e-12)
    cert = forest_projection_certificate(X, [1], W, b, [0, 1], edges, V, [0, 1, 2], [.5]*3)
    assert cert['feasible'] and cert['protected_count'] == 1
    np.testing.assert_allclose(cert['original_margins'], cert['reduced_margins'], atol=1e-12)


def test_projection_rejects_cycles_even_when_all_features_omitted():
    with pytest.raises(ValueError):
        project_forest_parameters(np.zeros((2, 3, 3)), np.zeros(2), [(0, 1), (0, 2), (1, 2)],
                                  np.zeros((2, 3, 3, 3)), [], [.5]*3)
