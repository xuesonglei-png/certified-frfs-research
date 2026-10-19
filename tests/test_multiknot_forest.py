import itertools
import numpy as np
import pytest
from certified_frfs.multiknot_forest import memberships, scores, box_margins, fit_multiknot_forest
from certified_frfs.triangular_classifier import triangular_memberships
from certified_frfs.orthogonal_forest_logistic import fit_orthogonal_forest, orthogonal_scores
from certified_frfs.forest_fuzzy_classifier import forest_box_margins


def test_three_knots_match_legacy_training_scores_and_box_margins():
    X = np.random.default_rng(73).uniform(size=(30, 3))
    y = np.arange(30) % 3
    old = fit_orthogonal_forest(X, y, [0, 1, 2], [[0, 1], [1, 2]])
    new = fit_multiknot_forest(X, y, [0, 1, 2], [[0, 1], [1, 2]], knot_count=3)
    np.testing.assert_allclose(memberships(X, [0, .5, 1]), triangular_memberships(X), atol=1e-15)
    np.testing.assert_allclose(scores(X, new), orthogonal_scores(X, old), atol=1e-9)
    expected = forest_box_margins(X, y, old['coef'], old['intercept'], old['classes'],
                                 old['edges'], np.array(old['pair_coef']), eps=.16)
    np.testing.assert_allclose(box_margins(X, y, new, eps=.16), expected, atol=1e-9)


def test_general_nonuniform_knots_exact_against_enumeration_with_shared_coordinates():
    rng = np.random.default_rng(81)
    knots = [0., .2, .5, .8, 1.]
    fit = {'knots': knots, 'classes': [0, 1, 2], 'edges': [[0, 1], [1, 2]],
           'coef': rng.normal(size=(3, 3, 5)), 'pair_coef': rng.normal(size=(3, 2, 5, 5)),
           'intercept': rng.normal(size=3)}
    X, y, eps = rng.uniform(size=(4, 3)), np.arange(4) % 3, .3
    output = box_margins(X, y, fit, eps=eps, witnesses=True)
    for i, x in enumerate(X):
        lo, hi = np.maximum(0, x-eps), np.minimum(1, x+eps)
        vertices = [[lo[a], hi[a]]+[k for k in knots if lo[a] <= k <= hi[a]] for a in range(3)]
        grid = np.array(list(itertools.product(*vertices)))
        value = scores(grid, fit)
        for r in range(3):
            if r != y[i]:
                np.testing.assert_allclose(output['pair_margins'][i, r], np.min(value[:, y[i]]-value[:, r]), atol=1e-12)
    witness = scores(output['witness_raw'], fit)
    np.testing.assert_allclose(witness[np.arange(4), y]-witness[np.arange(4), output['rivals']], output['margins'], atol=1e-12)


def test_refined_score_family_contains_three_knot_unary_and_pair_functions():
    rng = np.random.default_rng(53)
    W, V = rng.normal(size=(2, 3, 3)), rng.normal(size=(2, 1, 3, 3))
    transform = memberships(np.linspace(0, 1, 5), [0, .5, 1])
    original = {'knots': [0, .5, 1], 'classes': [0, 1], 'edges': [[0, 2]],
                'coef': W, 'pair_coef': V, 'intercept': [1., -1.]}
    refined = {**original, 'knots': np.linspace(0, 1, 5), 'coef': W@transform.T,
               'pair_coef': np.einsum('ak,qekl,bl->qeab', transform, V, transform)}
    X = rng.uniform(size=(20, 3))
    np.testing.assert_allclose(scores(X, original), scores(X, refined), atol=1e-12)
    y = np.arange(20) % 2
    np.testing.assert_allclose(box_margins(X, y, original, eps=.19), box_margins(X, y, refined, eps=.19), atol=1e-12)


def test_partition_of_unity_and_reject_cyclic_graph():
    X = np.array([[0., .01, .2, .7, 1.]])
    phi = memberships(X, [0., .1, .4, 1.])
    np.testing.assert_allclose(phi.sum(axis=-1), 1.)
    assert np.all(phi >= 0.)
    with pytest.raises(ValueError, match='forest'):
        fit_multiknot_forest(np.tile(X[:, :3], (6, 1)), np.arange(6)%2,
                             [0, 1, 2], [[0, 1], [0, 2], [1, 2]])
