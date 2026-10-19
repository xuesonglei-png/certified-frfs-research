import numpy as np
from certified_frfs.vote_objective import VoteMassObjective
from certified_frfs.vote_objective import soft_neighbor_inclusion


def test_vote_objective_has_crossfit_disjoint_splits_and_gradient_shape():
    rng = np.random.default_rng(2001); y = np.repeat([0, 1], 12); X = rng.uniform(0, .1, (24, 4))+.7*y[:, None]
    objective = VoteMassObjective(X, y, top_k=5)
    value, gradient = objective(np.ones(4)/2)
    assert np.isfinite(value) and gradient.shape == (4,)
    assert all(not set(s['reference']) & set(s['query']) for s in objective.splits)


def test_soft_inclusion_fixed_mass_ties_and_all_reference_case():
    distance = np.array([[1., 1., 1., 1., 1., 1.], [0., .1, .2, .3, .4, .5]])
    s = soft_neighbor_inclusion(distance, k=5)
    np.testing.assert_allclose(s.sum(axis=1), 5., atol=1e-12)
    np.testing.assert_allclose(s[0], np.ones(6)*5/6, atol=1e-12)
    np.testing.assert_array_equal(soft_neighbor_inclusion(distance, k=6), np.ones_like(distance))


def test_implicit_vote_gradient_matches_finite_differences():
    rng = np.random.default_rng(2002); X = rng.uniform(0, 1, (30, 4)); y = np.repeat([0, 1, 2], 10)
    objective = VoteMassObjective(X, y); w = np.array([.1, .2, .3, .4]); _, gradient = objective(w); numeric = []
    for a in range(4):
        step = np.eye(4)[a]*1e-6
        numeric.append((objective(w+step)[0]-objective(w-step)[0])/2e-6)
    np.testing.assert_allclose(gradient, numeric, atol=1e-7, rtol=1e-6)
