import itertools
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from certified_frfs.sparse_neighbor import sparse_simplex_projection, sparse_neighbor_objective, sparse_metric_candidates
from certified_frfs.softvote_metric import softvote_terms


def test_sparse_projection_matches_exhaustive_support_projection():
    rng = np.random.default_rng(987)
    for _ in range(12):
        z = rng.normal(size=5); k = 3; q = sparse_simplex_projection(z, k)
        best = np.inf
        for ids in itertools.combinations(range(5), k):
            v = np.zeros(5); v[list(ids)] = sparse_simplex_projection(z[list(ids)], k)
            best = min(best, np.sum((v-z)**2))
        assert abs(np.sum((q-z)**2)-best) < 1e-10
        assert np.count_nonzero(q) <= k and abs(q.sum()-1) < 1e-12


def test_sparse_objective_gradient_including_zero_coordinates():
    rng = np.random.default_rng(988); X = rng.uniform(0, 1, (16, 4)); y = np.repeat([0, 1], 8)
    terms = softvote_terms(X, y); w = np.array([.7, .3, 0., 0.])
    f = lambda v: sparse_neighbor_objective(v, terms)
    np.testing.assert_allclose(f(w)[1], approx_derivative(lambda v: f(v)[0], w).ravel(), atol=1e-6, rtol=1e-5)


def test_joint_optimization_descends_and_respects_support_budget():
    rng = np.random.default_rng(989); X = rng.uniform(0, 1, (24, 6)); y = np.repeat([0, 1], 12)
    X[:, 0] = .1+.8*y
    for r in sparse_metric_candidates(X, y):
        w = np.asarray(r['weights']); values = [q['objective'] for q in r['fit']['trace']]
        assert all(b <= a+1e-10 for a, b in zip(values, values[1:]))
        assert len(r['features']) <= r['fit']['requested_cardinality']
        assert abs(w.sum()-1) < 1e-12
