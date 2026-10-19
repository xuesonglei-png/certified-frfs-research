import numpy as np
from scipy.optimize._numdiff import approx_derivative
from certified_frfs.sparse_majority import MajorityMarginObjective, sparse_majority_candidates


def test_majority_gradient_away_from_order_statistic_ties():
    rng = np.random.default_rng(991); X = rng.uniform(.1, .9, (16, 4)); y = np.repeat([0, 1], 8)
    objective = MajorityMarginObjective(X, y, ridge=.1); w = np.array([.41, .29, .19, .11])
    np.testing.assert_allclose(objective(w)[1], approx_derivative(lambda v: objective(v)[0], w).ravel(), atol=1e-6, rtol=1e-5)


def test_majority_sparse_optimizer_preserves_constraints_and_descent():
    rng = np.random.default_rng(992); X = rng.uniform(.1, .9, (16, 4)); y = np.repeat([0, 1], 8); X[:, 0] = .1+.8*y
    for r in sparse_majority_candidates(X, y, ridge=.01):
        w = np.array(r['weights']); trace = r['fit']['trace']
        assert abs(w.sum()-1) < 1e-12 and (w >= 0).all()
        assert len(r['features']) <= r['fit']['requested_cardinality']
        assert all(b['objective'] <= a['objective']+1e-10 for a, b in zip(trace, trace[1:]))


def test_positive_third_enemy_contrast_implies_sampled_five_nn_majority():
    rng = np.random.default_rng(993); X = rng.uniform(.1, .9, (16, 3)); y = np.repeat([0, 1], 8)
    X[:, 0] = .1+.8*y; objective = MajorityMarginObjective(X, y, ridge=.01)
    w = np.array([.8, .1, .1]); ws = objective.workspace
    T = np.einsum('d,dijf->ijf', w, ws.terms); guaranteed = []
    for i in objective.ids:
        values = T[i, ws.cross[i]].min(axis=1)
        if np.sort(values)[2] > 1e-10: guaranteed.append(i)
    assert guaranteed
    for _ in range(20):
        Z = np.clip(X+rng.uniform(-.01, .01, X.shape), 0, 1)
        D = abs(Z[:, None, :]-Z[None, :, :])@w; np.fill_diagonal(D, np.inf)
        nn = np.argsort(D, axis=1, kind='stable')[:, :5]
        for i in guaranteed: assert (y[nn[i]] == y[i]).sum() >= 3
