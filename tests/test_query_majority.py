import numpy as np
from scipy.optimize._numdiff import approx_derivative
from certified_frfs.query_majority import QueryMajorityObjective, query_majority_candidates


def test_queries_disjoint_from_their_reference_and_gradient_correct():
    rng = np.random.default_rng(1014); X = rng.uniform(.1, .9, (24, 4)); y = np.repeat([0, 1], 12)
    objective = QueryMajorityObjective(X, y); seen = []
    for split in objective.splits:
        assert set(split['reference']).isdisjoint(split['query'])
        assert set(split['reference']) | set(split['query']) == set(range(len(y)))
        seen.extend(split['query'])
    assert sorted(seen) == list(range(len(y))) and objective.applicable_count == len(y)
    w = np.array([.43, .29, .19, .09])
    np.testing.assert_allclose(objective(w)[1], approx_derivative(lambda v: objective(v)[0], w).ravel(), atol=1e-6, rtol=1e-5)


def test_query_joint_optimizer_descent_and_cardinality():
    rng = np.random.default_rng(1015); X = rng.uniform(.1, .9, (24, 4)); y = np.repeat([0, 1], 12); X[:, 0] = .1+.8*y
    for c in query_majority_candidates(X, y):
        trace = c['fit']['trace']; w = np.array(c['weights'])
        assert len(c['features']) <= c['fit']['requested_cardinality'] and abs(w.sum()-1) < 1e-12
        assert all(b['objective'] <= a['objective']+1e-10 for a, b in zip(trace, trace[1:]))


def test_query_refit_cannot_add_attributes_and_improves_its_objective():
    from certified_frfs.query_majority import refit_query_support
    rng = np.random.default_rng(1016); X = rng.uniform(.1, .9, (24, 4)); y = np.repeat([0, 1], 12)
    candidate = {'features': [0, 2], 'weights': [.6, 0., .4, 0.], 'fit': {'initial': True}}
    objective = QueryMajorityObjective(X, y)
    fitted = refit_query_support(X, y, candidate, objective=objective)
    assert set(fitted['features']) <= set(candidate['features'])
    assert objective(np.array(fitted['weights']))[0] <= objective(np.array(candidate['weights']))[0]+1e-12


def test_singletons_remain_reference_only():
    rng=np.random.default_rng(1018);X=rng.uniform(0,1,(25,4));y=np.r_[np.repeat([0,1],12),2]
    objective=QueryMajorityObjective(X,y)
    for split in objective.splits:
        assert 24 in split['reference'] and 24 not in split['query']
