import numpy as np
from scipy.optimize._numdiff import approx_derivative
from certified_frfs.query_probability import QueryProbabilityObjective,refit_probability_support


def test_probability_gradient_and_disjoint_query_coverage():
    rng=np.random.default_rng(1023);X=rng.uniform(0,1,(25,4));y=np.r_[np.repeat([0,1],12),2]
    objective=QueryProbabilityObjective(X,y);w=np.array([.5,.3,.2,0.]);seen=[]
    for split in objective.splits:
        assert set(split['reference']).isdisjoint(split['query']);assert 24 in split['reference'];seen.extend(split['query'])
    assert sorted(seen)==list(range(24))
    np.testing.assert_allclose(objective(w)[1],approx_derivative(lambda v:objective(v)[0],w).ravel(),rtol=1e-5,atol=1e-6)


def test_probability_refit_descends_without_adding_attributes():
    rng=np.random.default_rng(1024);X=rng.uniform(0,1,(24,4));y=np.repeat([0,1],12)
    candidate={'features':[0,2],'weights':[.6,0.,.4,0.],'fit':{}}
    objective=QueryProbabilityObjective(X,y);fit=refit_probability_support(X,y,candidate,objective=objective)
    assert set(fit['features'])<=set(candidate['features'])
    assert objective(np.array(fit['weights']))[0]<=objective(np.array(candidate['weights']))[0]+1e-12
