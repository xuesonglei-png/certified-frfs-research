import numpy as np
from scipy.optimize._numdiff import approx_derivative
from certified_frfs.class_transport import fit_class_transport,class_transport_objective
from certified_frfs.softvote_metric import softvote_terms


def test_combined_gradient_matches_finite_difference():
    rng=np.random.default_rng(970);X=rng.uniform(.1,.9,(12,3));y=np.repeat([0,1],6)
    A=rng.uniform(0,1,(20,3));target=rng.uniform(0,1,20);v=np.array([.4,.3,.3]);initial=np.ones(3)/3
    terms=softvote_terms(X,y);f=lambda q:class_transport_objective(q,A,target,initial,terms)
    np.testing.assert_allclose(f(v)[1],approx_derivative(lambda q:f(q)[0],v).ravel(),rtol=1e-5,atol=1e-6)


def test_local_fit_preserves_mass_and_never_worsens_saved_objective():
    rng=np.random.default_rng(971);y=np.repeat([0,1],12);X=rng.uniform(.1,.9,(24,4));X[:,0]=.1+.8*y
    w=np.ones(4)/2;v,info=fit_class_transport(X,y,[0,1],w)
    assert np.all(v>0) and abs(v.sum()-w.sum())<1e-12
    assert info['objective_after']<=info['objective_before']+1e-12
