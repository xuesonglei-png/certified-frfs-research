import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative
from certified_frfs.metric_transport import fit_transport_weights,transport_objective
from certified_frfs.redistributed_loss import redistributed_loss_bound


def test_quadratic_gradient_matches_finite_difference():
    rng=np.random.default_rng(961);A=rng.uniform(0,1,(32,4));t=rng.uniform(0,1,32);v=np.ones(4);initial=v*.8
    _,gradient=transport_objective(v,A,t,initial,.001)
    numerical=approx_derivative(lambda x:transport_objective(x,A,t,initial,.001)[0],v).ravel()
    np.testing.assert_allclose(gradient,numerical,rtol=1e-6,atol=1e-7)


def test_transport_preserves_mass_improves_objective_and_is_reproducible():
    rng=np.random.default_rng(962);X=rng.uniform(0,1,(50,6));w=rng.uniform(0,1,6)
    a,info=fit_transport_weights(X,[0,2,4],w,max_pairs=64)
    b,other=fit_transport_weights(X,[0,2,4],w,max_pairs=64)
    assert np.all(a>=0) and abs(a.sum()-w.sum())<1e-12
    assert info['regularized_objective_after']<=info['regularized_objective_before']+1e-12
    np.testing.assert_array_equal(a,b);assert info==other
    assert len({tuple(p) for p in info['pair_indices']})==64


def test_identical_duplicate_has_exact_reconstruction_and_small_loss_bound():
    t=np.r_[np.linspace(.05,.15,8),np.linspace(.85,.95,8)];X=np.column_stack([t,t]);w=np.ones(2)/np.sqrt(2)
    v,info=fit_transport_weights(X,[0],w)
    assert info['training_mse_after']<1e-20
    assert redistributed_loss_bound(X,np.repeat([0,1],8),[0],w,v)['upper_bound']<.01


def test_pair_budget_and_empty_inputs_rejected():
    X=np.ones((4,2));w=np.ones(2)
    for kwargs in [{'max_pairs':0},{'max_pairs':1.5},{'reg':0}]:
        with pytest.raises(ValueError):fit_transport_weights(X,[0],w,**kwargs)
    with pytest.raises(ValueError):fit_transport_weights(X,[],w)
    v,info=fit_transport_weights(X,[0],np.zeros(2));assert v[0]==0 and info['status']=='degenerate'
