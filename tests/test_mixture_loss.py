import itertools
import numpy as np
from certified_frfs.core import gamma,interval_bounds
from certified_frfs.mixture_loss import mixture_loss_bound,mixture_signed_bound
from certified_frfs.shared_loss import shared_loss_bound


def test_mixture_bound_dominates_grid_losses_and_preserves_old_bound():
    X=np.array([[.2,.3],[.35,.7],[.65,.4],[.8,.8]])
    y=np.array([0,0,1,1]);w=np.array([.65,.45]);eps=.08;fs=[0]
    bound=mixture_loss_bound(X,y,fs,w,eps=eps)
    assert bound['upper_bound']<=shared_loss_bound(X,y,fs,w,eps=eps)['upper_bound']
    lo,hi=interval_bounds(X,eps)
    for bits in itertools.product([0,.5,1],repeat=X.size):
        Z=lo+np.array(bits).reshape(X.shape)*(hi-lo)
        assert gamma(Z,y,weights=w)-gamma(Z,y,fs,weights=w)<=bound['upper_bound']+1e-10


def test_zero_radius_with_all_witnesses_is_empirical_loss():
    rng=np.random.default_rng(904);X=rng.uniform(.1,.9,(8,3));y=np.repeat([0,1],4);w=np.array([.6,.3,.2])
    bound=mixture_loss_bound(X,y,[0,2],w,eps=0,witness_pool=None)
    assert abs(bound['upper_bound']-(gamma(X,y,weights=w)-gamma(X,y,[0,2],weights=w)))<1e-7


def test_same_object_cancellation_and_heterogeneous_clipping():
    X=np.array([[.01,.95],[.1,.8],[.9,.1],[.99,.05]])
    lo,hi=interval_bounds(X,np.array([.04,.08]));w=np.array([.7,.3])
    bound,success=mixture_signed_bound(lo,hi,0,2,np.array([2]),[0,1],w)
    assert success and abs(bound)<1e-7
    assert mixture_loss_bound(X,[0,0,1,1],[0,1],w)['upper_bound']==0
