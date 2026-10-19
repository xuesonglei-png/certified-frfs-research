import itertools
import numpy as np
from certified_frfs.shared_loss import shared_loss_bound
from certified_frfs.core import gamma,certificate


def test_zero_radius_matches_exact_empirical_loss():
    rng=np.random.default_rng(209);X=rng.uniform(.1,.9,(12,4));y=np.repeat([0,1,2],4);w=np.array([.2,.4,.6,.8])
    result=shared_loss_bound(X,y,[1,3],w,eps=0,witness_pool=None)
    expected=gamma(X,y,weights=w)-gamma(X,y,[1,3],weights=w)
    assert abs(result['raw_shared_anchor_bound']-expected)<1e-12


def test_enumerated_shared_box_realizations_obey_bound():
    X=np.array([[.1,.3],[.2,.8],[.7,.2],[.8,.9]]);y=np.array([0,0,1,1]);w=np.array([.7,.4]);eps=.03
    result=shared_loss_bound(X,y,[0],w,eps=eps,witness_pool=None)
    for offsets in itertools.product([-eps,0,eps],repeat=8):
        Z=X+np.array(offsets).reshape(X.shape)
        assert gamma(Z,y,weights=w)-gamma(Z,y,[0],weights=w)<=result['upper_bound']+1e-12


def test_more_witnesses_do_not_weaken_bound_and_existing_bound_preserved():
    rng=np.random.default_rng(210);X=rng.uniform(.1,.9,(18,5));y=np.repeat([0,1,2],6);w=np.ones(5)/np.sqrt(5)
    a=shared_loss_bound(X,y,[0,2],w,witness_pool=1);b=shared_loss_bound(X,y,[0,2],w,witness_pool=None)
    assert b['shared_anchor_bound']<=a['shared_anchor_bound']+1e-12
    assert b['upper_bound']<=certificate(X,y,[0,2],weights=w,eps=.01)['upper_bound']


def test_clipped_heterogeneous_boxes_with_zero_weights_and_saturation():
    rng=np.random.default_rng(211);X=rng.uniform(0,1,(10,3));X[0]=[0,1,0];X[1]=[1,0,1]
    y=np.repeat([0,1],5);w=np.array([2.,0.,1.5]);radius=rng.uniform(0,.12,X.shape)
    result=shared_loss_bound(X,y,[0,1],w,eps=radius,witness_pool=None)
    for _ in range(200):
        Z=np.clip(X+rng.uniform(-1,1,X.shape)*radius,0,1)
        assert gamma(Z,y,weights=w)-gamma(Z,y,[0,1],weights=w)<=result['upper_bound']+1e-12


def test_zero_radius_distance_ties_and_empty_subset():
    X=np.array([[.1,.2],[.1,.2],[.6,.7],[.6,.7]]);y=np.array([0,0,1,1]);w=np.array([.3,.4])
    for fs in ([],[0]):
        result=shared_loss_bound(X,y,fs,w,eps=0,witness_pool=None)
        assert abs(result['raw_shared_anchor_bound']-(gamma(X,y,weights=w)-gamma(X,y,fs,weights=w)))<1e-12
