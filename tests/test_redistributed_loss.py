import itertools
import numpy as np
import pytest
from certified_frfs.core import gamma,interval_bounds
from certified_frfs.redistributed_loss import redistributed_loss_bound,transfer_weights


def test_duplicate_attribute_transfer_has_zero_nominal_and_small_box_loss():
    t=np.r_[np.linspace(.05,.15,8),np.linspace(.85,.95,8)]
    X=np.column_stack([t,t]);y=np.repeat([0,1],8);wf=np.ones(2)/np.sqrt(2)
    wr=transfer_weights(X,[0],wf)
    assert np.allclose(wr,[wf.sum()])
    bound=redistributed_loss_bound(X,y,[0],wf,wr,eps=.01)
    assert bound['upper_bound']<.01
    assert redistributed_loss_bound(X,y,[0],wf,wr,eps=0)['upper_bound']<=1e-10


def test_bound_covers_shared_box_grid_for_arbitrary_transfer():
    X=np.array([[.2,.3,.8],[.35,.7,.4],[.65,.4,.2],[.8,.8,.1]])
    y=np.array([0,0,1,1]);wf=np.array([.5,.3,.2]);wr=np.array([.7,.1])
    bound=redistributed_loss_bound(X,y,[0,2],wf,wr,eps=.08)
    lo,hi=interval_bounds(X,.08);wide=np.zeros(3);wide[[0,2]]=wr
    for bits in itertools.product([0,1],repeat=X.size):
        Z=lo+np.array(bits).reshape(X.shape)*(hi-lo)
        loss=gamma(Z,y,weights=wf)-gamma(Z,y,[0,2],weights=wide)
        assert loss<=bound['upper_bound']+1e-10


def test_empty_and_full_transfer_validation():
    X=np.arange(12,dtype=float).reshape(4,3)/12;wf=np.ones(3)
    assert np.allclose(transfer_weights(X,[0,1,2],wf),wf)
    for fs in ([],[0,0]):
        try:transfer_weights(X,fs,wf);raise AssertionError
        except ValueError:pass


def test_changed_full_metric_is_not_automatically_zero_and_transfer_preserves_mass():
    X=np.array([[.1,.2],[.2,.3],[.8,.9],[.9,.8]]);y=[0,0,1,1];wf=np.array([.6,.4])
    bound=redistributed_loss_bound(X,y,[0,1],wf,[.1,.1],eps=0,witness_pool=None)
    assert bound['upper_bound']>.3
    assert np.isclose(transfer_weights(X,[1],wf).sum(),wf.sum())
    assert redistributed_loss_bound(X,[0]*4,[0],wf,[1.])['upper_bound']==0
    with pytest.raises(ValueError):redistributed_loss_bound(X,y,[0],wf,[1.],eps=-.1)


def test_random_heterogeneous_boxes_and_zero_radius_validity():
    rng=np.random.default_rng(921)
    for _ in range(8):
        X=rng.uniform(0,1,(6,3));y=np.repeat([0,1],3);wf=rng.uniform(0,1,3);wr=rng.uniform(0,1,2)
        radius=rng.uniform(0,.08,X.shape);lo,hi=interval_bounds(X,radius)
        bound=redistributed_loss_bound(X,y,[0,2],wf,wr,eps=radius,witness_pool=None)
        wide=np.zeros(3);wide[[0,2]]=wr
        for __ in range(30):
            Z=rng.uniform(lo,hi);loss=gamma(Z,y,weights=wf)-gamma(Z,y,[0,2],weights=wide)
            assert loss<=bound['upper_bound']+1e-10
        exact=redistributed_loss_bound(X,y,[0,2],wf,wr,eps=0,witness_pool=None)
        assert exact['upper_bound']+1e-10>=max(0,gamma(X,y,weights=wf)-gamma(X,y,[0,2],weights=wide))
