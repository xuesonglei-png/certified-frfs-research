import numpy as np
from scipy.spatial.distance import cdist
from certified_frfs.flexible_witness import flexible_bounds
from certified_frfs.adaptive_witness import adaptive_bounds


def test_three_friend_pool_matches_refreshed_bound():
    rng=np.random.default_rng(41);X=rng.uniform(.1,.9,(24,4));y=np.repeat([0,1,2],8);w=np.ones(4)/2
    np.testing.assert_allclose(flexible_bounds(X,y,[0,2,3],w,pool_size=3),adaptive_bounds(X,y,[0,2,3],w))


def test_larger_pools_dominate_three_friends():
    rng=np.random.default_rng(42);X=rng.uniform(.1,.9,(24,4));y=np.repeat([0,1,2],8);w=np.ones(4)/2
    small=flexible_bounds(X,y,range(4),w,pool_size=3)
    big=flexible_bounds(X,y,range(4),w,pool_size=None)
    assert np.all(big>=small-1e-12)


def test_zero_radius_and_shared_perturbation_majorities():
    rng=np.random.default_rng(43);y=np.repeat([0,1,2],8)
    X=np.column_stack([.1+.4*y,rng.uniform(.1,.9,(24,2))]);w=np.array([1.,.1,.1])
    for eps in (0.,.01,.05):
        mask=flexible_bounds(X,y,range(3),w,eps=eps,pool_size=None)>1e-10
        assert mask.any()
        for _ in range(100):
            Z=np.clip(X+rng.uniform(-eps,eps,X.shape),0,1);D=cdist(Z*w,Z*w,'cityblock');np.fill_diagonal(D,np.inf)
            nn=np.argsort(D,axis=1,kind='stable')[:,:5]
            assert np.all(np.sum(y[nn]==y[:,None],axis=1)[mask]>=3)


def test_all_friends_zero_radius_matches_strict_majority_without_ties():
    rng=np.random.default_rng(44);X=rng.uniform(.1,.9,(36,5));y=np.repeat([0,1,2],12);w=np.array([.1,.2,.3,.4,.5])
    bounds=flexible_bounds(X,y,range(5),w,eps=0,pool_size=None)
    D=cdist(X*w,X*w,'cityblock');np.fill_diagonal(D,np.inf)
    nn=np.argsort(D,axis=1,kind='stable')[:,:5]
    np.testing.assert_array_equal(bounds>1e-10,np.sum(y[nn]==y[:,None],axis=1)>=3)
