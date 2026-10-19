import numpy as np
from expanded_majority_guard_v1 import expanded_bounds
from certified_frfs.majority_metric import MajorityTerms
from certified_frfs.relation_support_selector_v1 import weighted_knn

def test_expansion_dominates_fixed_three_and_never_certifies_wrong_loo():
    rng=np.random.default_rng(212)
    for trial in range(12):
        A=rng.random((18,3));y=np.repeat([0,1,2],6);w=np.array([.2,.3,.5])
        old=MajorityTerms(A,y,weights=w,eps=.01).bounds(range(3),allowed_enemies=2)
        small=expanded_bounds(A,y,w,n_friends=3);large=expanded_bounds(A,y,w,n_friends=15)
        np.testing.assert_allclose(small,old,atol=1e-12)
        assert np.all(large>=small-1e-12)
        for _ in range(5):
            Z=np.clip(A+rng.uniform(-.01,.01,A.shape),0,1)
            for i in np.flatnonzero(large>1e-10):
                keep=np.arange(len(A))!=i
                score=weighted_knn(Z[keep],y[keep],Z[i:i+1],y[i:i+1],list(range(3)),w)
                assert score['predictions']==[int(y[i])]

def test_strict_improvement_for_opposite_side_friend_witnesses():
    # The nominal closest three friends sit opposite the enemies. Their shared
    # query uncertainty makes them worse witnesses than the next three friends
    # on the enemy side. This example needs expanded witnesses at radius .01.
    A=np.array([.5,.54,.54,.54,.459,.459,.459,.43,.43,.43])[:,None]
    y=np.array([0]*7+[1]*3);w=np.array([1.])
    small=expanded_bounds(A,y,w,n_friends=3);large=expanded_bounds(A,y,w,n_friends=15)
    assert small[0]<0 and large[0]>0
