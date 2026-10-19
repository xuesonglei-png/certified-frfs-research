from fractions import Fraction as F
import numpy as np
from crossgroup_soft5_terms import assignments
from latent_friend_rank_hinge import LatentFriendRankHinge
from capped_third_enemy_hinge import CappedThirdEnemyHinge
from label_equivariant_vote import predict


def test_exact_mechanism_fixed_triple_is_vacuous_but_adaptive_triple_is_zero():
    y=np.repeat([0,1],12);group=assignments(y)
    same=np.flatnonzero((y==0)&(group!=group[0]));enemy=np.flatnonzero((y!=0)&(group!=group[0]))
    A=np.full((24,2),.9);A[0]=0.;A[same[0]]=[.01,.1];A[same[1:3]]=[.2,0.]
    A[same[3:5]]=[0.,.3];A[enemy]=[.12,.12];anchor=np.ones(2)/2;w=np.array([1.,0.])
    fixed=CappedThirdEnemyHinge(A,y,anchor);latent=LatentFriendRankHinge(A,y,anchor)
    assert fixed.rows[0]['friends']==same[:3].tolist() and latent.rows[0]['friends']==same[:5].tolist()
    assert fixed.losses(w)[0]==1. and latent.losses(w)[0]==0.
    refs=np.asarray(latent.rows[0]['references']);prediction=predict((abs(A[refs]-A[0])@w)[None,:],y[refs])
    assert prediction['predictions']==[0]
    # Rational distance ranges prove correctness for EVERY shared table,
    # beyond the nominal floating-point observation. Three alternative
    # friend distances are <=.02 and ALL enemy distances are >=.10.
    query=(F(0),F(1,100));near_friend_intervals=[(F(0),F(2,100)),(F(0),F(1,100)),(F(0),F(1,100))]
    enemy_interval=(F(11,100),F(13,100))
    far=lambda interval:max(abs(a-b) for a in query for b in interval)
    minimum_enemy=max(F(0),enemy_interval[0]-query[1],query[0]-enemy_interval[1])
    assert max(far(interval) for interval in near_friend_intervals)==F(2,100)
    assert minimum_enemy==F(1,10)
    assert all(far(interval)<minimum_enemy for interval in near_friend_intervals)
