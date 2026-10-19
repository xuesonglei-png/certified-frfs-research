from itertools import combinations, product
import numpy as np
import pytest
from latent_friend_rank_hinge import LatentFriendRankHinge
from capped_third_enemy_hinge import CappedThirdEnemyHinge
from independent_majority_hinge import coefficient
from label_equivariant_vote import predict


def scalar_top(model, i, w, k, choice):
    row=model.rows[i]
    if row['constant_loss'] is not None:
        return row['constant_loss'] if k==3 else 0.
    triple=model.triples[i][choice]
    hinges=[max(0.,1-min(float(coefficient(model.A,i,e,row['friends'][f],model.eps)@w) for f in triple)/model.margin)
        for e in row['enemies']]
    return sum(sorted(hinges,reverse=True)[:k])


def test_latent_loss_dominates_original_and_scalar_triples_replay():
    rng=np.random.default_rng(348);A=rng.uniform(0,1,(24,3));y=np.repeat([0,1],12);anchor=np.ones(3)/3
    model=LatentFriendRankHinge(A,y,anchor);fixed=CappedThirdEnemyHinge(A,y,anchor);strict=False
    for _ in range(12):
        w=rng.dirichlet(np.ones(3));fresh=[]
        for i,row in enumerate(model.rows):
            value=row['constant_loss'] if row['constant_loss'] is not None else min(
                min(1.,scalar_top(model,i,w,3,j)-scalar_top(model,i,w,2,j)) for j in range(len(model.triples[i])))
            fresh.append(value)
        np.testing.assert_allclose(model.losses(w),fresh,atol=1e-10)
        assert np.all(model.losses(w)<=fixed.losses(w)+1e-10)
        strict |= bool(np.any(model.losses(w)<fixed.losses(w)-1e-6))
    assert strict


def test_selected_triple_support_is_tangent_global_majorizer_of_latent_loss():
    rng=np.random.default_rng(521);A=rng.uniform(0,1,(24,3));y=np.repeat([0,1],12);model=LatentFriendRankHinge(A,y,np.ones(3)/3)
    for _ in range(8):
        point=rng.dirichlet(np.ones(3));elsewhere=rng.dirichlet(np.ones(3));info=model.majorizer_at(point)
        choices=info['triple_choices'];top_at,_=model.top_sum(point,3,choices);top_elsewhere,_=model.top_sum(elsewhere,3,choices)
        tangent=[];away=[]
        for i,(face,shift) in enumerate(zip(info['minor_faces'],info['minor_cap_shifts'])):
            row=model.rows[i];gradient=np.zeros(3)
            for eid,friend in zip(face['enemy_indices'],face['worst_friend_indices']):
                if eid in face['active_enemy_indices']:
                    gradient-=coefficient(model.A,i,row['enemies'][eid],row['friends'][friend],.01)/.02
            np.testing.assert_allclose(gradient,face['gradient'],atol=1e-10)
            tangent.append(gradient@point+face['intercept']-shift)
            away.append(gradient@elsewhere+face['intercept']-shift)
            np.testing.assert_allclose(gradient@point+face['intercept'],scalar_top(model,i,point,face['k'],choices[i]),atol=1e-10)
        np.testing.assert_allclose(top_at-np.array(tangent),model.losses(point),atol=1e-10)
        assert np.all(top_elsewhere-np.array(away)>=model.losses(elsewhere)-1e-10)


def test_minimum_of_triple_bounds_controls_every_shared_corner_vote():
    nominal=np.array([.5,.43,.45,.48,.54,.57,.46,.52,.58,.72]);labels=np.r_[np.zeros(6),np.ones(4)]
    lo,hi=nominal-.01,nominal+.01;margin=.02;bounds=[]
    for triple in combinations(range(1,6),3):
        hinges=[]
        for enemy in range(6,10):
            contrasts=[min(max(0.,lo[enemy]-t,t-hi[enemy])-max(abs(t-lo[f]),abs(t-hi[f])) for t in (lo[0],hi[0])) for f in triple]
            hinges.append(max(0.,1-min(contrasts)/margin))
        bounds.append(min(1.,sorted(hinges,reverse=True)[2]))
    bound=min(bounds)
    for bits in product([0,1],repeat=10):
        Z=np.where(bits,hi,lo);error=float(predict(abs(Z[1:]-Z[0])[None,:],labels[1:])['predictions'][0]!=labels[0])
        assert error<=bound+1e-12


def test_undeclared_pool_sizes_are_rejected():
    A=np.linspace(0,1,18)[:,None];y=np.repeat([0,1],9)
    with pytest.raises(ValueError,match='fixed'):
        LatentFriendRankHinge(A,y,np.ones(1),pool_size=10)
