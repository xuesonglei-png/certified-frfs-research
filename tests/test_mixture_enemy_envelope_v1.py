import itertools
import numpy as np
import pytest
from mixture_enemy_envelope_v1 import coordinate_affine_rows,solve_mixture_enemy_bound


def test_shared_enemy_cancellation_and_identity_validation():
    lo=np.array([.1,.7]);hi=np.array([.3,.9]);li=np.array([.4,.3]);ui=np.array([.6,.5])
    r=solve_mixture_enemy_bound(li,ui,lo,hi,lo[None,:],hi[None,:],[.4,.6],[.4,.6],shared_index=0)
    assert r['upper_bound_raw']==pytest.approx(0.,abs=1e-14)
    with pytest.raises(ValueError):
        solve_mixture_enemy_bound(li,ui,lo,hi,lo[None,:]+.01,hi[None,:],[.4,.6],[.4,.6],shared_index=0)


def test_scalar_envelope_dominates_dense_shared_box_samples():
    rng=np.random.default_rng(883)
    for shared in (None,0):
        il,ih=.1,.9;jl,jh=.25,.65;kl=np.array([.25,.72]);kh=np.array([.65,.95])
        coefficient,constant=coordinate_affine_rows(il,ih,jl,jh,kl,kh,.8,.7,shared)
        for _ in range(20):
            lam=rng.dirichlet([1,1]);bound=np.max(coefficient@lam+constant)
            for t in np.linspace(il,ih,25):
                for j in np.linspace(jl,jh,11):
                    for k0,k1 in itertools.product((kl[0],kh[0]),(kl[1],kh[1])):
                        positions=np.array([k0,k1]);
                        if shared is not None:positions[shared]=j
                        value=.8*np.dot(lam,np.abs(t-positions))-.7*abs(t-j)
                        assert value<=bound+1e-12


def test_mixture_strictly_tightens_a_complementary_witness_example():
    # Shared query can move toward either opposite witness. A mixture balances
    # their far distances, whereas a single witness has a larger maximum.
    r=solve_mixture_enemy_bound([0.],[1.],[.5],[.5],np.array([[0.],[1.]]),
        np.array([[0.],[1.]]),[1.],[.2])
    assert r['solver_success']
    assert r['best_single_bound']==pytest.approx(.9)
    assert r['upper_bound_raw']==pytest.approx(.5)
    assert sum(r['mixture_weights'])==pytest.approx(1.)
