import numpy as np
from certified_frfs.core import gamma
from mixture_dependency_certificate_v1 import mixture_dependency_certificate


def test_whole_table_bound_on_shared_perturbations_and_budget_stop():
    rng=np.random.default_rng(118);A=rng.uniform(.05,.95,(12,3));y=np.array([0]*6+[1]*6)
    full=np.ones(3)/np.sqrt(3);fs=[0,2];wr=np.array([.7,np.sqrt(3)-.7])
    result=mixture_dependency_certificate(A,y,fs,full,wr,pair_budget=2)
    assert result['LPs']<=24
    assert result['upper_bound']<=result['original_upper_bound']+1e-10
    assert np.all(np.asarray(result['object_bounds'])<=np.asarray(result['original_object_bounds'])+1e-12)
    embedded=np.zeros(3);embedded[fs]=wr
    for _ in range(50):
        Z=np.clip(A+rng.uniform(-.01,.01,A.shape),0,1)
        loss=gamma(Z,y,weights=full)-gamma(Z,y,fs,weights=embedded)
        assert loss<=result['upper_bound']+1e-10


def test_exact_full_metric_zero_loss():
    A=np.random.default_rng(188).random((10,2));y=np.array([0]*5+[1]*5);w=np.ones(2)/np.sqrt(2)
    result=mixture_dependency_certificate(A,y,[0,1],w,w)
    assert result['upper_bound']==0. and result['LPs']==0
