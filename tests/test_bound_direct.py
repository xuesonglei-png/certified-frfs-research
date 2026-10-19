import numpy as np
from certified_frfs.bound_direct import direct_bound_search


def test_direct_bound_search_preserves_mass_cap_and_monotonic_trace():
    rng=np.random.default_rng(12);X=rng.random((10,4));y=np.repeat([0,1],5);full=np.ones(4)/2;v=np.array([.6,.7,.7])
    fit=direct_bound_search(X,y,[0,1,2],full,v,rounds=2,proposals=5)
    w=np.array(fit['weights']);assert abs(w.sum()-full.sum())<1e-8 and w.max()<=1.+1e-8
    assert all(r['after']<=r['before']+1e-12 for r in fit['trace'])
