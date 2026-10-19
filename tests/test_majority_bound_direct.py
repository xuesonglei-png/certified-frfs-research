import numpy as np
from certified_frfs.majority_bound_direct import majority_bound_search


def test_every_accepted_metric_keeps_counts_and_bound_descent():
    rng=np.random.default_rng(38);X=np.r_[rng.uniform(0,.2,(5,4)),rng.uniform(.8,1,(5,4))];y=np.repeat([0,1],5)
    full=np.ones(4)/2;fit=majority_bound_search(X,y,[0,1,2,3],full,full,rounds=2,proposals=4)
    assert fit['certificate']['majority_feasible']
    for t in fit['trace']:
        assert t['after']<=t['before']+1e-12
        w=np.array(t['weights']);assert w.min()>=0 and abs(w.sum()-2)<1e-8 and w.max()<=1+1e-8
        checks=t['majority_checks']
        if t['after']<t['before']:
            assert checks[-1]['valid']
            assert all(c>=r for c,r in zip(checks[-1]['counts'],fit['required']))
