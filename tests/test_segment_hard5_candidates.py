"""Independent exhaustive tiny-table interval checks and no query interface."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from segment_hard5_candidates import segment_candidates


def brute_open_interval_best(A,y,anchor,target):
    n=len(A); endpoints=[]
    for w in (anchor,target):
        endpoints.append([[sum(abs(A[i,a]*w[a]-A[j,a]*w[a]) for a in range(A.shape[1]))
                           for j in range(n)] for i in range(n)])
    D0,D1=np.array(endpoints); delta=D1-D0; breaks={0.,1.}
    for i in range(n):
        for j in range(n):
            if i==j: continue
            for k in range(j+1,n):
                if i==k: continue
                den=delta[i,j]-delta[i,k]
                if den:
                    t=(D0[i,k]-D0[i,j])/den
                    if 0<t<1: breaks.add(float(t))
    ordered=sorted(breaks); scores=[]
    for lo,hi in zip(ordered[:-1],ordered[1:]):
        t=lo+(hi-lo)/2
        if not lo<t<hi: continue
        predictions=[]; classes=np.unique(y)
        for i in range(n):
            ids=sorted((j for j in range(n) if j!=i),key=lambda j:(D0[i,j]+t*delta[i,j],j))[:5]
            predictions.append(max(classes,key=lambda c:(sum(y[j]==c for j in ids),-c)))
        scores.append(np.mean([np.mean(np.array(predictions)[y==c]==c) for c in classes]))
    return max(scores)


def test_event_merge_matches_exhaustive_all_crossings():
    # Unequal class counts exercise balanced rather than ordinary accuracy.
    for seed in (72,83,149):
        rng=np.random.default_rng(seed); A=rng.uniform(0,1,(10,3)); y=np.array([0]*4+[1]*3+[2]*3)
        anchor=np.array([.8,.15,.05]); target=np.array([.05,.15,.8])
        proposals,info=segment_candidates(A,y,anchor,target,max_candidates=16,chunk_size=7)
        assert info['estimated_best_open_interval_BA']==pytest.approx(brute_open_interval_best(A,y,anchor,target),abs=1e-12)
        assert [r['t'] for r in proposals[:7]]==[0.,.05,.1,.25,.5,.75,1.]
        assert len(proposals)<=16 and len({r['t'] for r in proposals})==len(proposals)
        for r in proposals[7:]: assert r['left']<r['t']<r['right']
        fixed_only,_=segment_candidates(A,y,anchor,target,max_candidates=7)
        assert len(fixed_only)==7


def test_degenerate_segment_and_stable_multiclass_ties():
    A=np.zeros((8,2)); y=np.array([0,0,0,1,1,1,2,2]); w=np.array([.5,.5])
    proposals,info=segment_candidates(A,y,w,w)
    assert info['pair_crossings']==0 and info['merged_intervals']==1
    assert info['estimated_best_open_interval_BA']==pytest.approx(brute_open_interval_best(A,y,w,w))
    with pytest.raises(ValueError): segment_candidates(A,y,w,np.array([1.,1.]))
