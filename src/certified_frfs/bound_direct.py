"""Deterministic fixed-support search for the existing valid dependency bound."""
import numpy as np
from .feasible_transport import weight_certificate
from .redistributed_loss import redistributed_loss_bound


def _project(z, cap):
    z=np.asarray(z,float);d=len(z);lo,hi=float(z.min()-cap),float(z.max())
    for _ in range(80):
        t=(lo+hi)/2
        if np.clip(z-t,0,cap).sum()>1:lo=t
        else:hi=t
    return np.clip(z-(lo+hi)/2,0,cap)


def direct_bound_search(X,y,features,full,initial,*,rounds=8,proposals=64,seed=29031):
    X,y=np.asarray(X,float),np.asarray(y);full=np.asarray(full,float);fs=np.asarray(features,int)
    mass=full.sum();cap=2*mass/len(full);v=np.asarray(initial,float).copy()
    if v.shape!=(len(fs),) or np.any(v<0) or abs(v.sum()-mass)>1e-8:raise ValueError('invalid initial support weights')
    rng=np.random.default_rng(seed);trace=[]
    def evaluate(w):return redistributed_loss_bound(X,y,fs,full,w,eps=.01)['upper_bound']
    current=evaluate(v)
    for iteration in range(rounds):
        candidates=[v]
        scale=cap*.25/(iteration+1)
        for _ in range(proposals):
            proposal=_project((v+rng.normal(0,scale,len(fs)))/mass,cap/mass)*mass
            candidates.append(proposal)
        values=np.array([evaluate(w) for w in candidates]);best=int(np.argmin(values))
        trace.append({'iteration':iteration,'before':current,'after':float(values[best]),'proposals':len(candidates)})
        if values[best]>=current-1e-12:break
        v=candidates[best];current=float(values[best])
    certificate=weight_certificate(X,y,fs,full,v)
    return {'features':fs.tolist(),'weights':v.tolist(),'bound':current,'certificate':certificate,
        'trace':trace,'seed':seed,'rounds':rounds,'proposals':proposals,
        'scope':'fixed_support_direct_bound_search_not_global_optimum_or_test_risk'}
