"""Bound descent with unchanged majority protection on every accepted step."""
import numpy as np
from .bound_direct import _project
from .majority_metric import MajorityTerms
from .feasible_transport import weight_certificate
from .redistributed_loss import redistributed_loss_bound


def majority_bound_search(X,y,features,full,initial,*,rounds=8,proposals=64,seed=29031):
    X,y=np.asarray(X,float),np.asarray(y);full=np.asarray(full,float);fs=np.asarray(features,int)
    mass=full.sum();cap=2*mass/len(full);v=np.asarray(initial,float).copy()
    if v.shape!=(len(fs),) or np.any(v<0) or abs(v.sum()-mass)>1e-8 or np.any(v>cap+1e-8):
        raise ValueError('invalid support metric')
    original=MajorityTerms(X,y,weights=full).bounds(range(X.shape[1]),allowed_enemies=2)>1e-10
    groups=[(y==label)&original for label in np.unique(y)]
    required=[int(np.ceil(.9*g.sum()-1e-12)) for g in groups]
    def coverage(w):
        margins=MajorityTerms(X[:,fs],y,weights=w).bounds(range(len(fs)),allowed_enemies=2)
        counts=[int(np.sum((margins>1e-10)&g)) for g in groups]
        return counts,all(c>=r for c,r in zip(counts,required))
    initial_counts,valid=coverage(v)
    if not valid:raise ValueError('initial metric must retain original majority')
    rng=np.random.default_rng(seed);trace=[]
    current=redistributed_loss_bound(X,y,fs,full,v)['upper_bound']
    for iteration in range(rounds):
        candidates=[];scale=cap*.25/(iteration+1)
        for index in range(proposals):
            w=_project((v+rng.normal(0,scale,len(fs)))/mass,cap/mass)*mass
            value=redistributed_loss_bound(X,y,fs,full,w)['upper_bound']
            if value<current-1e-12:candidates.append((value,index,w))
        candidates.sort(key=lambda item:(item[0],item[1]));checks=[];accepted=None
        for value,index,w in candidates:
            counts,valid=coverage(w);checks.append({'proposal':index,'bound':value,'counts':counts,'valid':valid})
            if valid:accepted=(value,w);break
        before=current
        if accepted is not None:current,v=accepted
        trace.append({'iteration':iteration,'before':before,'after':current,'weights':v.tolist(),
            'majority_checks':checks,'proposals':proposals})
        if accepted is None:break
    cert=weight_certificate(X,y,fs,full,v,protected=original)
    if not cert['majority_feasible']:raise RuntimeError('protected majority lost on accepted path')
    return {'features':fs.tolist(),'weights':v.tolist(),'bound':current,'certificate':cert,'trace':trace,
        'required':required,'initial_counts':initial_counts,'seed':seed,'rounds':rounds,'proposals':proposals,
        'scope':'fixed_support_majority_preserving_bound_descent_not_global_optimum_or_prediction'}
