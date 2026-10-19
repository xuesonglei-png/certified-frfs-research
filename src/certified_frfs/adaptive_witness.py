"""Refresh sound majority witnesses for a fixed attribute subset.

Witness selection uses nominal training coordinates. Bounds still quantify
over the full perturbation box; full-table weights remain unchanged.
"""
import numpy as np
from .majority_metric import MajorityTerms


def adaptive_bounds(X,y,features,weights,*,eps=.01):
    X=np.asarray(X,float);fs=list(features);w=np.asarray(weights,float)
    if X.ndim!=2 or w.shape!=(X.shape[1],) or np.any(w<0) or not np.isfinite(w).all():
        raise ValueError('invalid full-table weights')
    if not fs or len(set(fs))!=len(fs) or any(int(a)!=a or a<0 or a>=X.shape[1] for a in fs):
        raise ValueError('nonempty valid distinct features required')
    radius=np.asarray(eps)
    if radius.ndim:radius=np.broadcast_to(radius,X.shape)[:,fs]
    ws=MajorityTerms(X[:,fs],y,weights=w[fs],eps=radius)
    return ws.bounds(range(len(fs)),allowed_enemies=2)


def adaptive_path(X,y,ranking,weights,*,retention=.9,eps=.01,cache=None,flexible=False):
    """Retain either fixed or refreshed witness certificates for each object."""
    X=np.asarray(X,float);y=np.asarray(y);d=X.shape[1]
    if sorted(ranking)!=list(range(d)) or not 0<retention<=1:raise ValueError('invalid path')
    fixed=MajorityTerms(X,y,weights=weights,eps=eps)
    cache={} if cache is None else cache
    def bound(fs):
        key=tuple(fs)
        if key not in cache:
            cache[key]=np.maximum(fixed.bounds(fs,allowed_enemies=2),adaptive_bounds(X,y,fs,weights,eps=eps))
            if flexible:
                from .flexible_witness import flexible_bounds
                cache[key]=np.maximum(cache[key],flexible_bounds(X,y,fs,weights,eps=eps,pool_size=8))
        return cache[key]
    current=list(range(d));protected=fixed.bounds(current,allowed_enemies=2)>1e-10
    groups=[(y==label)&protected for label in np.unique(y)]
    required=[int(np.ceil(retention*g.sum()-1e-12)) for g in groups]
    candidates=[current.copy()];trace=[]
    if protected.any():
        for a in ranking:
            if len(current)==1:break
            fs=[b for b in current if b!=a];bounds=bound(fs)
            counts=[int(((bounds>1e-10)&g).sum()) for g in groups]
            feasible=all(c>=r for c,r in zip(counts,required))
            trace.append({'feature':int(a),'features':fs,'feasible':feasible,'retained':counts})
            if feasible:current=fs;candidates.append(current.copy())
    return {'candidates':candidates,'trace':trace,'required':required,
        'protected_counts':[int(g.sum()) for g in groups],'allowed_enemies':2,
        'scope':'finite_training_majority_retention_fixed_or_refreshed_witnesses'}
