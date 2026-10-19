"""Shared-anchor full/reduced nearest-enemy comparison bound.

For a possible reduced nearest enemy j and ANY full-space enemy witness k,
gamma_C(i)-gamma_B(i) <= min(1-d_B(i,j), d_C(i,k)-d_B(i,j))
when d_B(i,j)<1; otherwise it is zero. Retained coordinates share i and
their signed difference is bounded jointly. Omitted coordinates contribute
their full-distance maxima. Minimize over k, maximize over possible j.
This remains an objectwise bound and does not retain across-object coupling.
"""
import numpy as np
from .core import _inputs,interval_bounds,_interval_distances,_distance,certificate
from .margin import triplet_interval_min


def shared_loss_bound(X,y,features,weights,*,eps=.01,witness_pool=8):
    X,y,fs,w=_inputs(X,y,features,weights);n,d=X.shape
    if witness_pool is not None and (int(witness_pool)!=witness_pool or witness_pool<1):
        raise ValueError('positive witness pool required')
    base=certificate(X,y,fs,weights=w,eps=eps,method='hybrid')
    if len(fs)==d or len(np.unique(y))<2:return {**base,'shared_anchor_bound':0.,'raw_shared_anchor_bound':0.}
    lo,hi=interval_bounds(X,eps);omitted=np.setdiff1d(np.arange(d),fs)
    lb,ub=_interval_distances(lo,hi,fs,w)
    full_distance=_distance(X,np.arange(d),w);cross=y[:,None]!=y[None,:];values=[]
    for i in range(n):
        enemies=np.flatnonzero(cross[i]);cap=np.min(ub[i,enemies])
        possible=enemies[lb[i,enemies]<=cap+1e-12]
        witnesses=enemies[np.lexsort((enemies,full_distance[i,enemies]))]
        if witness_pool is not None:witnesses=witnesses[:witness_pool]
        # |anchor-k| - |anchor-j| = -(|anchor-j|-|anchor-k|).
        terms=-triplet_interval_min(lo[i,fs][None,None,:],hi[i,fs][None,None,:],
            lo[possible][:,fs][:,None,:],hi[possible][:,fs][:,None,:],
            lo[witnesses][:,fs][None,:,:],hi[witnesses][:,fs][None,:,:])
        terms[possible[:,None]==witnesses[None,:]]=0.
        retained=terms@w[fs]
        far=np.maximum(abs(lo[i,omitted][None,:]-hi[witnesses][:,omitted]),
                       abs(hi[i,omitted][None,:]-lo[witnesses][:,omitted]))@w[omitted]
        signed=np.min(retained+far[None,:],axis=1)
        values.append(float(np.max(np.maximum(0,np.minimum(1-lb[i,possible],signed)))))
    raw=float(np.mean(values));bound=float(np.clip(raw+1e-10*(1+abs(raw)),0,1))
    return {**base,'upper_bound':min(base['upper_bound'],bound),
        'shared_anchor_bound':bound,'raw_shared_anchor_bound':raw,
        'shared_object_bounds':values,'witness_pool':witness_pool,
        'scope':'finite_shared_box_fuzzy_dependency_loss_not_classification_risk'}
