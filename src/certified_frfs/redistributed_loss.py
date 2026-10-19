"""Shared-box dependency bound for a redistributed reduced metric.

The full metric uses weights ``w_full`` on all coordinates. The reduced metric
uses ``w_reduced`` on ``features``; these need not be the same. For each
object i, possible reduced nearest enemy j and full-space enemy witness k,
the bound controls d_full(i,k)-d_reduced(i,j) on the same coordinate box.
It is objectwise and finite-table only, not a classification-risk guarantee.
"""
import numpy as np
from .core import _inputs, interval_bounds, _interval_distances, _distance, certificate


def _max_weighted_difference(lo_i, hi_i, lo_j, hi_j, lo_k, hi_k, wf, wr):
    """Exact scalar-box maximum of wf|t-k|-wr|t-j| over t in [lo_i,hi_i]."""
    points=[lo_i,hi_i]
    for v in (lo_j,hi_j,lo_k,hi_k):
        if lo_i<=v<=hi_i:points.append(v)
    values=[]
    for t in points:
        values.append(wf*max(abs(t-lo_k),abs(t-hi_k))-
                      wr*max(0.,lo_j-t,t-hi_j))
    return max(values)


def redistributed_loss_bound(X,y,features,w_full,w_reduced,*,eps=.01,witness_pool=8):
    X,y,fs,wf=_inputs(X,y,features,w_full)
    wr=np.asarray(w_reduced,float)
    if wr.shape!=(len(fs),) or np.any(wr<0) or not np.isfinite(wr).all():
        raise ValueError('w_reduced must be a finite nonnegative retained vector')
    if witness_pool is not None and (int(witness_pool)!=witness_pool or witness_pool<1):
        raise ValueError('positive witness pool required')
    n,d=X.shape
    lo,hi=interval_bounds(X,eps)
    if len(np.unique(y))<2:
        return {'upper_bound':0.,'redistributed_bound':0.,'raw_redistributed_bound':0.,
                'scope':'finite_shared_box_redistributed_metric_dependency_loss_not_test_risk'}
    if len(fs)==d and np.array_equal(wr,wf):
        base=certificate(X,y,fs,weights=wf,eps=eps,method='hybrid')
        return {**base,'redistributed_bound':0.,'raw_redistributed_bound':0.,
                'scope':'finite_shared_box_redistributed_metric_dependency_loss_not_test_risk'}
    omitted=np.setdiff1d(np.arange(d),fs)
    reduced_full=np.zeros(d);reduced_full[fs]=wr
    lb,ub=_interval_distances(lo,hi,fs,reduced_full)
    full_distance=_distance(X,np.arange(d),wf);cross=y[:,None]!=y[None,:];values=[]
    for i in range(n):
        enemies=np.flatnonzero(cross[i]);cap=np.min(ub[i,enemies])
        possible=enemies[lb[i,enemies]<=cap+1e-12]
        witnesses=enemies[np.lexsort((enemies,full_distance[i,enemies]))]
        if witness_pool is not None:witnesses=witnesses[:witness_pool]
        jl,jh=lo[possible][:,fs][:,None,:],hi[possible][:,fs][:,None,:]
        kl,kh=lo[witnesses][:,fs][None,:,:],hi[witnesses][:,fs][None,:,:]
        il,ih=lo[i,fs],hi[i,fs]
        def at(t):
            far=np.maximum(abs(t-kl),abs(t-kh))
            near=np.maximum(0,np.maximum(jl-t,t-jh))
            return wf[fs]*far-wr*near
        terms=np.maximum(at(il),at(ih))
        for point in (jl,jh,kl,kh):
            terms=np.maximum(terms,at(np.clip(point,il,ih)))
        coef=wf[fs]-wr
        near=np.maximum(0,np.maximum(il-jh,jl-ih))
        far=np.maximum(abs(il-jh),abs(ih-jl))
        same=possible[:,None]==witnesses[None,:]
        terms=np.where(same[:,:,None],coef*np.where(coef>=0,far,near),terms)
        retained=terms.sum(axis=2)
        omitted_max=np.maximum(abs(lo[i,omitted]-hi[witnesses][:,omitted]),
                               abs(hi[i,omitted]-lo[witnesses][:,omitted]))@wf[omitted]
        signed=(retained+omitted_max[None,:]).min(axis=1)
        values.append(float(np.max(np.maximum(0,np.minimum(1-lb[i,possible],signed)))))
    raw=float(np.mean(values));bound=float(np.clip(raw+1e-10*(1+abs(raw)),0,1))
    return {'upper_bound':bound,'redistributed_bound':bound,'raw_redistributed_bound':raw,
            'redistributed_object_bounds':values,'witness_pool':witness_pool,
            'features':fs.tolist(),'full_weights':wf.tolist(),'reduced_weights':wr.tolist(),
            'scope':'finite_shared_box_redistributed_metric_dependency_loss_not_test_risk'}


def transfer_weights(X,features,w_full,*,power=2.,floor=1e-12):
    """Transfer omitted weights to retained coordinates by robust similarity.

    Similarity is training-only: absolute Pearson correlation after finite
    input validation. Each retained coordinate receives its own weight plus
    omitted weights proportionally to ``|corr|**power``. If all correlations
    vanish, omitted mass is spread uniformly. The returned vector preserves
    the full weight sum and is deliberately not normalized.
    """
    X=np.asarray(X,float);wf=np.asarray(w_full,float);fs=np.asarray(features,int)
    if X.ndim!=2 or wf.shape!=(X.shape[1],) or np.any(wf<0) or not np.isfinite(wf).all() or not np.isfinite(X).all():
        raise ValueError('invalid training table or weights')
    if len(fs)==0 or len(np.unique(fs))!=len(fs) or np.any(fs<0) or np.any(fs>=X.shape[1]):raise ValueError('nonempty valid distinct features required')
    omitted=np.setdiff1d(np.arange(X.shape[1]),fs);result=wf[fs].copy()
    if not len(omitted):return result
    for b in omitted:
        scores=np.zeros(len(fs));xb=X[:,b]
        for q,a in enumerate(fs):
            xa=X[:,a]
            if np.std(xa)>floor and np.std(xb)>floor:
                scores[q]=abs(np.corrcoef(xa,xb)[0,1])**power
        if scores.sum()<=floor:scores=np.ones(len(fs))
        result+=wf[b]*scores/scores.sum()
    return result
