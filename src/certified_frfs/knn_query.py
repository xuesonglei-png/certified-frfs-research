"""Sufficient weighted kNN query majority certificates on shared boxes.

Targets are frozen nominal predictions. No correctness/population risk claim.
"""
import numpy as np
from scipy.spatial.distance import cdist
from .core import interval_bounds
from .margin import triplet_interval_min


def certify_knn_queries(reference,labels,queries,targets,features,*,weights=None,
                        eps=.01,k=5,n_friends=None,allowed_enemies=0):
    A,B=np.asarray(reference,float),np.asarray(queries,float)
    labels,targets=np.asarray(labels),np.asarray(targets)
    fs=list(features)
    if (A.ndim!=2 or B.ndim!=2 or A.shape[1]!=B.shape[1] or
            labels.shape!=(len(A),) or targets.shape!=(len(B),) or
            not fs or len(set(fs))!=len(fs) or int(k)!=k or not 1<=k<=len(A) or
            any(int(a)!=a or a<0 or a>=A.shape[1] for a in fs)):
        raise ValueError('invalid weighted query inputs')
    w=np.ones(A.shape[1]) if weights is None else np.asarray(weights,float)
    if w.shape!=(A.shape[1],) or not np.isfinite(w).all() or np.any(w<0):
        raise ValueError('invalid weights')
    alo,ahi=interval_bounds(A,eps);blo,bhi=interval_bounds(B,eps)
    q=int(k)//2+1
    if int(allowed_enemies)!=allowed_enemies or not 0<=allowed_enemies<=k-q:
        raise ValueError('enemy budget must preserve a strict kNN majority')
    if n_friends is not None and (int(n_friends)!=n_friends or n_friends<q):
        raise ValueError('n_friends must cover the strict majority')
    nf=q if n_friends is None else int(n_friends)
    distance=cdist(B[:,fs]*w[fs],A[:,fs]*w[fs],'cityblock')
    lower=np.full(len(B),-np.inf)
    for i,target in enumerate(targets):
        friends=np.flatnonzero(labels==target); enemies=np.flatnonzero(labels!=target)
        if len(friends)<q:continue
        if len(enemies)<=allowed_enemies:lower[i]=np.inf;continue
        friends=friends[np.lexsort((friends,distance[i,friends]))][:nf]
        terms=np.zeros((len(enemies),len(friends)))
        for a in fs:
            terms+=w[a]*triplet_interval_min(blo[i,a],bhi[i,a],
                alo[enemies,a,None],ahi[enemies,a,None],
                alo[friends,a][None,:],ahi[friends,a][None,:])
        enemy_bounds=np.partition(terms,-q,axis=1)[:,-q]
        bound=np.partition(enemy_bounds,int(allowed_enemies))[int(allowed_enemies)]
        lower[i]=bound-1e-10*(1+abs(bound))
    certified=lower>1e-10
    return {'certified':certified.tolist(),
            'lower_bounds':[float(v) if np.isfinite(v) else None for v in lower],
            'coverage':float(certified.mean()) if len(B) else 0.,'k':int(k),'q':q,
            'allowed_enemies':int(allowed_enemies),
            'scope':'fixed_nominal_query_label_shared_reference_and_query_boxes_not_correctness',
            'rounding':'ordinary_float_with_conservative_tolerance_not_directed_rounding'}
