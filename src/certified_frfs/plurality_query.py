"""Conservative kNN plurality certificates from shared-anchor rank bounds.

No strict-majority assumption. Frozen nominal targets may still be wrong.
O(query_count * feature_count * reference_count**2); one query at a time.
"""
import numpy as np
from .core import _inputs,interval_bounds
from .margin import triplet_interval_min


def certify_plurality_queries(reference,labels,queries,targets,features,*,weights=None,
                             eps=.01,k=5,excluded_reference=None):
    A,y,fs,w=_inputs(reference,labels,features,weights)
    B=np.asarray(queries,float);targets=np.asarray(targets)
    if B.ndim!=2 or B.shape[1]!=A.shape[1] or targets.shape!=(len(B),) or not len(fs):
        raise ValueError('invalid queries or empty features')
    if int(k)!=k or not 1<=k<=len(A):raise ValueError('invalid k')
    excluded=np.full(len(B),-1,int) if excluded_reference is None else np.asarray(excluded_reference)
    if excluded.shape!=(len(B),) or np.any(excluded!=excluded.astype(int)) or np.any((excluded< -1)|(excluded>=len(A))):
        raise ValueError('invalid excluded reference indices')
    if np.any((excluded>=0)&(k>len(A)-1)):raise ValueError('insufficient LOO reference size')
    lo,hi=interval_bounds(A,eps);blo,bhi=interval_bounds(B,eps)
    classes=np.unique(y);flags=[];records=[]
    for i,target in enumerate(targets):
        active=np.arange(len(A))!=excluded[i]
        ids=np.flatnonzero(active);n=len(ids)
        # L[j,h] <= distance(query,j)-distance(query,h).
        L=np.zeros((n,n))
        for a in fs:
            L+=w[a]*triplet_interval_min(blo[i,a],bhi[i,a],
                lo[ids,a,None],hi[ids,a,None],lo[ids,a][None,:],hi[ids,a][None,:])
        precedes=L>1e-10*(1+abs(L))
        np.fill_diagonal(precedes,False)
        lower_rank=1+precedes.sum(axis=1)
        upper_rank=n-precedes.sum(axis=0)
        guaranteed=upper_rank<=k;possible=lower_rank<=k
        lower=np.array([np.sum(guaranteed&(y[ids]==label)) for label in classes])
        upper=np.array([np.sum(possible&(y[ids]==label)) for label in classes])
        # A k-neighbor list has exactly k members, regardless of tie-breaking.
        lower=np.maximum(lower,k-(upper.sum()-upper))
        upper=np.minimum(upper,k-(lower.sum()-lower))
        match=np.flatnonzero(classes==target)
        certified=bool(len(match) and lower[match[0]]>np.max(np.delete(upper,match[0]),initial=-1))
        flags.append(certified)
        records.append({'vote_lower':lower.astype(int).tolist(),'vote_upper':upper.astype(int).tolist(),
            'guaranteed_neighbor_count':int(guaranteed.sum()),'possible_neighbor_count':int(possible.sum())})
    return {'certified':flags,'classes':classes.tolist(),'records':records,
        'coverage':float(np.mean(flags)) if flags else 0.,'k':int(k),
        'scope':'fixed_target_unique_plurality_on_shared_boxes_not_correctness_or_population_risk',
        'rounding':'ordinary_float_conservative_comparison_not_directed_rounding'}
