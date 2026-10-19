"""Fold-local recipe selection: refit certified paths instead of fixed subsets."""
import numpy as np
from sklearn.svm import LinearSVC
from . import baselines
from .majority_metric import protected_majority_path
from .recent import select_sfrip


FRACTIONS=(.2,.35,.5,.65,.8,1.)


def recipe_candidates(A,y,w,method,*,include_margin=False,refresh_witness=False,flexible_witness=False):
    d=A.shape[1];rows=[];paths=[];witness_cache={}
    if method=='protected':
        ranks=('mi','relief','weights','margin') if include_margin else ('mi','relief','weights')
        for rank in ranks:
            relevance=(baselines.mi_scores(A*w,y,random_state=101) if rank=='mi'
                else baselines.relieff_scores(A*w,y,random_state=101) if rank=='relief' else w if rank=='weights'
                else np.mean(np.abs(LinearSVC(C=1.,class_weight='balanced',dual='auto',random_state=101)
                    .fit(A*w,y).coef_),axis=0))
            ranking=np.argsort(relevance,kind='stable').tolist()
            if refresh_witness or flexible_witness:
                from .adaptive_witness import adaptive_path
                path=adaptive_path(A,y,ranking,w,cache=witness_cache,flexible=flexible_witness)
            else:path=protected_majority_path(A,y,ranking,w,allowed_enemies=2)
            paths.append({'rank':rank,**path})
            for fraction in FRACTIONS:
                target=max(1,int(np.ceil(fraction*d)))
                # Feasible sizes can have gaps. Choose the closest available
                # size; ties favor the larger subset rather than violating a bound.
                fs=min(path['candidates'],key=lambda v:(abs(len(v)-target),-len(v),tuple(v)))
                rows.append({'config':{'rank':rank,'fraction':fraction},'features':fs})
    elif method=='all':rows=[{'config':{},'features':list(range(d))}]
    elif method=='sfrip':
        rows=[{'config':{'beta':b,'confidence':c},'features':select_sfrip(A*w,y,beta=b,confidence=c)['features']}
              for b in (2.,3.,5.) for c in (.6,.8)]
    elif method in ('mi','relief','mrmr'):
        function={'mi':baselines.select_mi,'relief':baselines.select_relieff,'mrmr':baselines.select_mrmr}[method]
        rows=[{'config':{'fraction':f},'features':function(A*w,y,k=max(1,int(np.ceil(f*d))))} for f in FRACTIONS]
    else:raise ValueError('unknown method')
    return rows,paths
