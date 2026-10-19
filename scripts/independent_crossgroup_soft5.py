"""Scalar Brent replay with independent train-only grouping and box geometry."""
import numpy as np
from scipy.special import expit
from scipy.optimize import brentq


def independent_groups(y):
    y=np.asarray(y);rng=np.random.default_rng(20261071)
    group=np.empty(len(y),int)
    for label in sorted(set(y.tolist())):
        ids=np.array([i for i,value in enumerate(y) if value==label])
        for position,i in enumerate(rng.permutation(ids)):
            group[i]=position%3
    return group


def independent_group_info(y):
    y=np.asarray(y);group=independent_groups(y);valid=[];available=[]
    for i in range(len(y)):
        reference=[j for j in range(len(y)) if group[j]!=group[i]]
        if len(reference)>=5 and any(y[j]==y[i] for j in reference):
            valid.append(i);available.append(len(reference))
    return {'groups':3,'seed':20261071,'group_assignments':group.tolist(),
        'query_indices':valid,'excluded_query_indices':[i for i in range(len(y)) if i not in valid],
        'available_neighbors':available,
        'geometry':'enclosing_training_fold_minmax_not_crossfitted_preprocessing',
        'scope':'training_only_crossgroup_robust_soft5_not_hard_vote_or_risk'}


def independent_objective(A,y,w):
    A,y,w=np.asarray(A,float),np.asarray(y),np.asarray(w,float)
    info=independent_group_info(y);valid=info['query_indices']
    if not valid:return 0.
    g=np.array(info['group_assignments']);classes=np.unique(y[valid]);total=0.
    lo,hi=np.maximum(0,A-.01),np.minimum(1,A+.01)
    for i in valid:
        ids=np.flatnonzero(g!=g[i]);same=y[ids]==y[i];distance=np.zeros(len(ids))
        for a,weight in enumerate(w):
            lower=np.maximum(0,np.maximum(lo[ids,a]-hi[i,a],lo[i,a]-hi[ids,a]))
            upper=np.maximum(abs(lo[i,a]-hi[ids,a]),abs(hi[i,a]-lo[ids,a]))
            distance+=weight*np.where(same,upper,lower)
        if len(ids)==5:q=np.ones(5)
        else:
            radius=brentq(lambda t:float(expit((t-distance)/.05).sum()-5),
                float(distance.min()-2),float(distance.max()+2),xtol=1e-13,rtol=1e-14)
            q=expit((radius-distance)/.05)
            assert abs(q.sum()-5)<1e-10
        total-=np.log(q[same].sum()/5+1e-12)/(len(classes)*sum(y[valid]==y[i]))
    return float(total+.01*np.sum(w*np.log(len(w)*(w+1e-12))))
