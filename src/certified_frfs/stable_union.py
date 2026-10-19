"""Training-fold union with fresh majority retention certification."""
import numpy as np
from .majority_metric import MajorityTerms


def union_features(fold_subsets,d):
    if not fold_subsets or d<1:raise ValueError('nonempty fold collection required')
    fs=sorted({a for subset in fold_subsets for a in subset})
    if not fs or any(int(a)!=a or a<0 or a>=d for a in fs):raise ValueError('invalid union features')
    return fs


def certify_union(X,y,features,weights,*,retention=.9):
    X,y=np.asarray(X,float),np.asarray(y);d=X.shape[1]
    if not 0<retention<=1:raise ValueError('invalid retention')
    fs=union_features([features],d);workspace=MajorityTerms(X,y,weights=weights)
    full=workspace.bounds(range(d),allowed_enemies=2);protected=full>1e-10
    groups=[(y==label)&protected for label in np.unique(y)]
    required=[int(np.ceil(retention*g.sum()-1e-12)) for g in groups]
    order=np.argsort(-np.asarray(weights),kind='stable').tolist();restoration=[]
    while True:
        bounds=workspace.bounds(fs,allowed_enemies=2)
        counts=[int(np.sum((bounds>1e-10)&g)) for g in groups]
        if all(c>=r for c,r in zip(counts,required)):break
        a=next(a for a in order if a not in fs);fs=sorted(fs+[a]);restoration.append(a)
    return {'features':fs,'restoration':restoration,'required':required,'retained_counts':counts,
        'protected_counts':[int(g.sum()) for g in groups],
        'lower_bounds':np.nan_to_num(bounds,posinf=1e100,neginf=-1e100).tolist(),
        'scope':'fresh_training_majority_retention_not_union_monotonicity_or_test_risk'}
