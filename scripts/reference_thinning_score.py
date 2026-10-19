"""Training-only hard5 score under shared Bernoulli reference thinning.

This stress objective is not cross-validation: metric weights can depend on
all training labels. It supplies no unbiased risk or population guarantee.
"""
import numpy as np
from scipy.spatial.distance import cdist


def reference_masks(n, *, draws=8, retain=.8, seed=20261101):
    if int(n)!=n or n<7 or int(draws)!=draws or draws<1 or not 0<retain<=1:
        raise ValueError('at least seven rows and valid thinning settings required')
    rng=np.random.default_rng(seed); masks=[]
    for _ in range(draws):
        for attempt in range(10000):
            mask=rng.random(n)<retain
            # Six globally retained objects leave at least five after LOO.
            if mask.sum()>=6: break
        else: raise RuntimeError('reference mask rejection budget exhausted')
        masks.append(mask)
    return np.array(masks)


def thinning_score(A,y,features,weights,masks):
    A=np.asarray(A,float); y=np.asarray(y); fs=list(features)
    w=np.asarray(weights,float); masks=np.asarray(masks)
    if (A.ndim!=2 or len(A)<7 or not np.isfinite(A).all()
        or y.shape!=(len(A),) or not fs or fs!=sorted(set(fs))
        or any(int(a)!=a or not 0<=a<A.shape[1] for a in fs)
        or w.shape!=(len(fs),) or not np.isfinite(w).all() or np.any(w<0)
        or masks.ndim!=2 or masks.shape[1]!=len(A) or len(masks)==0
        or masks.dtype!=np.dtype(bool) or np.any(masks.sum(axis=1)<6)):
        raise ValueError('invalid training metric or shared reference masks')
    scaled=A[:,fs]*w
    D=cdist(scaled,scaled,'cityblock'); np.fill_diagonal(D,np.inf)
    order=np.argsort(D,axis=1,kind='stable')
    classes=np.unique(y); reports=[]
    for mask in masks:
        available=mask[order] & (order!=np.arange(len(A))[:,None])
        take=available & (np.cumsum(available,axis=1)<=5)
        assert np.all(take.sum(axis=1)==5)
        neighbors=order[take].reshape(len(A),5)
        votes=np.stack([(y[neighbors]==c).sum(axis=1) for c in classes],axis=1)
        predictions=classes[votes.argmax(axis=1)]
        ba=float(np.mean([np.mean(predictions[y==c]==c) for c in classes]))
        reports.append({'BA':ba,'predictions':predictions.tolist()})
    return {'mean_BA':float(np.mean([r['BA'] for r in reports])),
            'draws':reports,'scope':'training_reference_thinning_not_crossvalidation'}
