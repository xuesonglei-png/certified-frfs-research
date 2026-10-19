"""Conservative shared-box certificates for kNN majority decisions."""
import numpy as np
from scipy.spatial.distance import cdist
from .core import interval_bounds
from .margin import triplet_interval_min


def certified_knn_coverage(X, y, features, *, k=5, eps=.01, n_friends=None, weights=None):
    """Certify a target label when q=majority target friends beat every enemy.

    For each enemy, at least q nominal target friends are selected whose
    triplet lower bounds are positive. This is sufficient, conservative and
    finite-table only; it is not an exact kNN robust margin.
    """
    X, y = np.asarray(X, float), np.asarray(y)
    features = list(features)
    if (X.ndim!=2 or y.ndim!=1 or len(y)!=len(X) or k>=len(X) or
            not features or len(set(features)) != len(features) or k < 1 or int(k) != k or
            any(a<0 or a>=X.shape[1] for a in features)):
        raise ValueError('invalid kNN certificate inputs')
    q = int(k)//2 + 1
    weights=np.ones(X.shape[1]) if weights is None else np.asarray(weights,float)
    if weights.shape!=(X.shape[1],) or not np.isfinite(weights).all() or np.any(weights<0):
        raise ValueError('invalid metric weights')
    n_friends = max(q, int(n_friends or q))
    lo, hi = interval_bounds(X, eps)
    distance = cdist(X[:, features]*weights[features], X[:, features]*weights[features], 'cityblock')
    lower = np.full(len(X), -np.inf)
    for i in range(len(X)):
        friends = np.flatnonzero((y == y[i]) & (np.arange(len(X)) != i))
        enemies = np.flatnonzero(y != y[i])
        if len(friends) < q or not len(enemies):
            continue
        friends = friends[np.lexsort((friends, distance[i, friends]))][:n_friends]
        terms = np.zeros((len(enemies), len(friends)))
        for a in features:
            terms += weights[a]*triplet_interval_min(lo[i,a], hi[i,a], lo[enemies,a,None], hi[enemies,a,None],
                                          lo[friends,a][None,:], hi[friends,a][None,:])
        # For every enemy, q positive friend-vs-enemy bounds suffice.
        qth = np.partition(terms, -q, axis=1)[:, -q]
        lower[i] = qth.min() - 1e-10*(1+abs(qth.min()))
    positive = lower > 1e-10
    classes=[]
    for label in np.unique(y):
        mask=y==label
        classes.append({'label':int(label),'count':int(mask.sum()),'positive':int(positive[mask].sum()),
                        'fraction':float(positive[mask].mean())})
    return {'lower_bounds':[float(v) if np.isfinite(v) else None for v in lower],
            'certified_indices':np.flatnonzero(positive).tolist(),
            'class_coverage':classes,
            'balanced_coverage':float(np.mean([r['fraction'] for r in classes])),
            'min_class_coverage':min(r['fraction'] for r in classes),
            'k':int(k),'majority_q':q,
            'scope':'sufficient_robust_LOO_kNN_majority_training_decision_not_test_risk'}


def select_knn_majority(X,y,ranking,*,eps=.01,penalty=.02,k=5):
    d=np.asarray(X).shape[1]
    if sorted(ranking)!=list(range(d)) or penalty<0:
        raise ValueError('invalid ranking or penalty')
    trace=[]
    for size in sorted(set(max(1,int(np.ceil(f*d))) for f in [.1,.2,.35,.5,.65,.8,1.])):
        fs=ranking[:size]
        cert=certified_knn_coverage(X,y,fs,k=k,eps=eps)
        objective=.5*cert['balanced_coverage']+.5*cert['min_class_coverage']-penalty*size/d
        trace.append({'features':fs,'objective':objective,'coverage':cert})
    best=max(trace,key=lambda r:(round(r['objective'],12),-len(r['features'])))
    return {'features':best['features'],'trace':trace,'certificate':best['coverage']}
