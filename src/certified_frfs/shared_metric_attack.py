"""Shared-table changed-metric attacks: lower evidence, never certification."""
import numpy as np
from scipy.spatial.distance import cdist


def dependency_gradient(Z,y,w):
    Z,y,w=np.asarray(Z,float),np.asarray(y),np.asarray(w,float)
    if Z.ndim!=2 or y.shape!=(len(Z),) or w.shape!=(Z.shape[1],) or not np.isfinite(Z).all() or np.any(w<0) or not np.isfinite(w).all():
        raise ValueError('invalid numeric metric table')
    D=cdist(Z*w,Z*w,metric='cityblock');D[y[:,None]==y[None,:]]=np.inf
    if len(np.unique(y))<2:return 1.,np.zeros_like(Z)
    nearest=D.argmin(axis=1);values=D[np.arange(len(Z)),nearest];active=values<1
    gradient=np.zeros_like(Z);ids=np.flatnonzero(active)
    contributions=np.sign(Z[ids]-Z[nearest[ids]])*w/len(Z)
    gradient[ids]+=contributions;np.add.at(gradient,nearest[ids],-contributions)
    return float(np.minimum(1,values).mean()),gradient


def attack_shared_metric(X,y,w_full,w_reduced,*,eps=.01,starts=8,steps=20,seed=19201):
    X=np.asarray(X,float)
    if eps<0 or starts<1 or steps<1 or X.min()<0 or X.max()>1:raise ValueError('invalid box or budgets')
    low,high=np.maximum(0,X-eps),np.minimum(1,X+eps);rng=np.random.default_rng(seed)
    def loss(Z):
        a,ga=dependency_gradient(Z,y,w_full);b,gb=dependency_gradient(Z,y,w_reduced)
        return a-b,ga-gb
    best_value,best_gradient=loss(X);best=X.copy();runs=[]
    for start in range(starts):
        Z=X.copy() if start==0 else rng.uniform(low,high);value,gradient=loss(Z);trace=[value]
        for _ in range(steps):
            accepted=None
            for scale in (1.,.5,.25,.125):
                trial=np.clip(Z+eps*scale*np.sign(gradient),low,high);new,grad=loss(trial)
                if new>value+1e-12:accepted=(trial,new,grad);break
            if accepted is None:break
            Z,value,gradient=accepted;trace.append(value)
        if value>best_value:best_value,best=value,Z.copy()
        runs.append({'start':start,'trace':trace})
    return {'observed_loss_lower':float(best_value),'perturbed_table':best.tolist(),'runs':runs,
        'eps':eps,'starts':starts,'steps':steps,'seed':seed,
        'scope':'local_shared_table_attack_lower_evidence_no_safety_certificate'}
