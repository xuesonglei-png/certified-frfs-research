"""Enemy-specific three-friend witnesses for shared-box 5NN certification."""
import numpy as np
from scipy.spatial.distance import cdist
from .core import _inputs,interval_bounds
from .margin import triplet_interval_min


def flexible_bounds(X,y,features,weights,*,eps=.01,pool_size=8):
    X,y,fs,w=_inputs(X,y,features,weights)
    if len(fs)==0 or (pool_size is not None and (int(pool_size)!=pool_size or pool_size<3)):
        raise ValueError('nonempty features and at least three possible friends required')
    lo,hi=interval_bounds(X,eps);lo=lo[:,fs];hi=hi[:,fs]
    D=cdist(X[:,fs]*w[fs],X[:,fs]*w[fs],'cityblock');np.fill_diagonal(D,np.inf)
    result=np.full(len(y),-np.inf)
    for i,label in enumerate(y):
        friends=np.flatnonzero((y==label)&(np.arange(len(y))!=i))
        if len(friends)<3:continue
        friends=friends[np.lexsort((friends,D[i,friends]))]
        if pool_size is not None:friends=friends[:pool_size]
        enemies=np.flatnonzero(y!=label)
        if len(enemies)<3:
            result[i]=np.inf;continue
        C=triplet_interval_min(lo[i][None,None,:],hi[i][None,None,:],
            lo[enemies][:,None,:],hi[enemies][:,None,:],lo[friends][None,:,:],hi[friends][None,:,:])
        pair=C@w[fs]
        # Third-largest certified separation gives each enemy three preceding
        # true-label references; these references may differ across enemies.
        enemy=np.partition(pair,len(friends)-3,axis=1)[:,len(friends)-3]
        result[i]=np.partition(enemy,2)[2]
    return result-1e-10*(1+np.abs(np.where(np.isfinite(result),result,0)))
