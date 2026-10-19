import numpy as np
from scipy.special import expit,log_expit,logsumexp
from scipy.optimize import brentq
from independent_crossgroup_soft5 import independent_group_info

def independent_objective(A,y,w,temperature):
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
            centered=distance-sorted(distance.tolist())[4]
            order=np.argsort(centered,kind='stable');inside=order[:5];outside=order[5:]
            def balance(t):
                return float(logsumexp(log_expit((t-centered[outside])/temperature))
                    -logsumexp(log_expit((centered[inside]-t)/temperature)))
            radius=brentq(balance,
                float(centered.min()-40*temperature),float(centered.max()+40*temperature),xtol=1e-16,rtol=1e-14)
            q=expit((radius-centered)/temperature)
            assert abs(q.sum()-5)<1e-10
        total-=np.log(q[same].sum()/5+1e-12)/(len(classes)*sum(y[valid]==y[i]))
    return float(total+.01*np.sum(w*np.log(len(w)*(w+1e-12))))
