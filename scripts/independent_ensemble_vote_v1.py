"""Independent scalar distances, partition calibration and root/gradient replay."""
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit,log_expit,logsumexp


def grouping(y,seed):
    rng=np.random.default_rng(seed);out=np.empty(len(y),int)
    for label in sorted(set(y.tolist())):
        ids=np.array([i for i,value in enumerate(y) if value==label])
        for position,i in enumerate(rng.permutation(ids)):out[i]=position%3
    return out


def entries(A,y,w,seed):
    group=grouping(y,seed);rows=[]
    for i in range(len(A)):
        ids=np.flatnonzero(group!=group[i])
        if len(ids)<5 or not any(y[ids]==y[i]):continue
        C=np.array([[abs(A[j,a]-A[i,a]) for a in range(A.shape[1])] for j in ids])
        rows.append((i,ids,C,C@w))
    return rows


def temperature(A,y,anchor,seed):
    thresholds=[]
    for _,ids,C,D in entries(A,y,anchor,seed):
        if len(ids)<=5:continue
        ordered=sorted(D.tolist());gap=ordered[5]-ordered[4]
        if gap>0:thresholds.append(gap/np.log(4.5*(len(ids)-5.5)/.25))
    return float(np.clip(np.median(thresholds) if thresholds else .005,.0005,.05))


def objective(A,y,w,temperatures,seeds,*,family='plurality',reg=.01):
    if family not in ('own_mass','plurality'):raise ValueError('unknown nominal loss')
    A,y,w=np.asarray(A,float),np.asarray(y),np.asarray(w,float)
    total=0.;gradient=np.zeros_like(w);classes=np.unique(y)
    for seed,tau in zip(seeds,temperatures,strict=True):
        rows=entries(A,y,w,seed)
        if not rows:continue
        labels=np.array([y[i] for i,_,_,_ in rows])
        for i,ids,C,D in rows:
            if len(ids)==5:q=np.ones(5);s=np.zeros(5)
            else:
                rank=np.argsort(D,kind='stable');inside=np.zeros(len(D),bool);inside[rank[:5]]=True
                def balance(radius):
                    z=(radius-D)/tau
                    return float(logsumexp(log_expit(z[~inside]))-logsumexp(log_expit(-z[inside])))
                radius=brentq(balance,float(D.min()-40*tau),float(D.max()+40*tau),xtol=1e-14,rtol=1e-14)
                z=(radius-D)/tau;q=expit(z);s=expit(z)*expit(-z)
            center=(s[:,None]*C).sum(axis=0)/max(float(s.sum()),np.finfo(float).tiny)
            votes={c:float(q[y[ids]==c].sum()/5) for c in classes}
            derivatives={}
            for c in classes:
                selected=y[ids]==c
                derivatives[c]=(s[selected].sum()*center-(s[selected,None]*C[selected]).sum(axis=0))/(5*tau)
            alpha=1/(len(seeds)*len(np.unique(labels))*sum(labels==y[i]))
            if family=='own_mass':
                total-=alpha*np.log(votes[y[i]]+1e-12)
                gradient-=alpha*derivatives[y[i]]/(votes[y[i]]+1e-12)
            else:
                rivals=[c for c in classes if c!=y[i]]
                z=np.array([0.]+[(.1+votes[c]-votes[y[i]])/.05 for c in rivals])
                norm=logsumexp(z);total+=alpha*.05*norm
                for prob,c in zip(np.exp(z[1:]-norm),rivals):gradient+=alpha*prob*(derivatives[c]-derivatives[y[i]])
        log=np.log(len(w)*(w+1e-12));total+=reg*np.sum(w*log)/len(seeds)
        gradient+=reg*(log+w/(w+1e-12))/len(seeds)
    return float(total),gradient
