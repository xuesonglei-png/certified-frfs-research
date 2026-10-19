"""Scalar root and coordinate gradient replay; no production target imports."""
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit,log_expit,logsumexp
from independent_ensemble_vote_v1 import grouping


def scalar_membership(distances,weights,temperature):
    D,R=np.asarray(distances,float),np.asarray(weights,float)
    if abs(R.sum()-5)<=1e-12:return np.ones_like(D),np.zeros_like(D)
    rank=np.argsort(D,kind='stable');inside=np.zeros(len(D),bool)
    accumulated=0.
    for j in rank:
        inside[j]=True;accumulated+=R[j]
        if accumulated>=5:break
    surplus=R[inside].sum()-5
    log_surplus=np.log(surplus) if surplus>0 else -np.inf
    def residual(radius):
        z=(radius-D)/temperature
        outside=logsumexp(np.log(R[~inside])+log_expit(z[~inside]))
        deficit=logsumexp(np.log(R[inside])+log_expit(-z[inside]))
        return float(np.logaddexp(outside,log_surplus)-deficit)
    radius=brentq(residual,float(D.min()-40*temperature),float(D.max()+40*temperature),xtol=1e-14,rtol=1e-14)
    z=(radius-D)/temperature
    return expit(z),expit(z)*expit(-z)


def rows(A,y,w,seed):
    group=grouping(y,seed);classes=np.unique(y);entries=[]
    for i in range(len(A)):
        ids=np.flatnonzero(group!=group[i])
        if len(ids)<5 or not any(y[ids]==y[i]):continue
        population={c:int(sum(y==c))-(int(y[i]==c)) for c in classes}
        sampled={c:int(sum(y[ids]==c)) for c in classes}
        if any(population[c]>0 and sampled[c]==0 for c in classes):continue
        coverage=np.array([population[y[j]]/sampled[y[j]] for j in ids])
        C=np.array([[abs(A[j,a]-A[i,a]) for a in range(A.shape[1])] for j in ids])
        entries.append((i,ids,C,C@w,coverage))
    return entries


def calibrated_temperature(A,y,anchor,seed):
    candidates=[]
    for _,ids,_,D,_ in rows(A,y,anchor,seed):
        if len(ids)<=5:continue
        values=sorted(D.tolist());gap=values[5]-values[4]
        if gap>0:candidates.append(gap/np.log(4.5*(len(ids)-5.5)/.25))
    return float(np.clip(np.median(candidates) if candidates else .005,.0005,.05))


def objective(A,y,w,temperatures,seeds,*,family='plurality',correction=True,reg=.01):
    A,y,w=np.asarray(A,float),np.asarray(y),np.asarray(w,float)
    total=0.;gradient=np.zeros_like(w);classes=np.unique(y)
    for seed,tau in zip(seeds,temperatures,strict=True):
        part=rows(A,y,w,seed)
        if not part:continue
        labels=np.array([y[i] for i,_,_,_,_ in part])
        for i,ids,C,D,coverage in part:
            R=coverage if correction else np.ones(len(ids))
            Q,S=scalar_membership(D,R,tau);weighted=S*R
            center=(weighted[:,None]*C).sum(axis=0)/max(float(weighted.sum()),np.finfo(float).tiny)
            votes={};derivatives={}
            for c in classes:
                chosen=y[ids]==c;votes[c]=float((Q[chosen]*R[chosen]).sum()/5)
                derivatives[c]=(weighted[chosen].sum()*center-(weighted[chosen,None]*C[chosen]).sum(axis=0))/(5*tau)
            alpha=1/(len(seeds)*len(np.unique(labels))*sum(labels==y[i]))
            if family=='own_mass':
                total-=alpha*np.log(votes[y[i]]+1e-12)
                gradient-=alpha*derivatives[y[i]]/(votes[y[i]]+1e-12)
            else:
                rivals=[c for c in classes if c!=y[i]]
                z=np.array([0.]+[(.1+votes[c]-votes[y[i]])/.05 for c in rivals]);norm=logsumexp(z)
                total+=alpha*.05*norm
                for probability,c in zip(np.exp(z[1:]-norm),rivals):gradient+=alpha*probability*(derivatives[c]-derivatives[y[i]])
        log=np.log(len(w)*(w+1e-12));total+=reg*np.sum(w*log)/len(seeds)
        gradient+=reg*(log+w/(w+1e-12))/len(seeds)
    return float(total),gradient
