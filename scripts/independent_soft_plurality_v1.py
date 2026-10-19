"""Independent scalar-root multiclass fuzzy-vote value and gradient replay."""
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit,log_expit,logsumexp
from independent_crossgroup_soft5 import independent_groups


def objective(A,y,w,temperature,*,margin=.1,smoothing=.05,reg=.01,family='plurality'):
    A,y,w=np.asarray(A,float),np.asarray(y),np.asarray(w,float)
    if family not in ('plurality','own_mass'):raise ValueError('unknown nominal objective')
    group=independent_groups(y);eligible=[]
    for i in range(len(A)):
        ids=np.flatnonzero(group!=group[i])
        if len(ids)>=5 and any(y[ids]==y[i]):eligible.append((i,ids))
    if not eligible:return 0.,np.zeros_like(w)
    labels=np.array([y[i] for i,_ in eligible]);classes=np.unique(y);value=0.;gradient=np.zeros_like(w)
    for i,ids in eligible:
        C=np.array([[abs(A[j,a]-A[i,a]) for a in range(A.shape[1])] for j in ids]);D=C@w
        if len(ids)==5:q=np.ones(5);s=np.zeros(5)
        else:
            rank=np.argsort(D,kind='stable');inside=np.zeros(len(D),bool);inside[rank[:5]]=True
            def residual(radius):
                z=(radius-D)/temperature
                return float(logsumexp(log_expit(z[~inside]))-logsumexp(log_expit(-z[inside])))
            radius=brentq(residual,float(D.min()-40*temperature),float(D.max()+40*temperature),xtol=1e-14,rtol=1e-14)
            z=(radius-D)/temperature;q=expit(z);s=expit(z)*expit(-z)
        center=np.sum(s[:,None]*C,axis=0)/max(float(s.sum()),np.finfo(float).tiny)
        votes={};derivatives={}
        for c in classes:
            same=y[ids]==c;votes[c]=float(q[same].sum()/5)
            derivatives[c]=(s[same].sum()*center-np.sum(s[same,None]*C[same],axis=0))/(5*temperature)
        alpha=1/(len(np.unique(labels))*sum(labels==y[i]))
        if family=='own_mass':
            value-=alpha*np.log(votes[y[i]]+1e-12)
            gradient-=alpha*derivatives[y[i]]/(votes[y[i]]+1e-12)
        else:
            rivals=[c for c in classes if c!=y[i]]
            z=np.array([0.]+[(margin+votes[c]-votes[y[i]])/smoothing for c in rivals])
            norm=logsumexp(z);value+=alpha*smoothing*norm
            for probability,c in zip(np.exp(z[1:]-norm),rivals):gradient+=alpha*probability*(derivatives[c]-derivatives[y[i]])
    log=np.log(len(w)*(w+1e-12))
    return float(value+reg*np.sum(w*log)),gradient+reg*(log+w/(w+1e-12))
