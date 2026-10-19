"""Training-only diagonal metric learning for fuzzy-rough experiments.

The metric is learned from fixed nominal same/enemy triplets, then frozen.
It is deliberately separate from mutual-information weighting and does not
use validation or outer-test labels.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize


def learn_triplet_weights(X, y, *, margin=.02, reg=.02, seed=0, temperature=.05,
                          max_triplets=4000, min_weight=1e-4):
    X, y = np.asarray(X,float), np.asarray(y)
    if X.ndim != 2 or len(y)!=len(X) or not np.isfinite(X).all():
        raise ValueError('invalid training table')
    n,d=X.shape
    if d==0 or margin<0 or reg<0 or temperature<=0 or min_weight<=0 or d*min_weight>=1:
        raise ValueError('invalid metric parameters')
    rng=np.random.default_rng(seed)
    triples=[]
    for i in range(n):
        friends=np.flatnonzero((y==y[i]) & (np.arange(n)!=i))
        enemies=np.flatnonzero(y!=y[i])
        if not len(friends) or not len(enemies): continue
        # Coordinate-wise hard pairs expose complementary dimensions without
        # fitting a classifier or consulting held-out labels.
        friend=friends[np.argmin(np.abs(X[friends]-X[i]).sum(axis=1))]
        enemy=enemies[np.argmin(np.abs(X[enemies]-X[i]).sum(axis=1))]
        triples.append((i,friend,enemy))
    if not triples: return np.full(d,1/np.sqrt(d))
    if len(triples)>max_triplets:
        triples=[triples[k] for k in rng.choice(len(triples),max_triplets,replace=False)]
    diff_friend=np.abs(X[[i for i,_,_ in triples]]-X[[j for _,j,_ in triples]])
    diff_enemy=np.abs(X[[i for i,_,_ in triples]]-X[[j for _,_,j in triples]])
    delta=diff_friend-diff_enemy
    labels=y[[i for i,_,_ in triples]]
    counts={label:np.sum(labels==label) for label in np.unique(labels)}
    balanced=np.array([1/(len(counts)*counts[label]) for label in labels])
    # Minimize smooth hinge under an L1 normalization. Positive weights are
    # retained even for weak coordinates so deleting them has a fixed meaning.
    def objective(w):
        z=margin+delta@w
        return float(temperature*np.dot(balanced,np.logaddexp(0,z/temperature))+
                     reg*np.sum(w*np.log(d*w)))
    def gradient(w):
        z=margin+delta@w
        p=1/(1+np.exp(-np.clip(z/temperature,-50,50)))
        return delta.T@(balanced*p)+reg*(np.log(d*w)+1)
    x0=np.full(d,1/d)
    result=minimize(objective,x0,jac=gradient,method='SLSQP',bounds=[(min_weight,None)]*d,
                    constraints={'type':'eq','fun':lambda w:w.sum()-1,
                                 'jac':lambda w:np.ones(d)},options={'maxiter':300,'ftol':1e-10})
    w=np.asarray(result.x if result.success else x0,float)
    w=np.maximum(w,min_weight); w/=w.sum()
    return w/np.linalg.norm(w)
