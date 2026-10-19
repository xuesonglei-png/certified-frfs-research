"""Class-balanced robust soft-neighbor objective with diagonal L1 geometry.

Same-class similarities use distance upper bounds; enemies use lower bounds.
Their normalized mass lower-bounds the nominal stochastic-neighbor mass for
every admissible shared-box realization. It does not certify hard 5NN votes.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from .core import interval_bounds


def softvote_terms(X,y,eps=.01):
    X=np.asarray(X,float);y=np.asarray(y);lo,hi=interval_bounds(X,eps)
    if y.shape!=(len(X),):raise ValueError('invalid labels')
    same=y[:,None]==y[None,:];np.fill_diagonal(same,False)
    valid=same.any(axis=1)
    low=np.maximum(0,np.maximum(lo[:,None,:]-hi[None,:,:],lo[None,:,:]-hi[:,None,:]))
    high=np.maximum(abs(lo[:,None,:]-hi[None,:,:]),abs(hi[:,None,:]-lo[None,:,:]))
    C=np.where(same[:,:,None],high,low)
    mask=~np.eye(len(X),dtype=bool)
    return C[valid],same[valid],mask[valid],y[valid]


def softvote_objective(w,C,same,mask,labels,temperature=.05,reg=.01):
    if not len(labels):return 0.,np.zeros_like(w)
    classes=np.unique(labels)
    alpha=np.array([1/(len(classes)*np.sum(labels==label)) for label in labels])
    z=-np.einsum('ijd,d->ij',C,w)/temperature
    za=np.where(mask,z,-np.inf);zs=np.where(same,z,-np.inf)
    norm=logsumexp(za,axis=1);positive=logsumexp(zs,axis=1)
    pa=np.exp(za-norm[:,None]);ps=np.exp(zs-positive[:,None])
    loss=float(alpha@(norm-positive))+reg*np.sum(w*np.log(len(w)*w))
    grad=np.einsum('i,ij,ijd->d',alpha,ps-pa,C)/temperature+reg*(np.log(len(w)*w)+1)
    return float(loss),grad


def learn_softvote_weights(X,y,*,eps=.01,temperature=.05,reg=.01):
    d=np.asarray(X).shape[1];C,same,mask,labels=softvote_terms(X,y,eps)
    initial=np.ones(d)/d
    if not len(labels):return initial/np.linalg.norm(initial)
    result=minimize(lambda w:softvote_objective(w,C,same,mask,labels,temperature,reg),
        initial,jac=True,method='SLSQP',bounds=[(1e-6,None)]*d,
        constraints={'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(d)},
        options={'maxiter':200,'ftol':1e-9})
    if not result.success:raise RuntimeError('softvote metric fit failed: '+result.message)
    w=np.maximum(result.x,1e-6);return w/np.linalg.norm(w)
