"""Convex witness-mixture refinement of shared-anchor fuzzy dependency loss.

LP proposes simplex weights; the reported bound is independently evaluated
at those normalized nonnegative weights, not taken from the solver objective.
No generalization claim or novelty claim is implied by this implementation.
"""
import numpy as np
from scipy.optimize import linprog
from .core import _inputs,interval_bounds,_interval_distances,_distance
from .shared_loss import shared_loss_bound


def mixture_signed_bound(lo,hi,i,j,witnesses,features,weights):
    """Upper bound sup_Z min_k(d_C(i,k)-d_B(i,j)) by a witness mixture."""
    n,d=lo.shape;K=len(witnesses);fs=set(features);matrices=[]
    for a in range(d):
        if a in fs:
            anchors=np.unique([lo[i,a],hi[i,a],np.clip(lo[j,a],lo[i,a],hi[i,a]),
                               np.clip(hi[j,a],lo[i,a],hi[i,a])])
        else:anchors=np.unique([lo[i,a],hi[i,a]])
        high=np.maximum(abs(anchors[:,None]-lo[witnesses,a]),abs(anchors[:,None]-hi[witnesses,a]))
        if a in fs:
            near=np.maximum(0,np.maximum(lo[j,a]-anchors,anchors-hi[j,a]))
            high=high-near[:,None]
            high[:,np.asarray(witnesses)==j]=0.
        matrices.append(high)
    count=sum(len(M) for M in matrices)
    constraints=np.zeros((count,K+d));row=0
    for a,M in enumerate(matrices):
        constraints[row:row+len(M),:K]=M
        constraints[row:row+len(M),K+a]=-1;row+=len(M)
    equality=np.r_[np.ones(K),np.zeros(d)][None,:]
    fit=linprog(np.r_[np.zeros(K),weights],A_ub=constraints,b_ub=np.zeros(count),
        A_eq=equality,b_eq=[1.],bounds=[(0,None)]*K+[(None,None)]*d,method='highs')
    # A single witness is a valid fallback, including solver failure.
    candidates=np.eye(K)
    if fit.success and np.isfinite(fit.x[:K]).all():
        lam=np.maximum(fit.x[:K],0)
        if lam.sum()>0:candidates=np.vstack([candidates,lam/lam.sum()])
    values=[float(sum(weights[a]*np.max(M@lam) for a,M in enumerate(matrices))) for lam in candidates]
    raw=min(values)
    return raw+1e-8*(1+abs(raw)),bool(fit.success)


def mixture_loss_bound(X,y,features,weights,*,eps=.01,witness_pool=8):
    X,y,fs,w=_inputs(X,y,features,weights)
    old=shared_loss_bound(X,y,fs,w,eps=eps,witness_pool=witness_pool)
    if len(fs)==X.shape[1] or len(np.unique(y))<2:
        return {**old,'mixture_anchor_bound':0.,'mixture_lp_calls':0,'mixture_lp_failures':0}
    lo,hi=interval_bounds(X,eps);lb,ub=_interval_distances(lo,hi,fs,w)
    full=_distance(X,np.arange(X.shape[1]),w);values=[];calls=failures=0
    for i in range(len(y)):
        enemies=np.flatnonzero(y!=y[i]);cap=ub[i,enemies].min()
        possible=enemies[lb[i,enemies]<=cap+1e-12]
        witnesses=enemies[np.lexsort((enemies,full[i,enemies]))]
        if witness_pool is not None:witnesses=witnesses[:witness_pool]
        bounds=[]
        for j in possible:
            if lb[i,j]>=1:bounds.append(0.);continue
            signed,success=mixture_signed_bound(lo,hi,i,j,witnesses,fs,w)
            calls+=1;failures+=not success
            bounds.append(max(0.,min(1-lb[i,j],signed)))
        values.append(min(old['shared_object_bounds'][i],max(bounds)))
    raw=float(np.mean(values));bound=float(np.clip(raw+1e-8*(1+abs(raw)),0,1))
    return {**old,'upper_bound':min(old['upper_bound'],bound),'mixture_anchor_bound':bound,
            'mixture_object_bounds':values,'mixture_lp_calls':calls,'mixture_lp_failures':int(failures),
            'scope':'finite_shared_box_dependency_loss_convex_witness_relaxation_not_test_risk'}
