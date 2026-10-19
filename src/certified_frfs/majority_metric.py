"""Development robust 5NN metric and protected finite-table majority paths.

This changes the certified object; it is not an improved dependency-loss bound.
All weights and fixed friend pools are learned on the reference training table.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.spatial.distance import cdist
from .core import interval_bounds
from .margin import triplet_interval_min


class MajorityTerms:
    def __init__(self,X,y,*,eps=.01,weights=None):
        self.X=np.asarray(X,float);self.y=np.asarray(y)
        n,d=self.X.shape
        if not np.isfinite(self.X).all() or self.y.shape!=(n,):raise ValueError('invalid table')
        self.w=np.ones(d)/np.sqrt(d) if weights is None else np.asarray(weights,float)
        if self.w.shape!=(d,) or np.any(self.w<0) or not np.isfinite(self.w).all():raise ValueError('invalid weights')
        lo,hi=interval_bounds(self.X,eps)
        distance=cdist(self.X*self.w,self.X*self.w,'cityblock');np.fill_diagonal(distance,np.inf)
        self.cross=self.y[:,None]!=self.y[None,:]
        self.applicable=np.array([np.sum(self.y==label)>=4 for label in self.y])
        friends=[]
        for i,label in enumerate(self.y):
            ids=np.flatnonzero((self.y==label)&(np.arange(n)!=i))
            ids=ids[np.lexsort((ids,distance[i,ids]))][:3]
            friends.append(np.resize(ids,3) if len(ids) else np.repeat(i,3))
        friends=np.asarray(friends)
        self.terms=np.empty((d,n,n,3))
        for a in range(d):
            self.terms[a]=triplet_interval_min(lo[:,a,None,None],hi[:,a,None,None],
                lo[None,:,a,None],hi[None,:,a,None],
                lo[friends,a][:,None,:],hi[friends,a][:,None,:])

    def bounds(self,features,*,allowed_enemies=0):
        if int(allowed_enemies)!=allowed_enemies or not 0<=allowed_enemies<=2:
            raise ValueError('5NN majority allows at most two preceding enemies')
        terms=np.einsum('a,aijf->ijf',self.w[list(features)],self.terms[list(features)],optimize=True)
        # All three witnesses precede every enemy except at most this budget.
        enemy_bounds=np.where(self.cross,terms.min(axis=2),np.inf)
        result=np.partition(enemy_bounds,int(allowed_enemies),axis=1)[:,int(allowed_enemies)]
        result[~self.applicable]=-np.inf
        return result-1e-10*(1+np.abs(np.where(np.isfinite(result),result,0)))


def learn_majority_weights(X,y,*,eps=.01,reg=.01,temperature=.03,margin=.02):
    X,y=np.asarray(X,float),np.asarray(y);d=X.shape[1]
    workspace=MajorityTerms(X,y,eps=eps)
    ids=np.flatnonzero(workspace.applicable)
    if not len(ids):return np.ones(d)/np.sqrt(d)
    coefficients=[];groups=[]
    for i in ids:
        enemies=np.flatnonzero(workspace.cross[i])
        if not len(enemies):continue
        coefficients.append(workspace.terms[:,i,enemies,:].reshape(d,-1).T)
        groups.append(y[i])
    if not coefficients:return np.ones(d)/np.sqrt(d)
    labels=np.asarray(groups);classes=np.unique(labels)
    balance=np.array([1/(len(classes)*np.sum(labels==label)) for label in labels])
    def objective(w):
        loss=0.;grad=np.zeros(d)
        for alpha,C in zip(balance,coefficients):
            z=(margin-C@w)/temperature
            # Smooth maximum over enemies/friends plus the zero hinge branch.
            norm=logsumexp(np.r_[0.,z])
            loss+=alpha*temperature*norm
            grad-=alpha*(np.exp(z-norm)@C)
        loss+=reg*np.sum(w*np.log(d*w))
        grad+=reg*(np.log(d*w)+1)
        return float(loss),grad
    result=minimize(objective,np.ones(d)/d,jac=True,method='SLSQP',
        bounds=[(1e-5,None)]*d,constraints={'type':'eq','fun':lambda w:w.sum()-1,
        'jac':lambda w:np.ones(d)},options={'maxiter':200,'ftol':1e-9})
    w=result.x if result.success else np.ones(d)/d
    w=np.maximum(w,1e-5);return w/np.linalg.norm(w)


def protected_majority_path(X,y,ranking,weights,*,retention=.9,eps=.01,allowed_enemies=0):
    if not 0<retention<=1:raise ValueError('invalid retention')
    workspace=MajorityTerms(X,y,weights=weights,eps=eps)
    d=np.asarray(X).shape[1]
    if sorted(ranking)!=list(range(d)):raise ValueError('invalid ranking')
    current=list(range(d));protected=workspace.bounds(current,allowed_enemies=allowed_enemies)>1e-10
    labels=np.unique(y);groups=[(np.asarray(y)==label)&protected for label in labels]
    required=[int(np.ceil(retention*g.sum()-1e-12)) for g in groups]
    candidates=[current.copy()];trace=[]
    if protected.any():
        for a in ranking:
            if len(current)==1:break
            fs=[b for b in current if b!=a];bounds=workspace.bounds(fs,allowed_enemies=allowed_enemies)
            counts=[int(np.sum((bounds>1e-10)&g)) for g in groups]
            feasible=all(c>=r for c,r in zip(counts,required))
            trace.append({'feature':int(a),'features':fs,'feasible':feasible,'retained':counts})
            if feasible:current=fs;candidates.append(current.copy())
    return {'candidates':candidates,'trace':trace,'required':required,
        'protected_counts':[int(g.sum()) for g in groups],'allowed_enemies':allowed_enemies,
        'scope':'per_class_retention_of_certified_correct_training_LOO_5NN_objects_not_test_risk'}
