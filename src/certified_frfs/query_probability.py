"""Smooth robust query-neighbor probability loss, adaptive training only.

Disjoint reference/query folds; class-balanced negative log own-class soft
neighbor mass. Same-class distance upper/enemy lower bounds give a lower
bound on stochastic neighbor mass, not hard5NN or test-risk certification.
"""
import numpy as np
from scipy.special import logsumexp
from sklearn.model_selection import StratifiedKFold
from .validated_pool import preprocess_fit
from .core import interval_bounds
from .sparse_neighbor import optimize_cardinality_objective


class QueryProbabilityObjective:
    def __init__(self,X,y,*,ridge=.01,temperature=.05,eps=.01,seed=1013,geometry='minmax'):
        X=np.asarray(X,float);y=np.asarray(y)
        if X.ndim!=2 or y.shape!=(len(X),) or ridge<0 or temperature<=0:raise ValueError('invalid inputs')
        labels,counts=np.unique(y,return_counts=True);ids=np.flatnonzero(np.isin(y,labels[counts>1]));single=np.flatnonzero(np.isin(y,labels[counts==1]))
        if not len(ids):raise ValueError('repeated class required')
        self.ridge=ridge;self.temperature=temperature;self.terms=[];self.splits=[];query_labels=[]
        for a,b in StratifiedKFold(min(3,int(counts[counts>1].min())),shuffle=True,random_state=seed).split(ids,y[ids]):
            ii=np.r_[ids[a],single];jj=ids[b];A,B=preprocess_fit(X[ii],X[jj],geometry=geometry);al,ah=interval_bounds(A,eps);bl,bh=interval_bounds(B,eps)
            same=y[jj,None]==y[ii][None,:]
            low=np.maximum(0,np.maximum(bl[:,None,:]-ah[None,:,:],al[None,:,:]-bh[:,None,:]))
            high=np.maximum(abs(bl[:,None,:]-ah[None,:,:]),abs(bh[:,None,:]-al[None,:,:]))
            C=np.where(same[:,:,None],high,low)
            self.terms.append((C,same,y[jj]));query_labels.extend(y[jj].tolist())
            self.splits.append({'reference':ii.tolist(),'query':jj.tolist()})
        queries=np.array(query_labels);classes=np.unique(queries)
        self.balance={q:1/(len(classes)*sum(queries==q)) for q in classes}

    def __call__(self,w):
        value=self.ridge*np.sum(w*w);gradient=2*self.ridge*w
        for C,same,labels in self.terms:
            z=-np.einsum('ijd,d->ij',C,w)/self.temperature
            norm=logsumexp(z,axis=1);positive=logsumexp(np.where(same,z,-np.inf),axis=1)
            pa=np.exp(z-norm[:,None]);ps=np.exp(np.where(same,z,-np.inf)-positive[:,None])
            alpha=np.array([self.balance[q] for q in labels])
            value+=alpha@(norm-positive)
            gradient+=np.einsum('i,ij,ijd->d',alpha,ps-pa,C)/self.temperature
        return float(value),gradient


def refit_probability_support(X,y,candidate,*,objective=None):
    objective=objective or QueryProbabilityObjective(X,y)
    d=np.asarray(X).shape[1];fs=np.asarray(candidate['features'],int);w=np.asarray(candidate['weights'],float)
    if not len(fs) or len(set(fs))!=len(fs) or np.any(fs<0) or np.any(fs>=d) or w.shape!=(d,) or np.any(w<0) or not np.isfinite(w).all() or w[fs].sum()<=0:raise ValueError('invalid initial support/weights')
    initial=w[fs]/w[fs].sum()
    def reduced(v):
        wide=np.zeros(d);wide[fs]=v;value,gradient=objective(wide);return value,gradient[fs]
    v,fit=optimize_cardinality_objective(reduced,initial,len(fs),ridge=objective.ridge)
    wide=np.zeros(d);wide[fs]=v
    return {**candidate,'features':np.flatnonzero(wide>0).tolist(),'weights':wide.tolist(),'fit':fit,
        'allowed_support':fs.tolist(),'training_splits':objective.splits,'temperature':objective.temperature,
        'scope':'training_crossfit_query_probability_not_hard_vote_or_test_risk'}
