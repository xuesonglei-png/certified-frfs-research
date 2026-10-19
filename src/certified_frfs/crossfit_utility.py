"""Deletion ranking by subfold-local predictive utility; no outer labels.

These internal folds are adaptively reused by search, so their scores are
search objectives, not unbiased performance estimates. Outer inner-validation
and held-out testing remain separate.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from .validated_pool import preprocess_fit
from .softvote_metric import learn_softvote_weights


class CrossfitUtilityWorkspace:
    def __init__(self, X, y, *, family='uniform', eps=.01, seed=101):
        X=np.asarray(X,float);self.y=np.asarray(y)
        if family not in ('uniform','softvote'):
            raise ValueError('unknown metric family')
        self.labels,self.codes=np.unique(self.y,return_inverse=True)
        counts=np.bincount(self.codes);ids=np.flatnonzero(counts[self.codes]>1)
        singletons=np.flatnonzero(counts[self.codes]==1)
        if len(ids)==0:raise ValueError('crossfit utility needs repeated classes')
        nfold=min(3,int(counts[counts>1].min()))
        self.folds=[];self.cache={};self.splits=[]
        for index,(ii,jj) in enumerate(StratifiedKFold(nfold,shuffle=True,random_state=seed+41).split(ids,self.y[ids])):
            tr=np.r_[ids[ii],singletons];va=ids[jj]
            A,B=preprocess_fit(X[tr],X[va])
            w=np.ones(X.shape[1])/np.sqrt(X.shape[1]) if family=='uniform' else learn_softvote_weights(A,self.y[tr])
            rng=np.random.default_rng(seed+43+index)
            Z=np.clip(A+rng.uniform(-eps,eps,A.shape),0,1)
            Q=np.clip(B+rng.uniform(-eps,eps,B.shape),0,1)
            components=[np.moveaxis(abs(q[:,None,:]-a[None,:,:])*w,2,0) for a,q in [(A,B),(Z,Q)]]
            truth=self.codes[va];present=np.unique(truth)
            balance=1/(len(present)*np.bincount(truth,minlength=len(self.labels))[truth])
            self.folds.append((components,self.codes[tr],truth,balance,min(5,len(tr))))
            self.splits.append({'train':tr.tolist(),'validation':va.tolist(),'weights':w.tolist()})

    def evaluate(self,features):
        fs=tuple(features)
        if not fs:raise ValueError('empty distance subset')
        if fs not in self.cache:
            values=[]
            for components,labels,truth,balance,k in self.folds:
                scores=[]
                for c in components:
                    D=c[list(fs)].sum(axis=0)
                    nn=np.argsort(D,axis=1,kind='stable')[:,:k]
                    votes=np.stack([(labels[nn]==label).sum(axis=1) for label in range(len(self.labels))],axis=1)
                    scores.append(float(balance@(votes.argmax(axis=1)==truth)))
                values.append(scores)
            clean,noisy=np.mean(values,axis=0)
            # Legacy keys preserve the path schema; scope explicitly identifies CV.
            self.cache[fs]={'clean_LOO':float(clean),'shared_noise_LOO':float(noisy),
                            'utility':float((clean+noisy)/2),'score_scope':'adaptive_training_crossfit'}
        return self.cache[fs]
