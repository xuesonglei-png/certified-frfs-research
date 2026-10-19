"""Cached subfold-local transferred-metric utility with declared stable ties.

Training correlations and unweighted coordinate distance components are cached
per reference subfold. This is a new implementation; roundoff can change
near-tie neighbors, so old paths must not be reused without comparison.
"""
import numpy as np
from .redistributed_search import CrossfitRedistributedUtilityWorkspace
from .cached_transfer import CachedTransfer


class CachedCrossfitTransferUtility:
    def __init__(self,X,y,*,family='uniform',eps=.01,seed=101):
        source=CrossfitRedistributedUtilityWorkspace(X,y,family=family,eps=eps,seed=seed)
        self.splits=source.splits;self.cache={};self.frames=[]
        for index,(A,B,yt,yv,w) in enumerate(source.folds):
            rng=np.random.default_rng(seed+43+index)
            Z=np.clip(A+rng.uniform(-eps,eps,A.shape),0,1)
            Q=np.clip(B+rng.uniform(-eps,eps,B.shape),0,1)
            components=[np.abs(query[:,None,:]-ref[None,:,:]) for ref,query in [(A,B),(Z,Q)]]
            self.frames.append((CachedTransfer(A,w),components,yt,yv))

    def evaluate(self,features):
        fs=tuple(features)
        if fs not in self.cache:
            values=[];weights=[]
            for transfer,components,yt,yv in self.frames:
                wr=transfer.weights(fs);labels=np.unique(yt);scores=[]
                for c in components:
                    D=c[:,:,fs]@wr
                    ids=np.argsort(D,axis=1,kind='stable')[:,:min(5,len(yt))]
                    votes=np.stack([(yt[ids]==label).sum(axis=1) for label in labels],axis=1)
                    prediction=labels[votes.argmax(axis=1)]
                    scores.append(float(np.mean([(prediction[yv==q]==q).mean() for q in np.unique(yv)])))
                values.append(scores);weights.append(wr)
            clean,noisy=np.mean(values,axis=0)
            self.cache[fs]={'clean_LOO':float(clean),'shared_noise_LOO':float(noisy),
                'utility':float((clean+noisy)/2),'reduced_weights':np.mean(weights,axis=0).tolist(),
                'score_scope':'adaptive_training_crossfit_subset_specific_transfer'}
        return self.cache[fs]
