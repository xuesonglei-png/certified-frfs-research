"""Subfold-local distance-transport predictive objective for future searches."""
import numpy as np
from .redistributed_search import CrossfitRedistributedUtilityWorkspace
from .metric_transport import fit_transport_weights


class CrossfitTransportUtility:
    def __init__(self,X,y,*,family='uniform',eps=.01,seed=101):
        source=CrossfitRedistributedUtilityWorkspace(X,y,family=family,eps=eps,seed=seed)
        self.splits=source.splits;self.folds=source.folds;self.cache={};self.fit_records={}
        self.eps=eps;self.seed=seed

    def evaluate(self,features):
        fs=tuple(features)
        if fs not in self.cache:
            scores=[];fits=[]
            for index,(A,B,yt,yv,w) in enumerate(self.folds):
                v,info=fit_transport_weights(A,fs,w)
                rng=np.random.default_rng(self.seed+43+index)
                Z=np.clip(A+rng.uniform(-self.eps,self.eps,A.shape),0,1)
                Q=np.clip(B+rng.uniform(-self.eps,self.eps,B.shape),0,1)
                labels=np.unique(yt);values=[]
                for ref,query in [(A,B),(Z,Q)]:
                    D=abs(query[:,None,fs]-ref[None,:,fs])@v
                    nn=np.argsort(D,axis=1,kind='stable')[:,:min(5,len(yt))]
                    votes=np.stack([(yt[nn]==q).sum(axis=1) for q in labels],axis=1)
                    pred=labels[votes.argmax(axis=1)]
                    values.append(float(np.mean([(pred[yv==q]==q).mean() for q in np.unique(yv)])))
                scores.append(values);fits.append({'inner':index,'weights':v.tolist(),'fit':info})
            clean,noisy=np.mean(scores,axis=0);self.fit_records[fs]=fits
            self.cache[fs]={'clean_LOO':float(clean),'shared_noise_LOO':float(noisy),
                'utility':float((clean+noisy)/2),
                'score_scope':'adaptive_training_crossfit_distance_transport_not_generalization'}
        return self.cache[fs]
