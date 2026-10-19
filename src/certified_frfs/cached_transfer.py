"""Training-fitted cache for the unchanged squared-correlation transfer rule.

One cache per training table. A cache must never be fitted on validation or
test observations. This changes computation only, not the transfer model.
"""
import numpy as np


class CachedTransfer:
    def __init__(self,X,weights,*,power=2.,floor=1e-12):
        X=np.asarray(X,float);w=np.asarray(weights,float)
        if X.ndim!=2 or X.shape[0]<1 or X.shape[1]<1 or not np.isfinite(X).all():
            raise ValueError('finite nonempty training table required')
        if w.shape!=(X.shape[1],) or np.any(w<0) or not np.isfinite(w).all():
            raise ValueError('finite nonnegative weights required')
        if not np.isfinite(power) or power<=0 or not np.isfinite(floor) or floor<=0:
            raise ValueError('positive power and floor required')
        self.w=w.copy();self.d=X.shape[1];self.floor=floor
        self.similarity=np.zeros((self.d,self.d));valid=np.std(X,axis=0)>floor
        ids=np.flatnonzero(valid)
        if len(ids)>1:
            C=np.corrcoef(X[:,ids],rowvar=False)
            self.similarity[np.ix_(ids,ids)]=abs(C)**power
        elif len(ids)==1:self.similarity[ids[0],ids[0]]=1.

    def weights(self,features):
        raw=np.asarray(list(features));fs=raw.astype(int)
        if not len(fs) or np.any(raw!=fs) or len(np.unique(fs))!=len(fs) or np.any(fs<0) or np.any(fs>=self.d):
            raise ValueError('nonempty valid distinct integer subset required')
        omitted=np.setdiff1d(np.arange(self.d),fs)
        if not len(omitted):return self.w[fs].copy()
        scores=self.similarity[np.ix_(fs,omitted)].copy();total=scores.sum(axis=0)
        fallback=total<=self.floor;scores[:,fallback]=1.;total[fallback]=len(fs)
        return self.w[fs]+(scores/total)@self.w[omitted]
