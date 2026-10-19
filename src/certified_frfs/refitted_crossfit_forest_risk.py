"""Actually refit a support-local classifier on each reference-internal fold.

These scores guide adaptive support search, not independent risk estimation.
Original outer/inner query data are absent from this API.
"""
import numpy as np
from scipy.special import logsumexp
from .nested_teacher_regularized import inner_splits
from .validated_pool import preprocess_fit
from .cached_complementarity_forest import ComplementarityForestCache
from .orthogonal_forest_logistic import fit_orthogonal_forest,orthogonal_scores


class RefittedCrossfitForestRisk:
    def __init__(self,raw,y,*,C=1.,pair_penalty=1.,seed=20261014):
        raw,y = np.asarray(raw,float),np.asarray(y); self.C = C; self.pair_penalty = pair_penalty
        self.folds = []; self.records = []; self.cache = {}; self.models = []; self.scores = []
        for a,b in inner_splits(y,seed=seed):
            A,B = preprocess_fit(raw[a],raw[b])
            forest = None if pair_penalty is None else ComplementarityForestCache(A,y[a])
            self.folds.append((A,B,y[a],y[b],forest))
            self.records.append({'reference_indices':a.tolist(),'validation_indices':b.tolist()})

    def __call__(self,features):
        fs = tuple(sorted(features))
        if fs not in self.cache:
            values = []; mids = []
            for j,(A,B,train,truth,forest) in enumerate(self.folds):
                edges = [] if forest is None else forest.edges(fs)
                fit = fit_orthogonal_forest(A,train,list(fs),edges,C=self.C,pair_penalty=1. if self.pair_penalty is None else self.pair_penalty)
                scores = orthogonal_scores(B,fit); classes = np.array(fit['classes'])
                own = np.argmax(truth[:,None] == classes[None],axis=1)
                balance = np.array([1/(len(np.unique(truth))*sum(truth == c)) for c in truth])
                value = float(balance @ (logsumexp(scores,axis=1)-scores[np.arange(len(truth)),own]))
                mids.append(len(self.models)); self.models.append({'fold_index':j,'fit':fit,'balanced_logloss':value})
                values.append(value)
            index = len(self.scores); record = {'features':list(fs),'fold_model_ids':mids,'mean_balanced_logloss':float(np.mean(values))}
            self.scores.append(record); self.cache[fs] = index
        return self.scores[self.cache[fs]]
