"""Actual support-specific RBF refits, with componentwise fold descent.

The folds are adaptively reused and the FRS feasibility screen uses the whole
reference. These are selection scores, never unbiased generalization estimates.
"""
import numpy as np
from sklearn.svm import SVC
from .nested_teacher_regularized import inner_splits
from .validated_pool import preprocess_fit


def balanced_hinge(model, kernel, truth, balance):
    score = model.decision_function(kernel)
    target = np.searchsorted(model.classes_, truth)
    if score.ndim == 1:
        margin = (2*target-1)*score
    else:
        other = score.copy(); other[np.arange(len(truth)), target] = -np.inf
        margin = score[np.arange(len(truth)), target]-other.max(axis=1)
    return float(balance@np.maximum(1-margin, 0))


class CrossfitRBFRefit:
    def __init__(self, raw, y, *, C=1., seed=20261020):
        self.raw, self.y = np.asarray(raw, float), np.asarray(y)
        self.d, self.C, self.seed = self.raw.shape[1], C, seed
        self.cache, self.folds = {}, []
        for a,b in inner_splits(self.y, seed=seed):
            A,B = preprocess_fit(self.raw[a], self.raw[b])
            truth = self.y[b]
            balance = np.array([1/(len(np.unique(truth))*sum(truth == c)) for c in truth])
            self.folds.append({'reference_indices':a.tolist(), 'validation_indices':b.tolist(),
                'A':A, 'B':B, 'train':self.y[a], 'truth':truth, 'balance':balance,
                'train_components':(A[:,None,:]-A[None,:,:])**2,
                'query_components':(B[:,None,:]-A[None,:,:])**2})

    def __call__(self, features):
        fs = tuple(sorted(features))
        if not fs or len(set(fs)) != len(fs) or any(int(a) != a or not 0 <= a < self.d for a in fs):
            raise ValueError('nonempty valid support required')
        if fs not in self.cache:
            errors, hinges, fits = [], [], []
            for fold in self.folds:
                A = np.ascontiguousarray(fold['A'][:,fs])
                gamma = 1/(len(fs)*A.var()) if A.var() > 0 else 1.
                training = np.exp(-gamma*fold['train_components'][:,:,fs].sum(axis=2))
                query = np.exp(-gamma*fold['query_components'][:,:,fs].sum(axis=2))
                model = SVC(C=self.C, kernel='precomputed', class_weight='balanced',
                    decision_function_shape='ovr').fit(training, fold['train'])
                if model.fit_status_ != 0: raise RuntimeError('nonstationary RBF fit')
                pred = model.predict(query)
                errors.append(float(fold['balance']@(pred != fold['truth'])))
                hinges.append(balanced_hinge(model, query, fold['truth'], fold['balance']))
                fits.append({'gamma':float(gamma), 'fit_status':int(model.fit_status_),
                    'n_iter':model.n_iter_.tolist(), 'predictions':pred.tolist()})
            self.cache[fs] = {'balanced_error':float(np.mean(errors)), 'balanced_hinge':float(np.mean(hinges)),
                'fold_balanced_error':errors, 'fold_balanced_hinge':hinges, 'fold_fits':fits}
        return self.cache[fs]


def refit_risk_key(value):
    return (round(value['balanced_error'],12), round(value['balanced_hinge'],12))


def foldwise_improves(value, current):
    return (round(value['balanced_error'],12) < round(current['balanced_error'],12)
        and all(round(a,12) <= round(b,12) for a,b in
            zip(value['fold_balanced_error'],current['fold_balanced_error'])))


def certified_refit_swaps(initial, risk, candidate, *, pool=8, rounds=2):
    current = sorted(initial); certificate = candidate(current)
    if not certificate['certificate']['feasible']: raise ValueError('certified starting support required')
    value = initial_risk = risk(current); stages, attempts = [], []
    for step in range(rounds):
        choices = []; mean_improving = 0
        for remove in current:
            for add in range(risk.d):
                if add in current: continue
                fs = sorted([a for a in current if a != remove]+[add]); score = risk(fs)
                mean_improving += round(score['balanced_error'],12) < round(value['balanced_error'],12)
                if foldwise_improves(score,value): choices.append((refit_risk_key(score),tuple(fs),remove,add,score))
        choices.sort(key=lambda row:(row[0],row[1]))
        stages.append({'round':step,'features':current.copy(),'risk':value,
            'mean_improving_candidates':int(mean_improving),'foldwise_improving_candidates':len(choices),
            'top_pool':[{'features':list(fs),'remove':remove,'add':add,'risk':score} for _,fs,remove,add,score in choices[:pool]]})
        accepted = None
        for _,fs,remove,add,score in choices[:pool]:
            geometry = candidate(list(fs)); ok = geometry['certificate']['feasible']
            attempts.append({'round':step,'remove':remove,'add':add,'risk':score,'candidate':geometry,'accepted':ok})
            if ok: accepted = list(fs); value = score; certificate = geometry; break
        if accepted is None: break
        current = accepted
    assert len(current) == len(initial) and certificate['certificate']['feasible']
    assert refit_risk_key(value) <= refit_risk_key(initial_risk)
    return {'features':current,'candidate':certificate,'initial_features':sorted(initial),'initial_risk':initial_risk,
        'final_risk':value,'stages':stages,'attempts':attempts,'pool':pool,'rounds':rounds,'C':risk.C,'seed':risk.seed,
        'scope':'same_count_original_FRS_foldwise_RBF_refit_descent_adaptive_internal_scores'}
