"""Training-internal RBF teacher risk of center-imputed attribute supports.

Public precomputed-kernel SVC avoids approximate feature-space distillation.
The loss is adaptively selected training evidence, not generalization bound.
"""
import numpy as np
from sklearn.svm import SVC
from .nested_teacher_regularized import inner_splits
from .validated_pool import preprocess_fit


def class_scores(model,kernel):
    scores = model.decision_function(kernel)
    return np.column_stack([np.zeros(len(kernel)),scores]) if scores.ndim == 1 else scores


class CrossfitKernelProjection:
    def __init__(self,raw,y,*,C=1.,seed=20261020):
        self.raw,self.y = np.asarray(raw,float),np.asarray(y); self.d = self.raw.shape[1]; self.C,self.seed = C,seed
        self.folds = []; self.cache = {}
        for a,b in inner_splits(self.y,seed=seed):
            A,B = preprocess_fit(self.raw[a],self.raw[b]); train,truth = self.y[a],self.y[b]; q = len(np.unique(train))
            gamma = 1/(self.d*A.var()) if A.var() > 0 else 1.
            distances = np.sum((A[:,None,:]-A[None,:,:])**2,axis=2)
            model = SVC(C=C,kernel='precomputed',class_weight='balanced',decision_function_shape='ovr').fit(np.exp(-gamma*distances),train)
            mass = np.array([1/(q*sum(train == c)) for c in train]); center = mass@A
            components = (B[:,None,:]-A[None,:,:])**2
            replaced = (center[None,:]-A)**2
            delta = replaced[None,:,:]-components; full = components.sum(axis=2)
            original = class_scores(model,np.exp(-gamma*full)); original -= original.mean(axis=1,keepdims=True)
            balance = np.array([1/(len(np.unique(truth))*sum(truth == c)) for c in truth])
            self.folds.append({'reference_indices':a.tolist(),'validation_indices':b.tolist(),'A':A,'B':B,
                'train':train,'truth':truth,'model':model,'gamma':gamma,'center':center,
                'full_squared_distances':full,'delta':delta,'original_scores':original,'balance':balance})

    def __call__(self,features):
        fs = tuple(sorted(features))
        if len(set(fs)) != len(fs) or any(int(a) != a or not 0 <= a < self.d for a in fs): raise ValueError('invalid support')
        if fs not in self.cache:
            omitted = [a for a in range(self.d) if a not in fs]; errors,mse = [],[]
            for fold in self.folds:
                squared = fold['full_squared_distances']+fold['delta'][:,:,omitted].sum(axis=2)
                kernel = np.exp(-fold['gamma']*np.maximum(squared,0)); pred = fold['model'].predict(kernel)
                errors.append(float(fold['balance']@(pred != fold['truth'])))
                scores = class_scores(fold['model'],kernel); scores -= scores.mean(axis=1,keepdims=True)
                mse.append(float(fold['balance']@np.mean((scores-fold['original_scores'])**2,axis=1)))
            self.cache[fs] = {'balanced_error':float(np.mean(errors)),'score_mse':float(np.mean(mse)),
                'fold_balanced_error':errors,'fold_score_mse':mse}
        return self.cache[fs]


def certified_kernel_swaps(initial,risk,candidate,*,pool=8,rounds=2):
    current = sorted(initial); certificate = candidate(current)
    if not certificate['certificate']['feasible']: raise ValueError('certified starting support required')
    def key(value): return (round(value['balanced_error'],12),round(value['score_mse'],12))
    initial_risk = risk(current); value = initial_risk; stages = []; attempts = []
    for step in range(rounds):
        choices = []
        for remove in current:
            for add in range(risk.d):
                if add in current: continue
                fs = sorted([a for a in current if a != remove]+[add]); score = risk(fs)
                if key(score) < key(value): choices.append((key(score),tuple(fs),remove,add,score))
        choices.sort(key=lambda row:(row[0],row[1])); stages.append({'round':step,'features':current.copy(),'risk':value,
            'improving_candidates':len(choices),'top_pool':[{'features':list(fs),'remove':remove,'add':add,'risk':score} for _,fs,remove,add,score in choices[:pool]]})
        accepted = None
        for _,fs,remove,add,score in choices[:pool]:
            geometry = candidate(list(fs)); ok = geometry['certificate']['feasible']
            attempts.append({'round':step,'remove':remove,'add':add,'risk':score,'candidate':geometry,'accepted':ok})
            if ok: accepted = list(fs); value = score; certificate = geometry; break
        if accepted is None: break
        current = accepted
    assert len(current) == len(initial) and certificate['certificate']['feasible'] and key(value) <= key(initial_risk)
    return {'features':current,'candidate':certificate,'initial_features':sorted(initial),'initial_risk':initial_risk,
        'final_risk':value,'stages':stages,'attempts':attempts,'pool':pool,'rounds':rounds,'C':risk.C,'seed':risk.seed,
        'scope':'same_count_original_certificate_training_adaptive_kernel_risk_descent_not_test_advantage'}
