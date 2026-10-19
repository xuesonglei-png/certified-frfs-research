"""Crossfit query/reference majority objective learned within reference data.

Each object is a training query in exactly one fold. Its same-class witnesses
come from disjoint reference objects. This is an adaptive training objective,
not independent validation; external inner folds still select configurations.
"""
import numpy as np
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold
from .validated_pool import preprocess_fit
from .core import interval_bounds
from .margin import triplet_interval_min
from .sparse_neighbor import optimize_cardinality_objective


class QueryMajorityObjective:
    def __init__(self, X, y, *, eps=.01, ridge=.01, seed=1013, margin=.02, temperature=.03):
        X = np.asarray(X, float); y = np.asarray(y); labels, counts = np.unique(y, return_counts=True)
        if X.ndim != 2 or y.shape != (len(X),): raise ValueError('invalid table')
        ids=np.flatnonzero(np.isin(y,labels[counts>1]));single=np.flatnonzero(np.isin(y,labels[counts==1]))
        if not len(ids): raise ValueError('at least one repeated class required')
        self.ridge = ridge; self.margin = margin; self.temperature = temperature
        if ridge < 0 or temperature <= 0 or margin < 0: raise ValueError('invalid parameters')
        self.splits = []; self.contrasts = []; query_labels = []
        for i,j in StratifiedKFold(min(3,int(counts[counts>1].min())),shuffle=True,random_state=seed).split(ids,y[ids]):
            ii=np.r_[ids[i],single];jj=ids[j]
            A, B = preprocess_fit(X[ii], X[jj]); al, ah = interval_bounds(A, eps); bl, bh = interval_bounds(B, eps)
            self.splits.append({'reference': ii.tolist(), 'query': jj.tolist()})
            for q in range(len(jj)):
                friends = np.flatnonzero(y[ii] == y[jj[q]]); enemies = np.flatnonzero(y[ii] != y[jj[q]])
                if len(friends) < 3 or len(enemies) < 3: continue
                distances = abs(A[friends]-B[q]).sum(axis=1)
                friends = friends[np.lexsort((friends, distances))][:3]
                C = triplet_interval_min(bl[q][None, None, :], bh[q][None, None, :],
                    al[enemies][:, None, :], ah[enemies][:, None, :], al[friends][None, :, :], ah[friends][None, :, :])
                self.contrasts.append(C); query_labels.append(y[jj[q]])
        self.query_labels = np.asarray(query_labels); classes = np.unique(self.query_labels)
        self.balance = np.array([1/(len(classes)*sum(self.query_labels == q)) for q in self.query_labels])
        self.applicable_count = len(query_labels); self.d = X.shape[1]

    def __call__(self, w):
        loss = self.ridge*np.sum(w*w); gradient = 2*self.ridge*w
        for alpha, C in zip(self.balance, self.contrasts):
            values = C@w; friend = values.argmin(axis=1)
            enemy_values = values[np.arange(len(C)), friend]
            enemy = np.argsort(enemy_values, kind='stable')[2]; coefficients = C[enemy, friend[enemy]]
            z = (self.margin-enemy_values[enemy])/self.temperature
            loss += alpha*self.temperature*np.logaddexp(0, z)
            gradient -= alpha*expit(z)*coefficients
        return float(loss), gradient


def query_majority_candidates(X, y, *, ridge=.01, fractions=(.35, .65, 1.)):
    objective = QueryMajorityObjective(X, y, ridge=ridge); d = objective.d
    dense, full_fit = optimize_cardinality_objective(objective, np.ones(d)/d, d, ridge=ridge)
    rows = []
    for fraction in fractions:
        if not 0 < fraction <= 1: raise ValueError('invalid fraction')
        k = max(1, int(np.ceil(fraction*d)))
        w, fit = (dense.copy(), full_fit) if k == d else optimize_cardinality_objective(objective, dense, k, ridge=ridge)
        rows.append({'config': {'ridge': ridge, 'fraction': fraction}, 'features': np.flatnonzero(w > 0).tolist(),
            'weights': w.tolist(), 'fit': fit, 'training_splits': objective.splits,
            'applicable_query_count': objective.applicable_count,
            'scope': 'crossfit_training_queries_not_external_validation_or_test_risk'})
    return rows


def refit_query_support(X, y, candidate, *, ridge=.01, objective=None):
    """Refit a reference-trained support without reading validation labels."""
    objective = objective or QueryMajorityObjective(X, y, ridge=ridge)
    d = np.asarray(X).shape[1]; fs = np.asarray(candidate['features'], int)
    original = np.asarray(candidate['weights'], float)
    if original.shape != (d,) or np.any(original < 0) or not np.isfinite(original).all() or original.sum() <= 0:
        raise ValueError('invalid initial metric')
    if not len(fs) or len(set(fs)) != len(fs) or np.any(fs < 0) or np.any(fs >= d): raise ValueError('invalid support')
    initial = original[fs]/original[fs].sum()
    def reduced(v):
        wide = np.zeros(d); wide[fs] = v
        value, gradient = objective(wide)
        return value, gradient[fs]
    v, fit = optimize_cardinality_objective(reduced, initial, len(fs), ridge=ridge)
    w = np.zeros(d); w[fs] = v
    return {**candidate, 'weights': w.tolist(), 'features': np.flatnonzero(w > 0).tolist(),
        'fit': fit, 'initial_reference_fit': candidate['fit'], 'allowed_support': fs.tolist(),
        'training_splits': objective.splits, 'applicable_query_count': objective.applicable_count,
        'scope': 'reference_support_then_query_refit_inside_training_no_test_risk'}
