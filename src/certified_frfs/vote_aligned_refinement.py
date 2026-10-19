"""Certified local refinement using soft top-five class counts.

The implicit threshold maintains five units of neighbor membership. This is
an optimization surrogate, not a certificate of classification accuracy.
"""
import numpy as np
from scipy.special import expit, logsumexp
from .query_probability import QueryProbabilityObjective
from .validated_pool import preprocess_fit, validation_scores
from .bound_direct import _project
from .feasible_transport import weight_certificate


def soft_topk_loss(C, labels, truth, w, *, temperature=.01, k=5, vote_temperature=.5):
    distance = np.einsum('ijd,d->ij', C, w)
    k = min(k, distance.shape[1]-1)
    if k < 1:
        raise ValueError('at least two reference objects required')
    lo = distance.min(axis=1)-40*temperature
    hi = distance.max(axis=1)+40*temperature
    for _ in range(60):
        threshold = (lo+hi)/2
        membership = expit((threshold[:, None]-distance)/temperature)
        above = membership.sum(axis=1) > k
        hi = np.where(above, threshold, hi)
        lo = np.where(above, lo, threshold)
    membership = expit(((lo+hi)[:, None]/2-distance)/temperature)
    classes = np.unique(labels)
    indicator = (labels[:, None] == classes[None, :]).astype(float)
    votes = membership@indicator/vote_temperature
    norm = logsumexp(votes, axis=1)
    target = truth[:, None] == classes[None, :]
    if not np.all(target.sum(axis=1) == 1):
        raise ValueError('query class absent from reference')
    probabilities = np.exp(votes-norm[:, None])
    alpha = np.array([1/(len(np.unique(truth))*sum(truth == q)) for q in truth])
    loss = alpha@(norm-np.sum(votes*target, axis=1))
    dm = ((probabilities-target)@indicator.T)/vote_temperature
    slope = membership*(1-membership)/temperature
    # Differentiate the implicit constraint sum(membership) = k.
    centered = dm-np.sum(dm*slope, axis=1)[:, None]/np.maximum(slope.sum(axis=1)[:, None], 1e-30)
    gradient = -np.einsum('i,ij,ijd->d', alpha, slope*centered, C)
    return float(loss), gradient


class VoteAlignedObjective:
    def __init__(self, X, y):
        self.probability = QueryProbabilityObjective(X, y)
        self.splits = self.probability.splits
        self.y = np.asarray(y)
        self.folds = []
        for split in self.splits:
            ti, qi = np.array(split['reference']), np.array(split['query'])
            A, B = preprocess_fit(np.asarray(X)[ti], np.asarray(X)[qi])
            self.folds.append((A, self.y[ti], B, self.y[qi]))

    def __call__(self, w):
        value, gradient = self.probability(w)
        value *= .1
        gradient *= .1
        for (C, _, truth), split in zip(self.probability.terms, self.splits):
            v, g = soft_topk_loss(C, self.y[np.array(split['reference'])], truth, w)
            value += v/len(self.splits)
            gradient += g/len(self.splits)
        return value, gradient

    def hard_utility(self, w):
        fs = np.flatnonzero(w > 0).tolist()
        metrics = [validation_scores(A, y, B, truth, fs, w, 31001+i, joint=True)
                   for i, (A, y, B, truth) in enumerate(self.folds)]
        return float(np.mean([(m['clean']+m['noise01'])/2 for m in metrics]))


def refine_vote_aligned(X, y, initial, *, steps=10, backtracks=12):
    X, y = np.asarray(X, float), np.asarray(y)
    d = X.shape[1]; full = np.ones(d)/np.sqrt(d); mass = full.sum()
    fs = np.asarray(initial['features'], int); wide = np.asarray(initial['weights'], float)
    if wide.shape != (d,) or not len(fs) or len(set(fs)) != len(fs) or np.any(fs < 0) or np.any(fs >= d):
        raise ValueError('invalid support')
    if steps < 1 or backtracks < 1 or not np.isfinite(wide).all() or np.any(wide < 0) or abs(wide.sum()-mass) > 1e-8 or np.any(wide > 2*mass/d+1e-8):
        raise ValueError('invalid mass, cap or budget')
    if set(np.flatnonzero(wide > 0)) != set(fs):
        raise ValueError('support mismatch')
    A, _ = preprocess_fit(X, X); obj = VoteAlignedObjective(X, y); w = wide/mass
    def cert(v):
        ids = np.flatnonzero(v > 0).tolist()
        return weight_certificate(A, y, ids, full, v[ids]*mass)
    certificate = cert(w)
    if not certificate['feasible']:
        raise ValueError('initial metric not certified')
    value, gradient = obj(w); utility = obj.hard_utility(w)
    trace = [{'iteration': 0, 'weights': wide.tolist(), 'objective': value,
              'training_utility': utility, 'certificate': certificate}]; attempts = []
    for iteration in range(1, steps+1):
        accepted = None
        for exponent in range(backtracks):
            step = .05*2.**(-exponent); candidate = np.zeros(d)
            candidate[fs] = _project(w[fs]-step*gradient[fs], 2/d)
            nv, ng = obj(candidate)
            record = {'iteration': iteration, 'exponent': exponent, 'objective': nv}
            if nv < value-1e-9:
                nu = obj.hard_utility(candidate); record['training_utility'] = nu
                if nu >= utility-1e-12:
                    nc = cert(candidate); record['certificate'] = nc
                    if nc['feasible']:
                        accepted = candidate, nv, ng, nu, nc
            attempts.append(record)
            if accepted is not None:
                break
        if accepted is None:
            break
        w, value, gradient, utility, certificate = accepted
        trace.append({'iteration': iteration, 'weights': (w*mass).tolist(), 'objective': value,
                      'training_utility': utility, 'certificate': certificate, 'step': step})
    return {'features': np.flatnonzero(w > 0).tolist(), 'weights': (w*mass).tolist(),
            'certificate': certificate, 'trace': trace, 'attempts': attempts,
            'training_splits': obj.splits, 'steps': steps, 'backtracks': backtracks,
            'scope': 'adaptive_training_vote_aligned_certified_refinement_not_test_risk'}
