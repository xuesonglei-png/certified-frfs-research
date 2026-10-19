"""Joint sparse diagonal metric learning by projected gradient.

Standard cardinality-constrained metric-learning prototype, not a novelty
claim or a fuzzy-rough certificate. Hard projection allows attributes to
leave and re-enter the support; objective descent is checked by backtracking.
"""
import numpy as np
from scipy.special import logsumexp
from .softvote_metric import softvote_terms


def sparse_simplex_projection(z, k):
    z = np.asarray(z, float)
    if z.ndim != 1 or not np.isfinite(z).all() or int(k) != k or not 1 <= k <= len(z):
        raise ValueError('finite vector and valid integer cardinality required')
    ids = np.argsort(-z, kind='stable')[:k]
    u = np.sort(z[ids])[::-1]; css = np.cumsum(u)-1
    rho = np.flatnonzero(u-css/np.arange(1, len(u)+1) > 0)[-1]
    theta = css[rho]/(rho+1)
    out = np.zeros(len(z)); out[ids] = np.maximum(z[ids]-theta, 0)
    return out


def sparse_neighbor_objective(w, terms, *, temperature=.05, ridge=.001):
    C, same, mask, labels = terms
    if not len(labels): return 0., np.zeros_like(w)
    classes = np.unique(labels)
    balance = np.array([1/(len(classes)*sum(labels == q)) for q in labels])
    z = -np.einsum('ijd,d->ij', C, w)/temperature
    za = np.where(mask, z, -np.inf); zs = np.where(same, z, -np.inf)
    norm = logsumexp(za, axis=1); positive = logsumexp(zs, axis=1)
    pa = np.exp(za-norm[:, None]); ps = np.exp(zs-positive[:, None])
    loss = balance@(norm-positive)+ridge*np.sum(w*w)
    grad = np.einsum('i,ij,ijd->d', balance, ps-pa, C)/temperature+2*ridge*w
    return float(loss), grad


def optimize_sparse_metric(terms, initial, k, *, maxiter=120, ridge=.001):
    if not np.isfinite(ridge) or ridge < 0: raise ValueError('nonnegative ridge required')
    return optimize_cardinality_objective(lambda w: sparse_neighbor_objective(w, terms, ridge=ridge),
                                          initial, k, maxiter=maxiter, ridge=ridge)


def optimize_cardinality_objective(objective, initial, k, *, maxiter=120, ridge=0.):
    w = sparse_simplex_projection(initial, k)
    loss, grad = objective(w)
    trace = [{'iteration': 0, 'objective': loss, 'support': np.flatnonzero(w > 0).tolist()}]
    status = 'maxiter'; step = .05
    for iteration in range(1, maxiter+1):
        accepted = False
        for _ in range(24):
            candidate = sparse_simplex_projection(w-step*grad, k)
            value, gradient = objective(candidate)
            if value <= loss-1e-4*np.sum((candidate-w)**2)/step+1e-12:
                accepted = True; break
            step *= .5
        if not accepted: status = 'line_search_stopped'; break
        delta = np.linalg.norm(candidate-w)
        w, loss, grad = candidate, value, gradient
        trace.append({'iteration': iteration, 'objective': loss, 'support': np.flatnonzero(w > 0).tolist()})
        if delta < 1e-7: status = 'stationary_projection'; break
        step = min(step*1.2, .2)
    return w, {'status': status, 'trace': trace, 'iterations': len(trace)-1,
               'requested_cardinality': int(k), 'actual_cardinality': int(sum(w > 0)), 'ridge': ridge,
               'scope': 'local_training_sparse_metric_no_global_optimality_or_certificate'}


def sparse_metric_candidates(X, y, *, eps=.01, fractions=(.35, .65, 1.), ridge=.001):
    X = np.asarray(X, float); y = np.asarray(y)
    if X.ndim != 2 or y.shape != (len(X),) or not np.isfinite(X).all(): raise ValueError('invalid table')
    terms = softvote_terms(X, y, eps=eps); d = X.shape[1]
    dense, full_fit = optimize_sparse_metric(terms, np.ones(d)/d, d, ridge=ridge)
    rows = []
    for fraction in fractions:
        if not 0 < fraction <= 1: raise ValueError('invalid support fraction')
        k = max(1, int(np.ceil(fraction*d)))
        w, fit = (dense.copy(), full_fit) if k == d else optimize_sparse_metric(terms, dense, k, ridge=ridge)
        rows.append({'config': {'eps': eps, 'fraction': fraction},
                     'features': np.flatnonzero(w > 0).tolist(), 'weights': w.tolist(), 'fit': fit})
    return rows
