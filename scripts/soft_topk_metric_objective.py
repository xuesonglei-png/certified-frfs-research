"""Implicit soft-k-neighborhood objective, aligned with the deployed 5NN size.

A query-dependent threshold makes the neighbor weights sum to k. Robust
same-class upper distances and enemy lower distances pessimistically bound
this SOFT same-class neighbor mass, not the discrete hard-kNN vote or risk.
"""
import numpy as np
from scipy.special import expit


def topk_leakage_bound(gap, available, *, k=5, temperature=.05):
    """Cardinality/gap-specific SOFT-to-hard mass bound, ordinary float only.

    For m available distances with positive k/(k+1) gap g, leakage T solves
    T^2 <= exp(-g/tau)*(k-T)*(m-k-T). The nonnegative root bounds T.
    A class mass differs from its hard-k fraction by at most T/k. This is
    conditional on the supplied gap; it is not a learned-weight risk guarantee.
    """
    if (not np.isfinite(gap) or gap < 0 or int(available) != available
            or int(k) != k or not 1 <= k <= available or temperature <= 0):
        raise ValueError('invalid gap, cardinality or temperature')
    if available == k: return 0.
    a, b = k, available-k; h = np.exp(-gap/temperature)
    if h == 0: return 1e-12
    root = 2*h*a*b/(np.sqrt(h*h*available**2+4*h*(1-h)*a*b)+h*available)
    return float(min(a, b, root+1e-12))


def soft_topk_membership(distances, mask, *, k=5, temperature=.05, tolerance=1e-10, max_iter=80):
    D, mask = np.asarray(distances, float), np.asarray(mask, bool)
    if (D.ndim != 2 or mask.shape != D.shape or not np.isfinite(D).all()
            or int(k) != k or k < 1 or temperature <= 0 or tolerance <= 0 or max_iter < 1):
        raise ValueError('invalid distances, mask, k or solver parameters')
    m = mask.sum(axis=1)
    if np.any(m < k): raise ValueError('fewer available neighbors than requested k')
    Q = mask.astype(float)
    active = m > k
    if not np.any(active): return Q
    Z, M = -D[active]/temperature, mask[active]
    sizes = m[active]
    lower = -np.max(np.where(M, Z, -np.inf), axis=1)-40
    upper = -np.min(np.where(M, Z, np.inf), axis=1)+40
    mean = np.sum(np.where(M, Z, 0), axis=1)/sizes
    b = np.clip(np.log(k/(sizes-k))-mean, lower, upper)
    for _ in range(max_iter):
        q = np.where(M, expit(Z+b[:, None]), 0.)
        residual = q.sum(axis=1)-k
        if np.max(abs(residual)) <= tolerance:
            Q[active] = q
            return Q
        upper = np.where(residual > 0, b, upper)
        lower = np.where(residual <= 0, b, lower)
        slope = np.sum(q*(1-q), axis=1)
        newton = b-residual/np.maximum(slope, 1e-14)
        inside = (newton > lower) & (newton < upper) & np.isfinite(newton)
        b = np.where(inside, newton, (lower+upper)/2)
    raise RuntimeError('soft-k threshold did not meet its mass tolerance')


def soft_topk_objective(w, terms, temperature=.05, reg=.01, delta=1e-12, *, k=5):
    C, same, mask, labels = terms
    w = np.asarray(w, float)
    if w.ndim != 1 or np.any(w < 0) or not np.isfinite(w).all() or reg < 0 or delta <= 0:
        raise ValueError('invalid metric weights or regularization')
    if not len(labels): return 0., np.zeros_like(w)
    classes = np.unique(labels)
    alpha = np.array([1/(len(classes)*sum(labels == c)) for c in labels])
    D = np.einsum('ijd,d->ij', C, w)
    Q = soft_topk_membership(D, mask, k=k, temperature=temperature)
    own = np.sum(Q*same, axis=1)/k
    S = Q*(1-Q); total_slope = S.sum(axis=1)
    # Implicit differentiation of sum_j sigmoid((r_i-D_ij)/tau)=k.
    mean_C = np.einsum('ij,ijd->id', S, C)/np.maximum(total_slope[:, None], 1e-14)
    same_S = S*same
    derivative = (same_S.sum(axis=1)[:, None]*mean_C-np.einsum('ij,ijd->id', same_S, C))/(k*temperature)
    value = -alpha@np.log(own+delta)
    grad = -np.einsum('i,id->d', alpha/(own+delta), derivative)
    log = np.log(len(w)*(w+delta))
    return float(value+reg*np.sum(w*log)), grad+reg*(log+w/(w+delta))
