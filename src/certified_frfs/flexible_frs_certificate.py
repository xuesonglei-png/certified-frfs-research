"""Existing enemy-specific witnesses coupled to the unchanged FRS contract.

Protect the SAME original fixed-witness robust training objects and retain
the original dependency bound. This is a sufficient-condition refinement,
not a population accuracy guarantee or a new generic order-statistic theorem.
"""
import numpy as np
from .flexible_witness import flexible_bounds
from .majority_metric import MajorityTerms
from .redistributed_loss import redistributed_loss_bound


def flexible_weight_certificate(X,y,features,w_full,w_reduced,*,eps=.01,
    retention=.9,dependency_eta=.1,protected=None,pool_size=8):
    X,y = np.asarray(X,float),np.asarray(y); fs = sorted(features); w_full = np.asarray(w_full,float); w_reduced = np.asarray(w_reduced,float)
    if X.ndim != 2 or y.shape != (len(X),) or not fs or len(set(fs)) != len(fs) or any(int(a) != a or not 0 <= a < X.shape[1] for a in fs):
        raise ValueError('valid training table and support required')
    if w_full.shape != (X.shape[1],) or w_reduced.shape != (len(fs),) or not 0 < retention <= 1 or not 0 <= dependency_eta <= 1:
        raise ValueError('compatible weights and certificate thresholds required')
    if protected is None:
        protected = MajorityTerms(X,y,weights=w_full,eps=eps).bounds(range(X.shape[1]),allowed_enemies=2) > 1e-10
    protected = np.asarray(protected,bool)
    if protected.shape != y.shape: raise ValueError('original protected-mask shape required')
    groups = [(y == c)&protected for c in np.unique(y)]; required = [int(np.ceil(retention*g.sum()-1e-12)) for g in groups]
    embedded = np.zeros(X.shape[1]); embedded[fs] = w_reduced
    bounds = flexible_bounds(X,y,fs,embedded,eps=eps,pool_size=pool_size)
    counts = [int(((bounds > 1e-10)&g).sum()) for g in groups]; majority = all(c >= r for c,r in zip(counts,required))
    dep = redistributed_loss_bound(X,y,fs,w_full,w_reduced,eps=eps)
    return {'feasible':bool(majority and dep['upper_bound'] <= dependency_eta),'majority_feasible':bool(majority),
        'retained_counts':counts,'required':required,'dependency_certificate':dep,'pool_size':pool_size,
        'scope':'unchanged_original_protected_mask_and_FRS_dependency_with_enemy_specific_sufficient_majority'}
