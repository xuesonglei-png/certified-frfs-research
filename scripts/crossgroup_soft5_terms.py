"""Training-only three-group query/reference exclusion, fixed seed.

Geometry is the enclosing training fold's minmax geometry. This is a
cross-group neighbor objective, not fully cross-fitted preprocessing or risk.
"""
import numpy as np
from certified_frfs.softvote_metric import softvote_terms


def assignments(y, groups=3, seed=20261071):
    y = np.asarray(y)
    if y.ndim != 1 or int(groups) != groups or groups < 2:
        raise ValueError('one dimensional labels and at least two groups required')
    rng = np.random.default_rng(seed)
    out = np.empty(len(y),int)
    for label in np.unique(y):
        ids = np.flatnonzero(y==label)
        ids = rng.permutation(ids)
        out[ids] = np.arange(len(ids))%groups
    return out


def crossgroup_terms(A,y,eps=.01):
    A,y = np.asarray(A,float),np.asarray(y)
    group = assignments(y)
    C,same,mask,labels = softvote_terms(A,y,eps=eps)
    original_valid = np.array([np.sum(y==label)>1 for label in y])
    queries = np.flatnonzero(original_valid)
    allowed = group[queries,None] != group[None,:]
    mask = mask & allowed
    same = same & mask
    keep = same.any(axis=1) & (mask.sum(axis=1)>=5)
    info = {'groups':3,'seed':20261071,'group_assignments':group.tolist(),
        'query_indices':queries[keep].tolist(),
        'excluded_query_indices':sorted(set(range(len(y)))-set(queries[keep].tolist())),
        'available_neighbors':mask[keep].sum(axis=1).tolist(),
        'geometry':'enclosing_training_fold_minmax_not_crossfitted_preprocessing',
        'scope':'training_only_crossgroup_robust_soft5_not_hard_vote_or_risk'}
    return (C[keep],same[keep],mask[keep],labels[keep]),info
