"""Shared-box certificates for fixed query labels against a reference table.

Targets can be nominal predictions. Query targets never alter selection.
Positive bounds certify prediction stability, not correctness or population risk.
"""
import numpy as np
from scipy.spatial.distance import cdist
from .core import interval_bounds
from .margin import triplet_interval_min


def certify_queries(reference, labels, queries, targets, features, *, eps=.01,
                    n_friends=3):
    A, B = np.asarray(reference, float), np.asarray(queries, float)
    labels, targets = np.asarray(labels), np.asarray(targets)
    features = list(features)
    if (A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[1] or
            len(A) != len(labels) or len(B) != len(targets) or not features or
            len(set(features)) != len(features) or n_friends < 1):
        raise ValueError('invalid query certificate inputs')
    d = A.shape[1]
    if any(a < 0 or a >= d for a in features):
        raise ValueError('invalid attributes')
    alo, ahi = interval_bounds(A, eps)
    blo, bhi = interval_bounds(B, eps)
    distances = cdist(B[:, features], A[:, features], 'cityblock')
    lower = np.full(len(B), -np.inf)
    for i, target in enumerate(targets):
        friends = np.flatnonzero(labels == target)
        enemies = np.flatnonzero(labels != target)
        if not len(friends):
            continue
        if not len(enemies):
            lower[i] = np.inf
            continue
        friends = friends[np.lexsort((friends, distances[i, friends]))][:n_friends]
        terms = np.zeros((len(enemies), len(friends)))
        for a in features:
            terms += triplet_interval_min(
                blo[i, a], bhi[i, a], alo[enemies, a, None], ahi[enemies, a, None],
                alo[friends, a][None, :], ahi[friends, a][None, :])/d
        bound = terms.max(axis=1).min()
        lower[i] = bound-1e-10*(1+abs(bound))
    certified = lower > 1e-10
    return {'lower_bounds': [float(b) if np.isfinite(b) else None for b in lower],
            'certified': certified.tolist(),
            'coverage': float(certified.mean()) if len(B) else 0.,
            'scope': 'fixed_query_targets_shared_reference_and_query_boxes_not_correctness',
            'rounding': 'ordinary_float_with_conservative_tolerance'}
