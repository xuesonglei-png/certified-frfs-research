"""Predictive ranking followed by certified attribute restoration.

Restore high-priority omitted attributes until the original fixed-weight
loss criterion is satisfied. A final fresh certificate is mandatory. Ranking
and target cardinality must be chosen exclusively inside training validation.
"""
import numpy as np
from .core import certificate


def repair_to_certificate(X, y, ranking, k, *, eps=.01, eta=.10, weights=None):
    d = X.shape[1]
    ranking = list(ranking)
    if sorted(ranking) != list(range(d)) or not 1 <= k <= d:
        raise ValueError('ranking must be a permutation and k must be valid')
    weights = np.ones(d) / np.sqrt(d) if weights is None else np.asarray(weights)
    features = ranking[:k]
    history = []
    for size in range(k, d + 1):
        cert = certificate(X, y, features, weights, eps)
        history.append({'k': len(features), 'upper': cert['upper_bound']})
        if cert['upper_bound'] <= eta - 1e-10 or size == d:
            if cert['upper_bound'] > eta + 1e-10:
                raise AssertionError('final certificate infeasible')
            return {'features': features, 'certificate': cert, 'history': history,
                    'initial_k': k, 'restored_count': len(features) - k}
        features.append(ranking[size])
    raise AssertionError('unreachable')
