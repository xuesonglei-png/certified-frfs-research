"""Conventional BIC-adjusted complementarity maximum-weight forest.

Observed-cell degrees are an explicit small-sample heuristic. Neither BIC
consistency nor multiple-testing error control nor predictive gains are claimed.
"""
import numpy as np
from sklearn.metrics import mutual_info_score
from .triangular_classifier import triangular_memberships


def penalized_forest(X, y):
    X, y = np.asarray(X, float), np.asarray(y)
    if X.ndim != 2 or y.shape != (len(X),) or len(X) < 2 or X.shape[1] < 1:
        raise ValueError('invalid reference table')
    bins = triangular_memberships(X).argmax(axis=-1)
    n, d = X.shape
    q = len(np.unique(y))
    single = [float(mutual_info_score(bins[:, a], y)) for a in range(d)]
    categories = [len(np.unique(bins[:, a])) for a in range(d)]
    candidates = []
    for u in range(d):
        for v in range(u+1, d):
            joint_bins = 3*bins[:, u]+bins[:, v]
            joint = float(mutual_info_score(joint_bins, y))
            gain = max(0., joint-max(single[u], single[v]))
            df = (len(np.unique(joint_bins))-max(categories[u], categories[v]))*(q-1)
            penalty = float(df*np.log(n)/(2*n))
            candidates.append({'edge': [u, v], 'joint_MI': joint, 'complementarity': gain,
                               'degrees_increment': int(df), 'bic_penalty': penalty,
                               'adjusted_gain': gain-penalty})
    candidates.sort(key=lambda r: (-r['adjusted_gain'], r['edge']))
    parent = list(range(d))
    def root(a):
        while parent[a] != a:
            a = parent[a]
        return a
    selected = []
    for edge in candidates:
        if edge['adjusted_gain'] <= 0:
            break
        u, v = edge['edge']
        a, b = root(u), root(v)
        if a != b:
            parent[a] = b
            selected.append(edge)
    return {'edges': [r['edge'] for r in selected], 'selected': selected,
            'single_MI': single,
            'scope': 'reference_only_BIC_adjusted_maximum_weight_forest_not_FDR_causal_or_risk_guarantee'}
