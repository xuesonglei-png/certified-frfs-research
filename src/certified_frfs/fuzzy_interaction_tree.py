"""Training-only supervised complementarity tree for a capacity diagnostic.

Kruskal chooses a maximum spanning tree of discretized label information gains.
This is an explicitly chosen prototype, not an original-baseline reproduction
or a claim that an empirical mutual-information edge is a causal interaction.
"""
import numpy as np
from sklearn.metrics import mutual_info_score
from .triangular_classifier import triangular_memberships


def interaction_tree(X, y):
    X, y = np.asarray(X, float), np.asarray(y)
    if X.ndim != 2 or y.shape != (len(X),) or X.shape[1] < 1:
        raise ValueError('invalid reference table')
    bins = triangular_memberships(X).argmax(axis=-1)
    d = X.shape[1]
    single = [float(mutual_info_score(bins[:, a], y)) for a in range(d)]
    candidates = []
    for u in range(d):
        for v in range(u+1, d):
            joint = float(mutual_info_score(3*bins[:, u]+bins[:, v], y))
            candidates.append({'edge': [u, v], 'joint_MI': joint,
                               'complementarity': max(0., joint-max(single[u], single[v]))})
    candidates.sort(key=lambda r:(-r['complementarity'], r['edge']))
    parents = list(range(d))
    def root(a):
        while parents[a] != a:
            a = parents[a]
        return a
    chosen = []
    for candidate in candidates:
        u, v = candidate['edge']; a, b = root(u), root(v)
        if a != b:
            parents[a] = b; chosen.append(candidate)
        if len(chosen) == d-1:
            break
    return {'edges': [r['edge'] for r in chosen], 'selected': chosen,
            'single_MI': single, 'scope': 'reference_only_maximum_complementarity_tree_not_causal_or_risk_guarantee'}
