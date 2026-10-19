"""Direct, class-balanced finite-table robust LOO decision certificates.

For fixed training-learned weights, each triplet lower bound holds for all
shared perturbations. min_enemy max_friend of these lower bounds is <= the
robust LOO margin. Positive bounds certify the corresponding object's label.
Class-average coverage is a finite-table guarantee, not test risk.
"""
import numpy as np
from scipy.spatial.distance import cdist
from .core import interval_bounds
from .margin import triplet_interval_min


def certified_decision_coverage(X, y, features, *, eps=.01, weights=None,
                                n_friends=3):
    X, y = np.asarray(X, float), np.asarray(y)
    features = list(features)
    n, d = X.shape
    weights = np.ones(d)/d if weights is None else np.asarray(weights, float)
    if weights.shape != (d,) or np.any(weights < 0) or not np.isfinite(weights).all():
        raise ValueError('invalid fixed weights')
    if len(y) != n or not features or len(set(features)) != len(features):
        raise ValueError('invalid table or subset')
    if any(a < 0 or a >= d for a in features) or n_friends < 1:
        raise ValueError('invalid attributes or friend count')
    lo, hi = interval_bounds(X, eps)
    distance = cdist(X[:, features]*weights[features], X[:, features]*weights[features], 'cityblock')
    lower = np.full(n, -np.inf)
    for i in range(n):
        friends = np.flatnonzero((y == y[i]) & (np.arange(n) != i))
        enemies = np.flatnonzero(y != y[i])
        if not len(friends) or not len(enemies):
            continue
        friends = friends[np.lexsort((friends, distance[i, friends]))][:int(n_friends)]
        terms = np.zeros((len(enemies), len(friends)))
        for a in features:
            terms += weights[a]*triplet_interval_min(
                lo[i, a], hi[i, a], lo[enemies, a, None], hi[enemies, a, None],
                lo[friends, a][None, :], hi[friends, a][None, :])
        bound = terms.max(axis=1).min()
        lower[i] = bound - 1e-10*(1+abs(bound))
    positive = lower > 1e-10
    coverage = [{'label': int(label), 'count': int((y == label).sum()),
                 'positive': int(positive[y == label].sum()),
                 'fraction': float(positive[y == label].mean())}
                for label in np.unique(y)]
    return {'lower_bounds': [float(b) if np.isfinite(b) else None for b in lower],
            'class_coverage': coverage,
            'balanced_coverage': float(np.mean([r['fraction'] for r in coverage])),
            'min_class_coverage': min(r['fraction'] for r in coverage),
            'certified_indices': np.flatnonzero(positive).tolist(),
            'target': 'robust_LOO_1NN_decisions_on_this_table',
            'rounding': 'ordinary_float_with_conservative_tolerance'}


def select_decision_coverage(X, y, ranking, *, eps=.01, penalty=.02,
                             margin_scale=None):
    """Search ranking prefixes by balanced and worst-class robust coverage.

    This maximizes a finite-table objective; no prescribed coverage threshold
    or out-of-sample accuracy is guaranteed. Recompute subset-specific friends.
    """
    d = np.asarray(X).shape[1]
    ranking = list(ranking)
    if (sorted(ranking) != list(range(d)) or penalty < 0 or
            (margin_scale is not None and margin_scale <= 0)):
        raise ValueError('ranking must be a permutation; penalty nonnegative')
    sizes = sorted(set(max(1, int(np.ceil(f*d))) for f in
                       [.1, .2, .35, .5, .65, .8, 1.]))
    trace = []
    for k in sizes:
        cert = certified_decision_coverage(X, y, ranking[:k], eps=eps)
        if margin_scale is None:
            objective = (.5*cert['balanced_coverage'] +
                         .5*cert['min_class_coverage'] - penalty*k/d)
        else:
            bounds = np.array([b if b is not None else -np.inf for b in cert['lower_bounds']])
            # Scale out the common selected-weight sum, keeping the original certificate intact.
            utility = np.clip(bounds/(margin_scale*k/d), -1., 1.)
            classes = [utility[np.asarray(y) == label].mean() for label in np.unique(y)]
            objective = .5*np.mean(classes)+.5*min(classes)-penalty*k/d
        trace.append({'k': k, 'objective': objective,
                      'balanced_coverage': cert['balanced_coverage'],
                      'min_class_coverage': cert['min_class_coverage']})
    chosen = max(trace, key=lambda row: (row['objective'], -row['k']))
    features = ranking[:chosen['k']]
    return {'features': features, 'trace': trace,
            'certificate': certified_decision_coverage(X, y, features, eps=eps),
            'nonvacuous': chosen['min_class_coverage'] > 0}
