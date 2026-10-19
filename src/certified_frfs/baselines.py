"""Training-only feature-selection comparators with explicit definitions.

The nominal fuzzy-rough selector uses this project's fixed-weight relation.
It is not claimed to reproduce every historical QuickReduct implementation.
No function here is an implementation of RFRR or the authors' FD+ baseline.
The explicitly named additive-FD MAFRFS framework instance uses our relation.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score

from .core import gamma


def _training_table(X, y):
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    if X.ndim != 2 or min(X.shape) == 0 or not np.isfinite(X).all():
        raise ValueError("X must be a finite, nonempty 2-D training table")
    if y.ndim != 1 or len(y) != len(X):
        raise ValueError("y must provide one training label per row")
    return X, y


def _k(k, d):
    if isinstance(k, (bool, np.bool_)) or not np.isscalar(k) or int(k) != k or not 0 <= k <= d:
        raise ValueError("k must be an integer between zero and the number of columns")
    return int(k)


def _top(scores, k):
    return np.argsort(-np.asarray(scores), kind="stable")[:k].astype(int).tolist()


def select_all(X, y=None) -> list[int]:
    """Use all available predictors."""
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be a 2-D table")
    return list(range(X.shape[1]))


def select_nominal_frfs(X, y, eta=0.02, weights=None, prune=True) -> list[int]:
    """Greedy nominal dependency selection followed by optional deletion.

    Stop when gamma_B(X) >= gamma_C(X) - eta. Full-table nonnegative weights
    remain fixed, defaulting to 1/sqrt(d), as in core.gamma. Select the largest
    marginal dependency gain; break exact ties by source column index. A
    reverse-order deletion pass gives an inclusion-minimal feasible subset
    (not a minimum-cardinality subset). No perturbation certificate is used.
    """
    X, y = _training_table(X, y)
    if not np.isfinite(eta) or not 0 <= eta <= 1:
        raise ValueError("eta must lie in [0,1]")
    d = X.shape[1]
    weights = np.full(d, 1.0 / np.sqrt(d)) if weights is None else np.asarray(weights, float)
    if weights.shape != (d,) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("weights must be a nonnegative finite full-table vector")
    target = gamma(X, y, weights=weights) - eta
    selected, remaining = [], list(range(d))
    current = gamma(X, y, features=[], weights=weights)
    distances = np.zeros((len(X), len(X)), dtype=float)
    same_class = y[:, None] == y[None, :]
    # This is the same clipped nearest-opposite-class dependency as core.gamma;
    # incremental distances avoid rebuilding previously selected coordinates.
    while current + 1e-12 < target and remaining:
        best_feature, best_value = None, -np.inf
        for feature in remaining:
            candidate = distances + weights[feature] * np.abs(X[:, feature, None] - X[None, :, feature])
            candidate[same_class] = np.inf
            value = float(np.minimum(1.0, candidate.min(axis=1)).mean())
            if value > best_value:
                best_feature, best_value = feature, value
        selected.append(best_feature)
        remaining.remove(best_feature)
        distances += weights[best_feature] * np.abs(X[:, best_feature, None] - X[None, :, best_feature])
        current = best_value
    if prune:
        for feature in selected[::-1]:
            candidate = [a for a in selected if a != feature]
            if gamma(X, y, features=candidate, weights=weights) + 1e-12 >= target:
                selected = candidate
    return sorted(selected)


def mi_scores(X, y, random_state=0, n_neighbors=3, discrete_features="auto") -> np.ndarray:
    """Univariate mutual information from sklearn's training-only estimator.

    On dense input 'auto' treats predictors as continuous. Pass an explicit
    Boolean mask when categorical indicator columns should use discrete MI.
    """
    X, y = _training_table(X, y)
    if len(X) < 2:
        return np.zeros(X.shape[1])
    return mutual_info_classif(X, y, discrete_features=discrete_features,
                              n_neighbors=n_neighbors, random_state=random_state)


def select_mi(X, y, k, random_state=0, n_neighbors=3, discrete_features="auto") -> list[int]:
    """Top-k training-set MI ranking; k must be fixed without test labels."""
    X, y = _training_table(X, y)
    k = _k(k, X.shape[1])
    if k == 0:
        return []
    return _top(mi_scores(X, y, random_state, n_neighbors, discrete_features), k)


def select_mrmr(X, y, k, n_bins=5, random_state=0) -> list[int]:
    """Discrete-MI mRMR using the MID (relevance minus mean redundancy) rule.

    Each numerical predictor is quantile-discretized using training values
    only, with at most n_bins bins (duplicate cut points collapse). Relevance
    and pairwise redundancy are both empirical discrete mutual information
    in nats. random_state is accepted for a uniform caller API; this algorithm
    is deterministic. This is not an external package reproduction.
    """
    X, y = _training_table(X, y)
    k = _k(k, X.shape[1])
    if isinstance(n_bins, (bool, np.bool_)) or int(n_bins) != n_bins or n_bins < 2:
        raise ValueError("n_bins must be an integer >= 2")
    if k == 0:
        return []
    quantiles = np.linspace(0.0, 1.0, int(n_bins) + 1)[1:-1]
    discretized = np.empty_like(X, dtype=np.int64)
    for feature in range(X.shape[1]):
        cuts = np.unique(np.quantile(X[:, feature], quantiles))
        discretized[:, feature] = np.searchsorted(cuts, X[:, feature], side="right")
    relevance = np.asarray([mutual_info_score(y, discretized[:, a]) for a in range(X.shape[1])])
    selected, remaining = [], list(range(X.shape[1]))
    redundancy_sum = np.zeros(X.shape[1])
    while len(selected) < k:
        scores = relevance.copy() if not selected else relevance - redundancy_sum / len(selected)
        feature = max(remaining, key=lambda a: (scores[a], -a))
        selected.append(feature)
        remaining.remove(feature)
        for other in remaining:
            redundancy_sum[other] += mutual_info_score(discretized[:, feature], discretized[:, other])
    return selected


def relieff_scores(X, y, n_neighbors=10, max_samples=200, random_state=0) -> np.ndarray:
    """Multiclass ReliefF with class-prior-weighted nearest misses.

    Numeric differences use training ranges; L1 distance selects neighbors.
    Each reference uses up to n_neighbors hits and misses per class. Singleton
    classes contribute zero hit correction. At most max_samples references
    are drawn without replacement, or all rows when max_samples is None.
    """
    X, y = _training_table(X, y)
    if isinstance(n_neighbors, (bool, np.bool_)) or int(n_neighbors) != n_neighbors or n_neighbors < 1:
        raise ValueError("n_neighbors must be a positive integer")
    if max_samples is not None and (isinstance(max_samples, (bool, np.bool_)) or int(max_samples) != max_samples or max_samples < 1):
        raise ValueError("max_samples must be None or a positive integer")
    classes, encoded, counts = np.unique(y, return_inverse=True, return_counts=True)
    if len(classes) < 2:
        return np.zeros(X.shape[1])
    ranges = np.ptp(X, axis=0)
    ranges[ranges == 0] = 1.0
    normalized = (X - X.min(axis=0)) / ranges
    n_ref = len(X) if max_samples is None else min(len(X), int(max_samples))
    references = (np.arange(len(X)) if n_ref == len(X)
                  else np.random.default_rng(random_state).choice(len(X), n_ref, replace=False))
    distances = cdist(normalized[references], normalized, metric="cityblock")
    scores = np.zeros(X.shape[1])
    priors = counts / len(X)
    for row, reference in enumerate(references):
        own = encoded[reference]
        order = np.argsort(distances[row], kind="stable")
        hits = order[(encoded[order] == own) & (order != reference)][:int(n_neighbors)]
        if len(hits):
            scores -= np.abs(normalized[hits] - normalized[reference]).mean(axis=0)
        for label in range(len(classes)):
            if label == own:
                continue
            misses = order[encoded[order] == label][:int(n_neighbors)]
            scores += (priors[label] / (1.0 - priors[own])
                       * np.abs(normalized[misses] - normalized[reference]).mean(axis=0))
    return scores / n_ref


def select_relieff(X, y, k, n_neighbors=10, max_samples=200, random_state=0) -> list[int]:
    """Top-k multiclass ReliefF ranking fit to training rows only."""
    X, y = _training_table(X, y)
    k = _k(k, X.shape[1])
    if k == 0:
        return []
    return _top(relieff_scores(X, y, n_neighbors, max_samples, random_state), k)


def mafrfs_margin_ratio(X, y, features, margin_type="local") -> float:
    """WBMR from arXiv:2505.15250v1, Eqs. (12)-(22), with Euclidean norms.

    Within-class scatter is averaged inside each class, then summed equally
    across classes. 'local' sums pairwise class-center distances (Eq. 20);
    'global' sums distances to the sample-weighted overall center (Eq. 18).
    A zero between-class margin has no discriminative center separation and
    receives +infinity; this boundary convention is not specified in v1.
    """
    X, y = _training_table(X, y)
    features = np.asarray(list(features), dtype=int)
    if margin_type not in {"local", "global"}:
        raise ValueError("margin_type must be 'local' or 'global'")
    if len(features) == 0:
        return float("inf")
    if np.any(features < 0) or np.any(features >= X.shape[1]) or len(np.unique(features)) != len(features):
        raise ValueError("features must be distinct valid column indices")
    table = X[:, features]
    classes = np.unique(y)
    centers = np.asarray([table[y == label].mean(axis=0) for label in classes])
    within = sum(np.linalg.norm(table[y == label] - centers[index], axis=1).mean()
                 for index, label in enumerate(classes))
    if margin_type == "local":
        between = sum(np.linalg.norm(centers[i] - centers[j])
                      for i in range(len(classes)) for j in range(i + 1, len(classes)))
    else:
        between = np.linalg.norm(centers - table.mean(axis=0), axis=1).sum()
    return float(within / between) if between > 0 else float("inf")


def select_mafrfs_additive(X, y, k, sop=2, margin_type="local", weights=None) -> list[int]:
    """MAFRFS Algorithm 2 instantiated with this project's additive FD.

    This is NOT reproduction of the authors' FD+ numerical implementation.
    arXiv:2505.15250v1 specifies Algorithm 2 and WBMR, but does not specify
    the numerical feature-relation construction needed to reproduce its FD+.
    Here core.gamma is an explicit replacement. Report as 'maf_framework'.

    In every outer iteration, build a pool by greedily maximizing dependency
    over selected UNION the evolving pool, not by ranking singleton gains
    once. Select one pool member minimizing WBMR(selected + member). This
    gives Eq. (23)'s argmin without subtracting a common previous WBMR and
    avoids the undefined 0/0 of the empty subset. sop=2 and local margins
    are predeclared choices among v1's allowed settings, not author defaults.
    Stop after k attributes; k is supplied using training-only information.
    """
    X, y = _training_table(X, y)
    k = _k(k, X.shape[1])
    if isinstance(sop, (bool, np.bool_)) or int(sop) != sop or sop < 1:
        raise ValueError("sop must be a positive integer")
    if margin_type not in {"local", "global"}:
        raise ValueError("margin_type must be 'local' or 'global'")
    # Validate fixed weights even for k=0, using the public core API.
    gamma(X, y, features=[], weights=weights)
    selected = []
    remaining = list(range(X.shape[1]))
    dependency_cache = {}

    def dependency(features):
        key = tuple(sorted(features))
        if key not in dependency_cache:
            dependency_cache[key] = gamma(X, y, features=key, weights=weights)
        return dependency_cache[key]

    while len(selected) < k:
        pool = []
        for _ in range(min(int(sop), len(remaining))):
            base = selected + pool
            candidates = [a for a in remaining if a not in pool]
            # Subtracting FD(base) is common to every candidate and cannot
            # change the maximizer (Algorithm 2, steps 5-7; p. 6 text).
            feature = max(candidates, key=lambda a: (dependency(base + [a]), -a))
            pool.append(feature)
        chosen = min(pool, key=lambda a: (mafrfs_margin_ratio(X, y, selected + [a], margin_type), a))
        selected.append(chosen)
        remaining.remove(chosen)
    return selected
