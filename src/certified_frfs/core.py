"""Sound certificates for shared-box perturbations of fuzzy-rough dependency.

Weights belong to the full table and are never renormalized after deletion.
All returned bounds concern empirical dependency, not classifier predictions.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def _inputs(X, y, features=None, weights=None):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if X.ndim != 2 or len(X) == 0 or not np.isfinite(X).all():
        raise ValueError("X must be a nonempty finite two-dimensional array")
    if y.ndim != 1 or len(y) != len(X):
        raise ValueError("y must have one label per row")
    d = X.shape[1]
    if d == 0:
        raise ValueError("X must contain at least one full-table attribute")
    if weights is None:
        weights = np.full(d, 1.0 / np.sqrt(d))
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (d,) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("weights must be a finite nonnegative full-table vector")
    if features is None:
        features = np.arange(d, dtype=int)
    else:
        raw = np.asarray(list(features))
        if raw.dtype == bool:
            if raw.shape != (d,):
                raise ValueError("Boolean feature mask must have full-table length")
            features = np.flatnonzero(raw)
        else:
            features = np.asarray(raw, dtype=int)
            if np.any(raw != features):
                raise ValueError("features must be integer indices")
    if len(np.unique(features)) != len(features) or np.any(features < 0) or np.any(features >= d):
        raise ValueError("features must be distinct full-table indices")
    return X, y, np.sort(features), weights


def interval_bounds(X, eps, clip=True):
    """Return coordinate intervals; eps is scalar, length d, or shape (n,d).

    With clip=True the nominal table must be in [0,1], and intervals are
    intersected with [0,1]. With clip=False no domain clipping is applied.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError("X must be a finite two-dimensional array")
    eps = np.asarray(eps, dtype=float)
    if not np.isfinite(eps).all() or np.any(eps < 0):
        raise ValueError("eps must be finite and nonnegative")
    try:
        radius = np.broadcast_to(eps, X.shape)
    except ValueError as error:
        raise ValueError("eps must broadcast to X") from error
    if clip and (np.any(X < 0) or np.any(X > 1)):
        raise ValueError("clip=True requires nominal X in [0,1]")
    lower, upper = X - radius, X + radius
    if clip:
        lower, upper = np.maximum(lower, 0.0), np.minimum(upper, 1.0)
    return lower, upper


def _distance(X, features, weights):
    if len(features) == 0:
        return np.zeros((len(X), len(X)))
    return cdist(X[:, features] * weights[features],
                 X[:, features] * weights[features], metric="cityblock")


def positive_region(X, y, features=None, weights=None):
    """Per-object fuzzy positive-region memberships (single class: all ones)."""
    X, y, features, weights = _inputs(X, y, features, weights)
    distances = _distance(X, features, weights)
    distances[y[:, None] == y[None, :]] = np.inf
    return np.minimum(1.0, distances.min(axis=1))


def gamma(X, y, features=None, weights=None):
    """Mean fuzzy positive region under fixed weighted L1 similarity."""
    return float(positive_region(X, y, features, weights).mean())


def _interval_distances(lower, upper, features, weights, need_lower=True):
    """Exact pairwise minima/maxima; pairs need not attain them jointly."""
    midpoint, radii = (lower + upper) / 2, (upper - lower) / 2
    centers = _distance(midpoint, features, weights)
    radius_sums = radii[:, features] @ weights[features]
    hi = centers + radius_sums[:, None] + radius_sums[None, :]
    lo = np.zeros_like(hi)
    if need_lower:
        for k in features:
            gap = np.maximum(lower[:, k, None] - upper[None, :, k],
                             lower[None, :, k] - upper[:, k, None])
            lo += weights[k] * np.maximum(gap, 0.0)
    np.fill_diagonal(hi, 0.0)
    return lo, hi


def forest_maximum(lower, upper, edges, weights=None):
    """Maximize a weighted sum of absolute differences on a forest.

    edges is an iterable of (i,j,coefficient); repeated undirected edges are
    aggregated. Attribute weights are optional. The shared-coordinate
    maximum is exact; this is not a general vertex enumeration for E(B).
    """
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    if lower.ndim != 2 or upper.shape != lower.shape or np.any(lower > upper):
        raise ValueError("invalid interval arrays")
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("intervals must be finite")
    n, d = lower.shape
    weights = np.ones(d) if weights is None else np.asarray(weights, float)
    if weights.shape != (d,) or np.any(weights < 0) or not np.isfinite(weights).all():
        raise ValueError("invalid attribute weights")
    aggregated = {}
    for i, j, coefficient in edges:
        i, j, coefficient = int(i), int(j), float(coefficient)
        if not 0 <= i < n or not 0 <= j < n or coefficient < 0 or not np.isfinite(coefficient):
            raise ValueError("invalid forest edge")
        if i == j or coefficient == 0:
            continue
        key = (min(i, j), max(i, j))
        aggregated[key] = aggregated.get(key, 0.0) + coefficient
    adjacency = [[] for _ in range(n)]
    for (i, j), coefficient in aggregated.items():
        adjacency[i].append((j, coefficient))
        adjacency[j].append((i, coefficient))
    parent = np.full(n, -2, dtype=int)
    order, roots, incoming = [], [], np.zeros(n)
    for root in range(n):
        if parent[root] != -2:
            continue
        roots.append(root)
        parent[root] = -1
        stack = [root]
        while stack:
            i = stack.pop()
            order.append(i)
            for j, coefficient in adjacency[i]:
                if j == parent[i]:
                    continue
                if parent[j] != -2:
                    raise ValueError("edges contain an undirected cycle")
                parent[j], incoming[j] = i, coefficient
                stack.append(j)
    total = 0.0
    for k in range(d):
        if weights[k] == 0:
            continue
        values = np.stack((lower[:, k], upper[:, k]), axis=1)
        dp = np.zeros((n, 2))
        for i in reversed(order):
            p = parent[i]
            if p >= 0:
                pair = incoming[i] * np.abs(values[p, :, None] - values[i, None, :])
                dp[p] += np.max(pair + dp[i][None, :], axis=1)
        total += weights[k] * sum(float(dp[root].max()) for root in roots)
    return float(total)


def _shared_anchor_difference_upper(anchor_lower, anchor_upper,
                                    neighbor_lower, neighbor_upper,
                                    other_lower, other_upper):
    """Exact max of |a-b|-|a-c| over three independent scalar intervals.

    Arguments broadcast. The common anchor a is retained. After maximizing
    over b and minimizing over c, the two branches are respectively
    nondecreasing and nonincreasing in a, so anchor endpoints suffice.
    """
    far_lower = np.maximum(np.abs(anchor_lower - neighbor_lower),
                           np.abs(anchor_lower - neighbor_upper))
    far_upper = np.maximum(np.abs(anchor_upper - neighbor_lower),
                           np.abs(anchor_upper - neighbor_upper))
    near_lower = np.maximum(np.maximum(other_lower - anchor_lower,
                                       anchor_lower - other_upper), 0.0)
    near_upper = np.maximum(np.maximum(other_lower - anchor_upper,
                                       anchor_upper - other_upper), 0.0)
    return np.maximum(far_lower - near_lower, far_upper - near_upper)


def _exact_stability_mask(lower, upper, nearest, features, weights, cross):
    """Exact strict stability in real arithmetic for nominal unique neighbors.

    Margin >= 0 means the fixed neighbor is never worse; strict positive
    margin is required for uniqueness. A positive numerical tolerance makes
    acceptance conservative. Pairwise comparisons may have different
    maximizers: stability is a universal condition for every comparison.
    """
    n = len(lower)
    difference_upper = np.zeros((n, n))
    for k in features:
        if weights[k] == 0:
            continue
        difference_upper += weights[k] * _shared_anchor_difference_upper(
            lower[:, k, None], upper[:, k, None],
            lower[nearest, k, None], upper[nearest, k, None],
            lower[None, :, k], upper[None, :, k])
    comparisons = cross.copy()
    comparisons[np.arange(n), nearest] = False
    worst_difference = np.max(np.where(comparisons, difference_upper, -np.inf), axis=1)
    return worst_difference < -1e-12, -worst_difference


def certificate(X, y, features=None, weights=None, eps=0.0,
                method="hybrid", clip=True, stability="interval"):
    """Return a sound upper bound on max_z [gamma_C(z)-gamma_B(z)].

    method='independent' uses safe objectwise interval bounds; 'lipschitz'
    uses nominal loss plus perturbation radii; 'hybrid' also retains the
    common coordinates on the stable-nearest-neighbor forest. It takes the
    minimum of all three valid bounds. stability='interval' (the default)
    uses a sufficient interval test; 'exact' uses exact shared-anchor
    comparisons to recognize every strictly stable nominal neighbor,
    subject to a conservative numerical tolerance.
    """
    X, y, features, weights = _inputs(X, y, features, weights)
    if method not in {"hybrid", "independent", "lipschitz"}:
        raise ValueError("unknown certificate method")
    if stability not in {"interval", "exact"}:
        raise ValueError("stability must be 'interval' or 'exact'")
    lower, upper = interval_bounds(X, eps, clip=clip)
    n, d = X.shape
    full = np.arange(d)
    omitted = np.setdiff1d(full, features)
    nominal_full = positive_region(X, y, full, weights)
    nominal_selected = positive_region(X, y, features, weights)
    nominal_loss = float(np.maximum(nominal_full - nominal_selected, 0).mean())
    result = dict(nominal_loss=nominal_loss, stable_fraction=0.0,
                  stable_count=0, n_samples=n, features=features.tolist(),
                  method=method, stable_neighbors=[-1] * n,
                  stability_evaluated=False, stability_method=stability)
    if len(omitted) == 0 or len(np.unique(y)) < 2 or np.all(weights[omitted] == 0):
        result.update(upper_bound=0.0, independent_bound=0.0,
                      lipschitz_bound=0.0, hybrid_bound=0.0, forest_bound=0.0)
        return result
    cross = y[:, None] != y[None, :]
    lo_b, hi_b = _interval_distances(lower, upper, features, weights)
    _, hi_c = _interval_distances(lower, upper, full, weights, need_lower=False)
    hi_omitted = np.maximum(0.0, hi_c - hi_b)
    masked_lo = np.where(cross, lo_b, np.inf)
    lower_min = masked_lo.min(axis=1)
    upper_min_b = np.where(cross, hi_b, np.inf).min(axis=1)
    upper_pos_c = np.minimum(1.0, np.where(cross, hi_c, np.inf).min(axis=1))
    lower_pos_b = np.minimum(1.0, lower_min)
    # Every actual B-nearest neighbor must be in this candidate superset.
    candidates = cross & (lo_b <= upper_min_b[:, None] + 1e-12)
    omitted_max = np.max(np.where(candidates, hi_omitted, 0.0), axis=1)
    object_bounds = np.maximum(0.0, np.minimum(upper_pos_c - lower_pos_b, omitted_max))
    independent = float(object_bounds.mean())
    radii = np.maximum(X - lower, upper - X)
    r_c, r_b = radii @ weights, radii[:, features] @ weights[features]
    lip_c = np.max(np.where(cross, r_c[:, None] + r_c[None, :], 0.0), axis=1)
    lip_b = np.max(np.where(cross, r_b[:, None] + r_b[None, :], 0.0), axis=1)
    lip_objects = np.minimum(1.0, np.maximum(nominal_full - nominal_selected, 0.0) + lip_c + lip_b)
    lipschitz = float(lip_objects.mean())
    forest_bound = independent
    if method == "hybrid":
        result["stability_evaluated"] = True
        nominal_d = _distance(X, features, weights)
        nearest = np.argmin(np.where(cross, nominal_d, np.inf), axis=1)
        competitor_lowers = masked_lo.copy()
        competitor_lowers[np.arange(n), nearest] = np.inf
        # A positive margin avoids accepting numerical ties as unique.
        interval_stable = hi_b[np.arange(n), nearest] + 1e-12 < competitor_lowers.min(axis=1)
        stable = interval_stable.copy()
        if stability == "exact":
            exact_stable, _ = _exact_stability_mask(lower, upper, nearest, features, weights, cross)
            # The union preserves the inexpensive sufficient check, including
            # any harmless difference in floating-point accumulation order.
            stable |= exact_stable
        stable_indices = np.flatnonzero(stable)
        result["stable_count"] = int(stable.sum())
        result["stable_fraction"] = float(stable.mean())
        result["stable_neighbors"] = np.where(stable, nearest, -1).tolist()
        edges = [(int(i), int(nearest[i]), 1.0) for i in stable_indices]
        coupled = forest_maximum(lower[:, omitted], upper[:, omitted], edges, weights[omitted])
        forest_bound = float((coupled + object_bounds[~stable].sum()) / n)
        # Each group can be bounded separately, retaining the best stable
        # group bound without weakening the safe unstable group bound.
        forest_bound = min(forest_bound,
                           float((min(coupled, object_bounds[stable].sum()) + object_bounds[~stable].sum()) / n))
        if stability == "exact" and np.any(stable & ~interval_stable):
            # A larger stable set need not monotonically improve this
            # relaxation: a new distance term can exceed its pointwise cap.
            # Retain the valid interval-subset certificate as well.
            old_edges = [(int(i), int(nearest[i])) + (1.0,)
                         for i in np.flatnonzero(interval_stable)]
            old_coupled = forest_maximum(lower[:, omitted], upper[:, omitted], old_edges, weights[omitted])
            old_bound = (min(old_coupled, object_bounds[interval_stable].sum())
                         + object_bounds[~interval_stable].sum()) / n
            forest_bound = min(forest_bound, float(old_bound))
    hybrid = min(independent, lipschitz, forest_bound)
    bounds = dict(independent_bound=independent, lipschitz_bound=lipschitz,
                  hybrid_bound=hybrid, forest_bound=forest_bound)
    selected = {"independent": independent, "lipschitz": lipschitz, "hybrid": hybrid}[method]
    # Ordinary floating point, not interval arithmetic: a small outward
    # tolerance protects comparisons, but is not a formal rounding proof.
    result.update({key: float(np.clip(value + 1e-12, 0.0, 1.0)) for key, value in bounds.items()})
    result["upper_bound"] = float(np.clip(max(nominal_loss, selected) + 1e-12, 0.0, 1.0))
    return result
