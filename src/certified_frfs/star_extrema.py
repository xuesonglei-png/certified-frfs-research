"""Exact real-arithmetic extrema of nonnegative weighted interval-star costs.

The implementation uses ordinary floating point, not outward-rounded interval
arithmetic. It returns extremizing tables as well as values; callers that use a
value as a numerical certificate must apply their own numerical policy.
"""
from __future__ import annotations

import operator

import numpy as np


def _nonnegative_weights(values, size, name):
    weights = np.ones(size) if values is None else np.asarray(values, dtype=float)
    if (weights.shape != (size,) or not np.isfinite(weights).all()
            or np.any(weights < 0)):
        raise ValueError(f"{name} must be a finite nonnegative vector of length {size}")
    return weights


def _edge_indices(centers, n):
    edges = []
    for edge in centers:
        try:
            leaf, center = edge
        except (TypeError, ValueError) as error:
            raise ValueError("centers must contain (leaf, center) pairs") from error
        indices = []
        for value in (leaf, center):
            if isinstance(value, (bool, np.bool_)):
                raise ValueError("star vertex indices must be integers, not Boolean")
            try:
                index = operator.index(value)
            except TypeError as error:
                raise ValueError("star vertex indices must be integers") from error
            if not 0 <= index < n:
                raise ValueError("star vertex index is outside the interval table")
            indices.append(index)
        edges.append(indices)
    edges = np.asarray(edges, dtype=int).reshape(-1, 2)
    if len(np.unique(edges[:, 0])) != len(edges):
        raise ValueError("each leaf must occur in exactly one star edge")
    if np.intersect1d(edges[:, 0], edges[:, 1]).size:
        raise ValueError("a leaf cannot also be a center (including self edges)")
    return edges


def _weighted_interval_median(lower, upper, weights):
    """One minimizer of sum_i w_i dist(t,[lower_i,upper_i]) on the line."""
    positive = weights > 0
    endpoints = np.concatenate((lower[positive], upper[positive]))
    # Rescaling preserves the median and avoids overflow in the weight sum.
    active_weights = weights[positive]
    endpoint_weights = np.tile(active_weights / active_weights.max(), 2)
    order = np.argsort(endpoints, kind="stable")
    cumulative = np.cumsum(endpoint_weights[order])
    median_index = np.searchsorted(cumulative, cumulative[-1] / 2, side="left")
    return float(endpoints[order[median_index]])


def star_extrema(lower, upper, centers, leaf_weights=None, attribute_weights=None):
    """Minimize and maximize a weighted distance sum on disjoint stars.

    The objective on a table Z is

        sum_e leaf_weights[e] * sum_k attribute_weights[k]
              * abs(Z[leaf_e, k] - Z[center_e, k]).

    ``lower`` and ``upper`` have identical shape (n,d), with finite entries and
    lower <= upper. ``centers`` is an iterable of (leaf,center) integer-index
    pairs. A leaf appears only once and cannot occur as any edge's center;
    many leaves may share a center. These structural restrictions apply even
    to zero-weight edges. Both weight vectors default to ones and may contain
    zeros. No class labels are needed; the caller establishes their meaning.

    The returned dictionary contains scalar ``minimum`` and ``maximum``,
    feasible full-table ``minimizer`` and ``maximizer`` arrays, and
    ``minimum_by_attribute`` / ``maximum_by_attribute``. These per-attribute
    contributions already include attribute weights and sum to each scalar.
    Isolated vertices and zero-weight attributes stay at their lower bounds.
    Inputs are not modified. Empty edge sets return zero and lower-bound
    tables, including when n or d is zero.

    For a fixed center, leaves project onto their intervals for minimization
    and choose farthest endpoints for maximization. The minimizing center is
    a weighted median of the leaf interval endpoints, projected onto its own
    interval. The maximizing center is one of its interval endpoints. Each
    leaf gives its weight to BOTH endpoints. Sorting costs at most
    O(d * sum_s r_s log(r_s)), in addition to O(n*d) output storage.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if lower.ndim != 2 or upper.shape != lower.shape:
        raise ValueError("lower and upper must have identical two-dimensional shapes")
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("interval bounds must be finite")
    if np.any(lower > upper):
        raise ValueError("lower interval bounds must not exceed upper bounds")
    n, d = lower.shape
    edges = _edge_indices(centers, n)
    edge_weights = _nonnegative_weights(leaf_weights, len(edges), "leaf_weights")
    attribute_weights = _nonnegative_weights(attribute_weights, d, "attribute_weights")

    minimizer = lower.copy()
    maximizer = lower.copy()
    stars = {}
    for edge_index, (_, center) in enumerate(edges):
        stars.setdefault(int(center), []).append(edge_index)

    for center, edge_indices in stars.items():
        edge_indices = np.asarray(edge_indices, dtype=int)
        leaves = edges[edge_indices, 0]
        weights = edge_weights[edge_indices]
        if not np.any(weights > 0):
            continue
        # The normalized weights suffice for comparisons and the median;
        # reported objectives are recomputed with the original weights below.
        comparison_weights = weights / weights.max()
        for k in np.flatnonzero(attribute_weights > 0):
            leaf_lower, leaf_upper = lower[leaves, k], upper[leaves, k]
            median = _weighted_interval_median(leaf_lower, leaf_upper, weights)
            center_min = np.clip(median, lower[center, k], upper[center, k])
            minimizer[center, k] = center_min
            minimizer[leaves, k] = np.clip(center_min, leaf_lower, leaf_upper)

            center_options = np.array([lower[center, k], upper[center, k]])
            far_distances = np.maximum(
                np.abs(leaf_lower[:, None] - center_options),
                np.abs(leaf_upper[:, None] - center_options),
            )
            center_max = center_options[np.argmax(comparison_weights @ far_distances)]
            maximizer[center, k] = center_max
            choose_lower = np.abs(leaf_lower - center_max) >= np.abs(leaf_upper - center_max)
            maximizer[leaves, k] = np.where(choose_lower, leaf_lower, leaf_upper)

    def contributions(table):
        differences = np.abs(table[edges[:, 0]] - table[edges[:, 1]])
        return (edge_weights @ differences) * attribute_weights

    minimum_by_attribute = contributions(minimizer)
    maximum_by_attribute = contributions(maximizer)
    return {
        "minimum": float(minimum_by_attribute.sum()),
        "maximum": float(maximum_by_attribute.sum()),
        "minimizer": minimizer,
        "maximizer": maximizer,
        "minimum_by_attribute": minimum_by_attribute,
        "maximum_by_attribute": maximum_by_attribute,
    }
