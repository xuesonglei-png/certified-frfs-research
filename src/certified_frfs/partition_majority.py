"""Anchor-box partition with all-friend order-statistic sufficient bounds."""
import numpy as np
from .core import _inputs, interval_bounds


def cell_majority_lower(lo, hi, anchor, left, right, y, weights):
    """Third enemy distance lower minus third friend distance upper."""
    near = np.maximum(0., np.maximum(lo-right, left-hi))@weights
    far = np.maximum(np.abs(left-hi), np.abs(right-lo))@weights
    friends = (y == y[anchor]) & (np.arange(len(y)) != anchor)
    enemies = y != y[anchor]
    if friends.sum() < 3:
        return -np.inf
    enemy_third = np.partition(near[enemies], 2)[2] if enemies.sum() >= 3 else np.inf
    friend_third = np.partition(far[friends], 2)[2]
    return float(enemy_third-friend_third)


def partition_majority_bounds(X, y, weights, *, eps=.01, protected=None, max_leaves=16):
    """Cover anchor boxes by at most16cells; friends may change across cells.

    Every leaf bound must be positive for a certificate. Deterministic worst-
    leaf/widest-weighted-coordinate splits never discard an unresolved cell.
    This is conservative interval arithmetic, not a global exact optimizer.
    """
    X, y, _, weights = _inputs(X, y, None, weights)
    if int(max_leaves) != max_leaves or max_leaves < 1:
        raise ValueError('positive leaf budget required')
    lo, hi = interval_bounds(X, eps)
    ids = np.arange(len(y)) if protected is None else np.flatnonzero(np.asarray(protected, bool))
    result = np.full(len(y), -np.inf); counts = np.zeros(len(y), int)
    for i in ids:
        if np.sum(y == y[i]) < 4:
            continue
        left, right = lo[i].copy(), hi[i].copy()
        leaves = [(cell_majority_lower(lo, hi, i, left, right, y, weights), left, right)]
        while len(leaves) < max_leaves:
            worst = min(range(len(leaves)), key=lambda j: (leaves[j][0], j))
            value, left, right = leaves[worst]
            if value > 1e-10 or np.isneginf(value):
                break
            axis = int(np.argmax((right-left)*weights))
            if (right[axis]-left[axis])*weights[axis] <= 1e-15:
                break
            midpoint = (left[axis]+right[axis])/2
            child_right, child_left = right.copy(), left.copy()
            child_right[axis] = midpoint; child_left[axis] = midpoint
            leaves.pop(worst)
            for a, b in [(left, child_right), (child_left, right)]:
                leaves.append((cell_majority_lower(lo, hi, i, a, b, y, weights), a, b))
        bound = min(c[0] for c in leaves)
        result[i] = bound-1e-10*(1+abs(bound)) if np.isfinite(bound) else bound
        counts[i] = len(leaves)
    return result, counts
