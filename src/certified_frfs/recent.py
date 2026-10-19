"""Independent implementation of An et al. (FSS 513, 2025), SFRIP-FS.

Implements Eqs. (14)-(15), (37)-(44) and Algorithm 1 with explicit
configuration. This is not reproduction of the authors' result tables.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def sfrip_dependency_from_distances(distances, y, *, beta=3.0, k=10,
                                    support=3, confidence=0.8):
    """For any strictly decreasing radial similarity, Eq. 37 is Eq. 14.

    Include self in the closed neighborhood as the paper's U definition does.
    Equal enemy distances are ordered by object index; all boundary ties are
    included. Use all available enemies when their count is less than k.
    """
    distances = np.asarray(distances, dtype=float)
    y = np.asarray(y)
    n = len(y)
    if distances.shape != (n, n) or not np.isfinite(distances).all():
        raise ValueError("distances must be a finite square table")
    if beta <= 1 or k < 1 or int(k) != k or support < 1 or not 0 <= confidence <= 1:
        raise ValueError("invalid SFRIP configuration")
    same = y[:, None] == y[None, :]
    total = 0
    for i in range(n):
        enemies = distances[i, ~same[i]]
        if len(enemies) == 0:
            total += n if n >= support else 0
            continue
        radii = np.sort(enemies, kind="stable")[:int(k)]
        sorted_row = np.sort(distances[i])
        counts = np.searchsorted(sorted_row, radii, side="right")
        scores = counts - beta * np.arange(len(radii))
        radius = radii[int(np.argmax(scores))]
        neighbors = distances[i] <= radius
        normal = int(np.count_nonzero(neighbors & same[i]))
        size = int(np.count_nonzero(neighbors))
        if size >= support and normal / size >= confidence:
            total += normal
    return float(total / (n * n))


def select_sfrip(X, y, *, beta=3.0, k=10, support=3, confidence=0.8,
                 alpha=0.0):
    """Algorithm 1, Euclidean distance and native positive-gain stopping.

    Initial dependency is zero as specified in the pseudocode. Recompute
    core samples separately for each candidate. Break feature ties by index.
    The returned subset may be empty; evaluation must not silently fill it.
    """
    X, y = np.asarray(X, float), np.asarray(y)
    if X.ndim != 2 or len(y) != len(X) or not np.isfinite(X).all():
        raise ValueError("invalid training table")
    if alpha < 0:
        raise ValueError("alpha must be nonnegative")
    n, d = X.shape
    current_squared = np.zeros((n, n))
    selected, remaining, history = [], list(range(d)), []
    value = 0.0
    while remaining:
        best = None
        best_value = -np.inf
        best_squared = None
        for a in remaining:
            candidate = current_squared + (X[:, a, None] - X[None, :, a]) ** 2
            score = sfrip_dependency_from_distances(
                np.sqrt(candidate), y, beta=beta, k=k, support=support,
                confidence=confidence)
            if score > best_value:
                best, best_value, best_squared = a, score, candidate
        gain = best_value - value
        accepted = gain > alpha + 1e-12
        history.append({"feature": int(best), "dependency": best_value,
                        "gain": gain, "accepted": bool(accepted)})
        if not accepted:
            break
        selected.append(best)
        remaining.remove(best)
        current_squared, value = best_squared, best_value
    return {"features": selected, "history": history, "dependency": value,
            "configuration": {"beta": beta, "k": k, "support": support,
                              "confidence": confidence, "alpha": alpha},
            "implementation": "independent_equations_14_15_37_44_algorithm_1"}
