"""Independent scalar coordinate replay with the stored-metric tie convention."""
import numpy as np


def friend_enemy_bounds(A, y, weights, anchor):
    friends = np.flatnonzero((y == y[anchor]) & (np.arange(len(y)) != anchor))
    if len(friends) < 3: return None
    distance = np.zeros(len(friends))
    for a, w in enumerate(weights): distance += abs(A[friends, a]*w-A[anchor, a]*w)
    friends = friends[np.lexsort((friends, distance))][:3]
    enemies = np.flatnonzero(y != y[anchor])
    lo, hi = np.maximum(0, A-.01), np.minimum(1, A+.01)
    totals = np.zeros((len(enemies), 3))
    for a, w in enumerate(weights):
        alternatives = []
        for t in (lo[anchor, a], hi[anchor, a]):
            near = np.maximum(0, np.maximum(lo[enemies, a]-t, t-hi[enemies, a]))
            far = np.maximum(abs(t-lo[friends, a]), abs(t-hi[friends, a]))
            alternatives.append(near[:, None]-far[None, :])
        totals += w*np.minimum(*alternatives)
    return totals.min(axis=1)
