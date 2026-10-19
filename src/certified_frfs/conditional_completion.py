"""Prediction-preserving local reducts through shared query completion bounds.

Query values enter only through an explicit acquisition callback. Reference
objects are fixed; unobserved query coordinates may take ANY value in [0,1].
This is a sufficient 1NN decision certificate, not a correctness certificate.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.spatial.distance import cdist


def pair_completion_lower(reference, enemy, friend, observed, values, weights=None):
    """Exact infimum of one enemy/friend margin over missing unit coordinates."""
    R = np.asarray(reference, float)
    d = R.shape[1]
    w = np.full(d, 1 / d) if weights is None else np.asarray(weights, float)
    seen = np.asarray(observed, int)
    missing = np.ones(d, dtype=bool)
    missing[seen] = False
    q = np.asarray(values, float)
    return float(np.dot(w[seen], np.abs(q - R[enemy, seen]) - np.abs(q - R[friend, seen]))
                 - np.dot(w[missing], np.abs(R[enemy, missing] - R[friend, missing])))


class ConditionalCompletion1NN:
    """A fixed full 1NN teacher with certified, read-once attribute acquisition.

    Shared-MI caches reference-only suffix distances (O(d*n^2) memory). Dynamic
    acquisition avoids that cache, but recomputes missing-coordinate distances.
    All storage and preparation costs must be reported alongside online costs.
    """
    def __init__(self, reference, labels, order, *, bound='shared', adaptive=False,
                 n_friends=3, weights=None, costs=None):
        start = time.perf_counter()
        self.R = np.asarray(reference, float).copy()
        self.y = np.asarray(labels).copy()
        if self.R.ndim != 2 or min(self.R.shape) == 0:
            raise ValueError('nonempty 2D reference required')
        self.n, self.d = self.R.shape
        if (self.y.shape != (self.n,) or not np.isfinite(self.R).all()
                or np.any(self.R < 0) or np.any(self.R > 1)):
            raise ValueError('reference must be finite in [0,1], labels aligned')
        self.order = np.asarray(order, int)
        if self.order.shape != (self.d,) or sorted(self.order.tolist()) != list(range(self.d)):
            raise ValueError('order must be a coordinate permutation')
        if bound not in ('shared', 'independent') or (adaptive and bound != 'shared'):
            raise ValueError('invalid bound/policy combination')
        if isinstance(n_friends, bool) or int(n_friends) != n_friends or n_friends < 1:
            raise ValueError('positive integer witness budget required')
        self.bound, self.adaptive, self.n_friends = bound, adaptive, int(n_friends)
        self.w = np.full(self.d, 1 / self.d) if weights is None else np.asarray(weights, float).copy()
        self.costs = np.ones(self.d) if costs is None else np.asarray(costs, float).copy()
        if (self.w.shape != (self.d,) or not np.isfinite(self.w).all()
                or np.any(self.w < 0) or self.w.sum() <= 0
                or self.costs.shape != (self.d,) or not np.isfinite(self.costs).all()
                or np.any(self.costs <= 0)):
            raise ValueError('nonnegative positive-mass weights and positive costs required')
        self.tol = 1e-10 * (1 + self.w.sum())
        self.suffix = None
        if self.bound == 'shared' and not self.adaptive:
            self.suffix = np.zeros((self.d + 1, self.n, self.n))
            for stage in range(self.d - 1, -1, -1):
                a = self.order[stage]
                self.suffix[stage] = (self.suffix[stage + 1]
                                     + self.w[a] * np.abs(self.R[:, a, None] - self.R[None, :, a]))
        self.cache_bytes = 0 if self.suffix is None else self.suffix.nbytes
        self.setup_seconds = time.perf_counter() - start

    def _check(self, distances, seen):
        nearest = int(np.argmin(distances))  # full teacher's tie convention
        target = self.y[nearest]
        enemies = np.flatnonzero(self.y != target)
        if not len(enemies):
            return True, target, None
        friends = np.flatnonzero(self.y == target)
        friends = friends[np.argsort(distances[friends], kind='stable')[:self.n_friends]]
        unknown = np.flatnonzero(~seen)
        if self.bound == 'independent':
            far = np.maximum(self.R[friends][:, unknown], 1 - self.R[friends][:, unknown]) @ self.w[unknown]
            lower = distances[enemies].min() - (distances[friends] + far).min()
            return bool(lower > self.tol), target, None
        if self.suffix is not None:
            penalties = self.suffix[int(seen.sum())][np.ix_(enemies, friends)]
        else:
            penalties = cdist(self.R[enemies][:, unknown] * self.w[unknown],
                              self.R[friends][:, unknown] * self.w[unknown], 'cityblock')
        margins = distances[enemies, None] - distances[friends][None, :] - penalties
        best = margins.max(axis=1)
        row = int(best.argmin())
        col = int(margins[row].argmax())
        return bool(best[row] > self.tol), target, (int(enemies[row]), int(friends[col]))

    def predict_one(self, acquire, *, trace=False):
        """acquire(a) supplies ONLY queried coordinate a, once per prediction."""
        seen = np.zeros(self.d, dtype=bool)
        values = np.full(self.d, np.nan)
        distances = np.zeros(self.n)
        history = []
        checks = 0
        for stage in range(self.d + 1):
            if stage == self.d:
                target = self.predict_full(values)
                certified = False  # full-read fallback; could be a tie
                break
            checks += 1
            certified, target, pair = self._check(distances, seen)
            if certified:
                break
            if self.adaptive and stage and pair is not None:
                e, f = pair
                scores = self.w * np.abs(self.R[e] - self.R[f]) / self.costs
                scores[seen] = -np.inf
                a = int(np.argmax(scores))
            else:
                a = int(next(a for a in self.order if not seen[a]))
            value = float(acquire(a))
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('acquired value must lie in [0,1]')
            distances += self.w[a] * np.abs(self.R[:, a] - value)
            seen[a] = True
            values[a] = value
            history.append(a)
        return {'prediction': target.item() if hasattr(target, 'item') else target,
                'acquired': history, 'count': len(history), 'cost': float(self.costs[seen].sum()),
                'early_certified': bool(certified), 'bound_checks': checks}

    def predict_full(self, query):
        """Same fixed distance model and first-reference tie convention."""
        q = np.asarray(query, float)
        distances = np.abs(self.R - q) @ self.w
        return self.y[int(np.argmin(distances))]
