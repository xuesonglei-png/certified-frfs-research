"""Finite-table robust leave-one-out margins with shared triplet coordinates.

This is a separate third-cycle target. It certifies strict nearest-neighbour
decisions on specified training objects, not test accuracy or generalization.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linprog
from .core import _inputs, _distance, interval_bounds


def triplet_interval_min(anchor_lower, anchor_upper, enemy_lower, enemy_upper,
                         friend_lower, friend_upper):
    """Exact min |a-e|-|a-h| for independent scalar intervals; broadcasting."""
    def at(a):
        near = np.maximum(0., np.maximum(enemy_lower-a, a-enemy_upper))
        far = np.maximum(np.abs(a-friend_lower), np.abs(a-friend_upper))
        return near-far
    return np.minimum(at(anchor_lower), at(anchor_upper))


def mixture_enemy_bound(lower, upper, anchor, enemy, friends, features,
                        weights, time_limit=2.):
    """Maximize a simplex witness-mixture lower bound for one enemy.

    Any returned feasible mixture is re-evaluated by exact scalar piecewise
    minima, so a solver's primal objective is never used as a min-LP bound.
    One mixture is shared across all attributes. No nearest-neighbour
    stability is assumed; midpoint separation is an exactness condition only.
    """
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    features, friends = np.asarray(features, int), np.asarray(friends, int)
    weights = np.asarray(weights, float)
    if lower.shape != upper.shape or lower.ndim != 2 or np.any(lower > upper):
        raise ValueError('invalid box')
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError('box must be finite')
    if len(friends) == 0 or len(set(friends)) != len(friends) or anchor in friends or enemy in friends:
        raise ValueError('nonempty distinct friend indices must exclude anchor/enemy')
    if (anchor == enemy or min([anchor, enemy, *friends]) < 0
            or max([anchor, enemy, *friends]) >= len(lower)):
        raise ValueError('invalid object indices')
    if (len(features) == 0 or len(set(features)) != len(features) or features.min() < 0
            or features.max() >= lower.shape[1] or weights.shape != (lower.shape[1],)
            or not np.isfinite(weights).all() or np.any(weights < 0)):
        raise ValueError('invalid selected attributes/weights')
    r, m = len(friends), len(features)
    blocks, constraints, rhs = [], [], []
    separated = True
    for t, k in enumerate(features):
        lo, hi = lower[anchor, k], upper[anchor, k]
        centers = (lower[friends, k]+upper[friends, k])/2
        separated &= not np.any((centers > lo) & (centers < hi))
        points = np.unique(np.clip(np.r_[lo, hi, lower[enemy, k], upper[enemy, k], centers], lo, hi))
        near = np.maximum(0., np.maximum(lower[enemy, k]-points, points-upper[enemy, k]))
        far = np.maximum(np.abs(points[:, None]-lower[friends, k]),
                         np.abs(points[:, None]-upper[friends, k]))
        blocks.append((near, far))
        for a, b in zip(near, far):
            row = np.zeros(r+m)
            row[:r], row[r+t] = b, 1.
            constraints.append(row); rhs.append(a)

    def evaluate(lam):
        return float(sum(weights[k]*np.min(near-far@lam)
                         for k, (near, far) in zip(features, blocks)))

    unit = np.eye(r)
    initial = np.array([evaluate(lam) for lam in unit])
    best = unit[int(initial.argmax())]
    best_value = float(initial.max())
    objective = np.r_[np.zeros(r), -weights[features]]
    options = {} if time_limit is None else {'time_limit': float(time_limit)}
    solved = linprog(objective, A_ub=np.array(constraints), b_ub=np.array(rhs),
                     A_eq=np.array([np.r_[np.ones(r), np.zeros(m)]]), b_eq=[1.],
                     bounds=[(0., 1.)]*r+[(None, None)]*m, method='highs', options=options)
    if solved.x is not None and np.isfinite(solved.x[:r]).all():
        lam = np.maximum(solved.x[:r], 0.)
        if lam.max() > 0:
            lam /= lam.max()
            lam /= lam.sum()
            value = evaluate(lam)
            if value > best_value:
                best, best_value = lam, value
    return {'lower_bound': best_value-1e-10*(1.+abs(best_value)),
            'raw_feasible_mixture_value': best_value, 'mixture': best.tolist(),
            'onehot_bound': float(initial.max()), 'friends': friends.tolist(),
            'solver_success': bool(solved.success), 'solver_status': int(solved.status),
            'midpoint_separation': bool(separated),
            'exactness_scope': 'Exact for this enemy if midpoint separation holds and all same-class reference objects are included',
            'numerical_guarantee': 'floating_point_not_formal_verification'}


class MarginWorkspace:
    """Cache signed triplet coefficients for a declared same-class pool."""
    def __init__(self, X, y, weights=None, eps=.01, clip=True, n_friends=3):
        X = np.asarray(X, float)
        if weights is None:
            weights = np.full(X.shape[1], 1./X.shape[1])
        self.X, self.y, self.full, self.weights = _inputs(X, y, None, weights)
        if not isinstance(n_friends, int) or n_friends < 1:
            raise ValueError('n_friends must be a positive integer')
        if self.weights.sum() <= 0:
            raise ValueError('positive weight mass required')
        self.lower, self.upper = interval_bounds(X, eps, clip)
        self.cross = self.y[:, None] != self.y[None, :]
        n, d = X.shape
        self.applicable = np.array([np.sum(self.y == yi) >= 2 and np.any(self.y != yi) for yi in self.y])
        distances = _distance(X, self.full, self.weights)
        np.fill_diagonal(distances, np.inf)
        self.friends = np.zeros((n, n_friends), int)
        for i in range(n):
            choices = np.flatnonzero((self.y == self.y[i]) & (np.arange(n) != i))
            if len(choices):
                choices = choices[np.lexsort((choices, distances[i, choices]))][:n_friends]
                self.friends[i] = np.resize(choices, n_friends)
            else:
                self.friends[i] = i  # masked as not applicable; never certified
        self.coefficients = np.empty((d, n, n, n_friends))
        for k in range(d):
            self.coefficients[k] = self.weights[k] * triplet_interval_min(
                self.lower[:, k, None, None], self.upper[:, k, None, None],
                self.lower[None, :, k, None], self.upper[None, :, k, None],
                self.lower[self.friends, k][:, None, :], self.upper[self.friends, k][:, None, :])

    def from_scores(self, scores):
        pair_lower = scores.max(axis=2)
        pair_lower = np.where(self.cross, pair_lower, np.inf)
        bounds = pair_lower.min(axis=1)
        bounds[~self.applicable] = -np.inf
        return bounds-1e-10*(1.+np.abs(np.where(np.isfinite(bounds), bounds, 0.)))

    def bounds(self, features):
        return self.from_scores(self.coefficients[list(features)].sum(axis=0))

    def select(self, order, protected=None):
        if sorted(order) != list(self.full):
            raise ValueError('order must be a full permutation')
        features = list(self.full)
        scores = self.coefficients.sum(axis=0)
        full_bounds = self.from_scores(scores)
        if protected is None:
            protected = self.applicable & (full_bounds > 1e-10)
        protected = np.asarray(protected, bool)
        if protected.shape != (len(self.y),) or np.any(protected & (full_bounds <= 1e-10)):
            raise ValueError('protected objects must be certified by the full reference')
        history = []
        # With no protected object, no meaningful compression target exists.
        if protected.any():
            for k in order:
                if len(features) == 1:
                    break
                candidate_scores = scores-self.coefficients[k]
                lower = self.from_scores(candidate_scores)
                accepted = bool(np.all(lower[protected] > 1e-10))
                history.append({'removed_candidate': int(k), 'accepted': accepted,
                                'protected_min_margin_lower': float(lower[protected].min())})
                if accepted:
                    features.remove(k)
                    scores = candidate_scores
        final_bounds = self.from_scores(scores)
        assert np.all(final_bounds[protected] > 1e-10)
        cover = []
        for label in np.unique(self.y):
            ids = self.y == label
            cover.append({'class': label.item(), 'n': int(ids.sum()),
                          'applicable': int(self.applicable[ids].sum()),
                          'protected': int(protected[ids].sum()),
                          'certified_after': int(np.sum(final_bounds[ids] > 1e-10)),
                          'protected_fraction': float(protected[ids].mean())})
        return {'features': [int(k) for k in features], 'history': history,
                'protected_indices': np.flatnonzero(protected).tolist(),
                'full_margin_lower': [float(v) if np.isfinite(v) else None for v in full_bounds],
                'final_margin_lower': [float(v) if np.isfinite(v) else None for v in final_bounds],
                'coverage': cover, 'protected_fraction': float(protected.mean()),
                'min_class_protected_fraction': min(r['protected_fraction'] for r in cover),
                'minimality_proven': False, 'target': 'strict robust LOO decisions on full-certified objects',
                'n_friends': self.friends.shape[1], 'full_reference_pool': self.friends.tolist()}
