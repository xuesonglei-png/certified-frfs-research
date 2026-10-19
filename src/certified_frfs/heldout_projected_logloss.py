"""Exact raw-imputation loss on heldout observations with a training center."""
import numpy as np
from .multiknot_forest import validated_model, memberships, scores
from .projected_forest_logloss import MultiknotProjectedLogloss


class HeldoutProjectedLogloss(MultiknotProjectedLogloss):
    def __init__(self, X, y, fit, training_center):
        X, Z, W, b, V, classes, edges = validated_model(X, fit)
        y = np.asarray(y)
        center = np.asarray(training_center, float)
        if not len(X) or y.shape != (len(X),) or not np.all(np.isin(y, classes)):
            raise ValueError('invalid heldout labels')
        if center.shape != (X.shape[1],) or not np.all(np.isfinite(center)) or np.any((center < 0) | (center > 1)):
            raise ValueError('invalid training center')
        q, self.d = len(classes), X.shape[1]
        self.edges = [tuple(e) for e in edges]
        self.weights = np.array([1/(len(np.unique(y))*sum(y == c)) for c in y])
        self.own = np.argmax(y[:, None] == classes[None], axis=1)
        self.center = center.copy()
        Zc = memberships(center[None], fit['knots'])[0]
        difference = Z-Zc[None]
        unary = np.einsum('ndk,qdk->ndq', difference, W)
        pairs = np.zeros((len(X), len(edges), q))
        for e, (u, v) in enumerate(edges):
            unary[:, u] += np.einsum('nk,qkl,l->nq', difference[:, u], V[:, e], Zc[v])
            unary[:, v] += np.einsum('k,qkl,nl->nq', Zc[u], V[:, e], difference[:, v])
            pairs[:, e] = np.einsum('nk,qkl,nl->nq', difference[:, u], V[:, e], difference[:, v])
        self.terms = np.concatenate([unary, pairs], axis=1)
        self.full_scores = scores(X, fit)
        self.cache = {}


class CrossfitProjectedLogloss:
    def __init__(self, objectives):
        self.objectives = list(objectives)
        if not self.objectives:
            raise ValueError('no heldout folds')
        self.cache = {}

    def __call__(self, features):
        fs = tuple(sorted(features))
        if fs not in self.cache:
            self.cache[fs] = float(np.mean([o(fs) for o in self.objectives]))
        return self.cache[fs]
