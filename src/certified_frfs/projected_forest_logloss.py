"""True-label balanced logloss after fixed-teacher raw-coordinate imputation.

This exact training ranking objective does not bound test risk, improve
accuracy by theorem, or certify the separately refitted downstream model.
"""
import numpy as np
from scipy.special import logsumexp
from .multiknot_forest import memberships, validated_model, scores


class MultiknotProjectedLogloss:
    def __init__(self, X, y, fit):
        X, Z, W, b, V, classes, edges = validated_model(X, fit)
        y = np.asarray(y)
        if y.shape != (len(X),) or not np.all(np.isin(y, classes)) or not all(np.any(y == c) for c in classes):
            raise ValueError('reference must contain every model class')
        q, self.d = len(classes), X.shape[1]
        self.edges = [tuple(e) for e in edges]
        self.weights = np.array([1/(q*sum(y == c)) for c in y])
        self.own = np.argmax(y[:, None] == classes[None], axis=1)
        self.center = self.weights@X
        Zc = memberships(self.center[None], fit['knots'])[0]
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

    def projected_scores(self, features):
        fs = tuple(sorted(features))
        if len(set(fs)) != len(fs) or any(int(a) != a or not 0 <= a < self.d for a in fs):
            raise ValueError('invalid raw support')
        keep = set(fs)
        omitted = np.array([a not in keep for a in range(self.d)]+
                           [u not in keep or v not in keep for u, v in self.edges], float)
        return self.full_scores-np.einsum('ntq,t->nq', self.terms, omitted)

    def __call__(self, features):
        fs = tuple(sorted(features))
        if fs not in self.cache:
            value = self.projected_scores(fs)
            self.cache[fs] = float(self.weights@(logsumexp(value, axis=1)-value[np.arange(len(value)), self.own]))
        return self.cache[fs]
