"""Exact projected-score MSE via a small Gram matrix of forest contributions."""
import numpy as np
from .triangular_classifier import triangular_memberships
from .forest_fuzzy_classifier import forest_scores


class ForestProjectionLoss:
    def __init__(self, X, y, classes, full_fit):
        X, y, classes = np.asarray(X, float), np.asarray(y), np.asarray(classes)
        if y.shape != (len(X),) or not all(np.any(y == c) for c in classes) or not np.all(np.isin(y, classes)):
            raise ValueError('reference must contain every declared class')
        q, d = len(classes), X.shape[1]; self.d = d
        self.edges = [tuple(e) for e in full_fit.get('edges', [])]
        W = np.asarray(full_fit['coef']); bias = np.asarray(full_fit['intercept'])
        V = np.asarray(full_fit.get('pair_coef', np.zeros((q, 0, 3, 3)))).reshape(q, len(self.edges), 3, 3)
        forest_scores(X[:1], W, bias, self.edges, V)  # Validate canonical forest.
        weights = np.array([1/(q*np.sum(y == label)) for label in y])
        self.center = weights@X
        Zc = triangular_memberships(self.center[None])[0]
        D = triangular_memberships(X)-Zc[None]
        unary = np.einsum('ndk,qdk->ndq', D, W)
        pairs = np.zeros((len(X), len(self.edges), q))
        for e, (u, v) in enumerate(self.edges):
            unary[:, u] += np.einsum('nk,qkl,l->nq', D[:, u], V[:, e], Zc[v])
            unary[:, v] += np.einsum('k,qkl,nl->nq', Zc[u], V[:, e], D[:, v])
            pairs[:, e] = np.einsum('nk,qkl,nl->nq', D[:, u], V[:, e], D[:, v])
        terms = np.concatenate([unary, pairs], axis=1)
        terms -= terms.mean(axis=2, keepdims=True)
        design = terms.transpose(0, 2, 1).reshape(len(X)*q, -1)
        self.gram = design.T@(np.repeat(weights/q, q)[:, None]*design)
        self.cache = {}

    def __call__(self, features):
        fs = list(features); key = tuple(fs)
        if len(set(fs)) != len(fs) or any(int(a) != a or not 0 <= a < self.d for a in fs):
            raise ValueError('invalid raw support')
        if key not in self.cache:
            keep = set(fs)
            omitted = np.array([a not in keep for a in range(self.d)]+[u not in keep or v not in keep for u, v in self.edges], float)
            # The mathematical Gram quadratic is nonnegative. Tiny negative
            # roundoff is clipped; ranking uses the declared 12-decimal ties.
            self.cache[key] = max(0., float(omitted@self.gram@omitted))
        return self.cache[key]
