"""Exact centered score-projection loss for arbitrary nodal fuzzy forests.

The omitted raw coordinates are replaced by a fixed training center. This
ranking loss does not certify a refitted classifier or population accuracy.
"""
import numpy as np
from .multiknot_forest import memberships, validated_model


class MultiknotProjectionLoss:
    def __init__(self, X, y, fit):
        X, Z, W, b, V, classes, edges = validated_model(X, fit)
        y = np.asarray(y)
        if y.shape != (len(X),) or not np.all(np.isin(y, classes)) or not all(np.any(y == c) for c in classes):
            raise ValueError('reference must contain every model class')
        q, self.d = len(classes), X.shape[1]
        self.edges = [tuple(e) for e in edges]
        alpha = np.array([1/(q*sum(y == c)) for c in y])
        self.center = alpha@X
        center_hat = memberships(self.center[None], fit['knots'])[0]
        difference = Z-center_hat[None]
        unary = np.einsum('ndk,qdk->ndq', difference, W)
        pairs = np.zeros((len(X), len(edges), q))
        for e, (u, v) in enumerate(edges):
            unary[:, u] += np.einsum('nk,qkl,l->nq', difference[:, u], V[:, e], center_hat[v])
            unary[:, v] += np.einsum('k,qkl,nl->nq', center_hat[u], V[:, e], difference[:, v])
            pairs[:, e] = np.einsum('nk,qkl,nl->nq', difference[:, u], V[:, e], difference[:, v])
        terms = np.concatenate([unary, pairs], axis=1)
        terms -= terms.mean(axis=2, keepdims=True)
        design = terms.transpose(0, 2, 1).reshape(len(X)*q, -1)
        self.gram = design.T@(np.repeat(alpha/q, q)[:, None]*design)
        self.cache = {}

    def __call__(self, features):
        fs = tuple(sorted(features))
        if len(set(fs)) != len(fs) or any(int(a) != a or not 0 <= a < self.d for a in fs):
            raise ValueError('invalid raw attribute support')
        if fs not in self.cache:
            keep = set(fs)
            omitted = np.array([a not in keep for a in range(self.d)]+
                               [u not in keep or v not in keep for u, v in self.edges], float)
            self.cache[fs] = max(0., float(omitted@self.gram@omitted))
        return self.cache[fs]
