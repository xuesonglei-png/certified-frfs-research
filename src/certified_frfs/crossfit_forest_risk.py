"""Reference-internal out-of-fold projected forest log loss.

Fixed fold teachers are projected by training-center raw substitution. This
ranking is a development heuristic: it is neither a risk bound nor a refitted
classifier guarantee. Every fold scales, centers, builds its forest and fits
its teacher using that fold's reference alone.
"""
import numpy as np
from scipy.special import logsumexp
from .nested_teacher_regularized import inner_splits
from .validated_pool import preprocess_fit
from .fuzzy_interaction_tree import interaction_tree
from .orthogonal_forest_logistic import fit_orthogonal_forest
from .triangular_classifier import triangular_memberships


class ProjectedForestScores:
    def __init__(self, query, fit, center):
        query, center = np.asarray(query, float), np.asarray(center, float)
        self.d = query.shape[1]; self.edges = [tuple(e) for e in fit['edges']]
        q = len(fit['classes']); W = np.asarray(fit['coef'])
        V = np.asarray(fit['pair_coef']).reshape(q, len(self.edges), 3, 3)
        Zc = triangular_memberships(center[None])[0]
        D = triangular_memberships(query)-Zc[None]
        self.base = np.einsum('dk,qdk->q', Zc, W)+np.asarray(fit['intercept'])
        unary = np.einsum('ndk,qdk->ndq', D, W); pairs = np.zeros((len(query), len(self.edges), q))
        for e, (u, v) in enumerate(self.edges):
            self.base += np.einsum('k,qkl,l->q', Zc[u], V[:, e], Zc[v])
            unary[:, u] += np.einsum('nk,qkl,l->nq', D[:, u], V[:, e], Zc[v])
            unary[:, v] += np.einsum('k,qkl,nl->nq', Zc[u], V[:, e], D[:, v])
            pairs[:, e] = np.einsum('nk,qkl,nl->nq', D[:, u], V[:, e], D[:, v])
        self.terms = np.concatenate([unary, pairs], axis=1)

    def __call__(self, features):
        fs = list(features)
        if len(set(fs)) != len(fs) or any(int(a) != a or not 0 <= a < self.d for a in fs):
            raise ValueError('invalid support')
        keep = set(fs)
        mask = np.array([a in keep for a in range(self.d)]+[u in keep and v in keep for u, v in self.edges], float)
        return self.base[None]+np.einsum('ntq,t->nq', self.terms, mask)


class CrossfitForestRisk:
    def __init__(self, raw, y, *, C=1., pair_penalty=1., seed=20261012):
        raw, y = np.asarray(raw, float), np.asarray(y)
        self.folds = []; self.records = []; self.cache = {}
        self.singleton_indices = np.flatnonzero(np.isin(y, np.unique(y, return_counts=True)[0][np.unique(y, return_counts=True)[1] == 1])).tolist()
        for a, b in inner_splits(y, seed=seed):
            A, B = preprocess_fit(raw[a], raw[b]); labels = y[a]
            edges = [] if pair_penalty is None else interaction_tree(A, labels)['edges']
            fit = fit_orthogonal_forest(A, labels, list(range(A.shape[1])), edges,
                C=C, pair_penalty=1. if pair_penalty is None else pair_penalty)
            classes = np.asarray(fit['classes'])
            weights = np.array([1/(len(classes)*sum(labels == c)) for c in labels])
            center = weights @ A
            own = np.argmax(y[b, None] == classes[None], axis=1)
            validation_classes = np.unique(y[b])
            balance = np.array([1/(len(validation_classes)*sum(y[b] == c)) for c in y[b]])
            self.folds.append((ProjectedForestScores(B, fit, center), own, balance))
            self.records.append({'reference_indices': a.tolist(), 'validation_indices': b.tolist(),
                'fit': fit, 'center': center.tolist(), 'query_classes': validation_classes.tolist()})

    def __call__(self, features):
        key = tuple(features)
        if key not in self.cache:
            values = []
            for projected, own, balance in self.folds:
                scores = projected(features)
                values.append(float(balance @ (logsumexp(scores, axis=1)-scores[np.arange(len(own)), own])))
            self.cache[key] = float(np.mean(values))
        return self.cache[key]
