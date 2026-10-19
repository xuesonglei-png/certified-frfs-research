"""Pairwise triangular fuzzy scores and exact raw-box forest separation.

This is a representation-capacity prototype, not a predictive result or a
claim of novelty over graphical models. All interval extrema use shared raw
coordinates; fuzzy memberships are never treated as independent intervals.
"""
import numpy as np
from .triangular_classifier import triangular_memberships


def _model(X, W, b, edges, pair_coef):
    X, W, b = np.asarray(X, float), np.asarray(W, float), np.asarray(b, float)
    if X.ndim != 2 or X.shape[1] < 1:
        raise ValueError('nonempty feature matrix required')
    Z = triangular_memberships(X)
    q, d = len(b), X.shape[1]
    if q < 2 or W.shape != (q, d, 3) or b.shape != (q,):
        raise ValueError('invalid additive coefficients')
    raw_edges = list(edges)
    pairs = []
    parents = list(range(d))
    def root(a):
        while parents[a] != a:
            a = parents[a]
        return a
    for edge in raw_edges:
        if len(edge) != 2 or any(int(a) != a for a in edge):
            raise ValueError('integer edge endpoints required')
        u, v = map(int, edge)
        if not 0 <= u < v < d or (u, v) in pairs:
            raise ValueError('unique canonical edges required')
        a, c = root(u), root(v)
        if a == c:
            raise ValueError('interaction graph must be a forest')
        parents[a] = c
        pairs.append((u, v))
    V = np.asarray(pair_coef, float)
    if V.shape != (q, len(pairs), 3, 3) or not np.isfinite(W).all() or not np.isfinite(b).all() or not np.isfinite(V).all():
        raise ValueError('invalid pair coefficients')
    return X, W, b, pairs, V, Z


def forest_scores(X, W, b, edges, pair_coef):
    X, W, b, edges, V, Z = _model(X, W, b, edges, pair_coef)
    scores = np.einsum('ndk,qdk->nq', Z, W)+b
    for e, (u, v) in enumerate(edges):
        scores += np.einsum('nk,nl,qkl->nq', Z[:, u], Z[:, v], V[:, e])
    return scores


def forest_box_margins(X, y, W, b, classes, edges, pair_coef, *, eps=.01, witnesses=False):
    X, W, b, edges, V, _ = _model(X, W, b, edges, pair_coef)
    y, classes = np.asarray(y), np.asarray(classes)
    n, d = X.shape; q = len(b)
    if classes.shape != (q,) or len(np.unique(classes)) != q or y.shape != (n,) or not np.all(np.isin(y, classes)):
        raise ValueError('invalid labels or classes')
    if not np.isfinite(eps) or eps < 0:
        raise ValueError('invalid radius')
    own = np.argmax(y[:, None] == classes, axis=1)
    lo, hi = np.maximum(0, X-eps), np.minimum(1, X+eps)
    points = np.stack([lo, hi, np.clip(.5, lo, hi)], axis=-1)
    Z = triangular_memberships(points)
    contrast = W[own, None]-W[None]
    unary = np.einsum('nqdk,ndsk->nqds', contrast, Z)
    adjacency = [[] for _ in range(d)]; potentials = []
    for e, (u, v) in enumerate(edges):
        C = V[own, e][:, None]-V[None, :, e]
        potentials.append(np.einsum('nqkl,nsk,ntl->nqst', C, Z[:, u], Z[:, v]))
        adjacency[u].append((v, e)); adjacency[v].append((u, e))
    parent = np.full(d, -2, int); edge_to_parent = np.full(d, -1, int)
    order, roots = [], []
    for start in range(d):
        if parent[start] != -2:
            continue
        roots.append(start); parent[start] = -1; stack = [start]
        while stack:
            u = stack.pop(); order.append(u)
            for v, e in adjacency[u]:
                if v == parent[u]:
                    continue
                parent[v] = u; edge_to_parent[v] = e; stack.append(v)
    cost = unary.copy(); choices = {}
    for child in reversed(order):
        p = parent[child]
        if p < 0:
            continue
        e = edge_to_parent[child]
        potential = potentials[e] if edges[e][0] == p else potentials[e].swapaxes(-1, -2)
        total = potential+cost[:, :, child, None, :]
        choices[child] = total.argmin(axis=-1)
        cost[:, :, p] += total.min(axis=-1)
    pair_margins = b[own, None]-b[None]
    states = np.zeros((n, q, d), int)
    for root_node in roots:
        pair_margins += cost[:, :, root_node].min(axis=-1)
        states[:, :, root_node] = cost[:, :, root_node].argmin(axis=-1)
    pair_margins[np.arange(n), own] = np.inf
    margins = pair_margins.min(axis=1)
    if not witnesses:
        return margins
    for child in order:
        p = parent[child]
        if p >= 0:
            states[:, :, child] = np.take_along_axis(choices[child], states[:, :, p, None], axis=-1)[..., 0]
    rivals = pair_margins.argmin(axis=1)
    selected = states[np.arange(n), rivals]
    raw = np.take_along_axis(points, selected[:, :, None], axis=-1)[..., 0]
    return {'margins': margins, 'rivals': rivals, 'witness_raw': raw, 'pair_margins': pair_margins}
