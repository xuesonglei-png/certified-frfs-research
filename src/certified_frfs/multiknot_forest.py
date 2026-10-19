"""Uniform piecewise-linear fuzzy forests with exact shared-coordinate boxes.

More knots expand the score family, not a guarantee of predictive improvement.
Exact extrema follow from multi-affinity on cells and finite-state forest DP.
Neither training reference FRS preservation nor teacher retention is implied.
"""
import numpy as np
from scipy.linalg import helmert
from sklearn.linear_model import LogisticRegression
from .linear_retention import linear_parameters


def uniform_knots(count):
    if isinstance(count, (bool, np.bool_)) or int(count) != count or count < 2:
        raise ValueError('at least two integer knots required')
    return np.linspace(0., 1., int(count))


def memberships(X, knots):
    X, knots = np.asarray(X, float), np.asarray(knots, float)
    if knots.ndim != 1 or len(knots) < 2 or not np.isfinite(knots).all():
        raise ValueError('invalid knot vector')
    if knots[0] != 0. or knots[-1] != 1. or np.any(np.diff(knots) <= 0):
        raise ValueError('knots must strictly partition [0,1]')
    if not np.isfinite(X).all() or np.any((X < 0.) | (X > 1.)):
        raise ValueError('coordinates must be finite in [0,1]')
    # Linear interpolation of adjacent nodal hats works for nonuniform knots too.
    left = np.clip(np.searchsorted(knots, X, side='right')-1, 0, len(knots)-2)
    fraction = (X-knots[left])/(knots[left+1]-knots[left])
    result = np.zeros(X.shape+(len(knots),))
    np.put_along_axis(result, left[..., None], (1-fraction)[..., None], axis=-1)
    np.put_along_axis(result, (left+1)[..., None], fraction[..., None], axis=-1)
    return result


def validated_model(X, fit):
    X = np.asarray(X, float)
    if X.ndim != 2 or X.shape[1] < 1:
        raise ValueError('nonempty feature matrix required')
    Z = memberships(X, fit['knots'])
    n, d, m = Z.shape
    classes = np.asarray(fit['classes'])
    q = len(classes)
    W, b = np.asarray(fit['coef'], float), np.asarray(fit['intercept'], float)
    if q < 2 or len(np.unique(classes)) != q or W.shape != (q, d, m) or b.shape != (q,):
        raise ValueError('invalid class or unary coefficients')
    parents, edges = list(range(d)), []
    def root(a):
        while parents[a] != a:
            a = parents[a]
        return a
    for edge in fit['edges']:
        if len(edge) != 2 or any(int(a) != a for a in edge):
            raise ValueError('integer edge endpoints required')
        u, v = map(int, edge)
        if not 0 <= u < v < d or (u, v) in edges:
            raise ValueError('unique canonical edges required')
        a, c = root(u), root(v)
        if a == c:
            raise ValueError('interaction graph must be a forest')
        parents[a] = c
        edges.append([u, v])
    V = np.asarray(fit['pair_coef'], float).reshape(q, len(edges), m, m)
    if not all(np.isfinite(t).all() for t in (W, b, V)):
        raise ValueError('finite model coefficients required')
    return X, Z, W, b, V, classes, edges


def scores(X, fit):
    X, Z, W, b, V, classes, edges = validated_model(X, fit)
    result = np.einsum('ndk,qdk->nq', Z, W)+b
    for e, (u, v) in enumerate(edges):
        result += np.einsum('nk,nl,qkl->nq', Z[:, u], Z[:, v], V[:, e])
    return result


def box_margins(X, y, fit, *, eps=.01, witnesses=False):
    X, _, W, b, V, classes, edges = validated_model(X, fit)
    y = np.asarray(y)
    n, d = X.shape
    q, m = len(classes), len(fit['knots'])
    if y.shape != (n,) or not np.all(np.isin(y, classes)):
        raise ValueError('invalid target labels')
    if not np.isfinite(eps) or eps < 0:
        raise ValueError('finite nonnegative radius required')
    own = np.argmax(y[:, None] == classes, axis=1)
    lo, hi = np.maximum(0., X-eps), np.minimum(1., X+eps)
    # Endpoints plus all interior knots clipped into each coordinate interval.
    # Repeated states are harmless and keep a rectangular DP representation.
    points = np.concatenate([lo[..., None], hi[..., None],
        np.clip(np.asarray(fit['knots'])[None, None, 1:-1], lo[..., None], hi[..., None])], axis=-1)
    Z = memberships(points, fit['knots'])
    unary = np.einsum('nqdk,ndsk->nqds', W[own, None]-W[None], Z)
    adjacency, potentials = [[] for _ in range(d)], []
    for e, (u, v) in enumerate(edges):
        C = V[own, e][:, None]-V[None, :, e]
        potentials.append(np.einsum('nqkl,nsk,ntl->nqst', C, Z[:, u], Z[:, v]))
        adjacency[u].append((v, e))
        adjacency[v].append((u, e))
    parent, edge_to_parent = np.full(d, -2, int), np.full(d, -1, int)
    order, roots = [], []
    for start in range(d):
        if parent[start] != -2:
            continue
        roots.append(start)
        parent[start] = -1
        stack = [start]
        while stack:
            u = stack.pop()
            order.append(u)
            for v, e in adjacency[u]:
                if v != parent[u]:
                    parent[v], edge_to_parent[v] = u, e
                    stack.append(v)
    cost, choices = unary.copy(), {}
    for child in reversed(order):
        p = parent[child]
        if p < 0:
            continue
        e = edge_to_parent[child]
        potential = potentials[e] if edges[e][0] == p else potentials[e].swapaxes(-1, -2)
        total = potential+cost[:, :, child, None, :]
        choices[child] = total.argmin(axis=-1)
        cost[:, :, p] += total.min(axis=-1)
    pair = b[own, None]-b[None]
    states = np.zeros((n, q, d), int)
    for root_node in roots:
        pair += cost[:, :, root_node].min(axis=-1)
        states[:, :, root_node] = cost[:, :, root_node].argmin(axis=-1)
    pair[np.arange(n), own] = np.inf
    margins = pair.min(axis=1)
    if not witnesses:
        return margins
    for child in order:
        p = parent[child]
        if p >= 0:
            states[:, :, child] = np.take_along_axis(choices[child], states[:, :, p, None], axis=-1)[..., 0]
    rivals = pair.argmin(axis=1)
    selected = states[np.arange(n), rivals]
    raw = np.take_along_axis(points, selected[..., None], axis=-1)[..., 0]
    return {'margins': margins, 'pair_margins': pair, 'rivals': rivals, 'witness_raw': raw}


def fit_multiknot_forest(X, y, features, edges, *, knot_count=5, C=1., pair_penalty=1.):
    X, y = np.asarray(X, float), np.asarray(y)
    knots, fs = uniform_knots(knot_count), list(features)
    if X.ndim != 2 or y.shape != (len(X),) or not fs or len(set(fs)) != len(fs):
        raise ValueError('invalid training matrix or feature support')
    if any(int(a) != a or not 0 <= a < X.shape[1] for a in fs):
        raise ValueError('invalid feature indices')
    if not np.isfinite(C) or C <= 0 or not np.isfinite(pair_penalty) or pair_penalty <= 0:
        raise ValueError('positive regularization settings required')
    classes, d, m = np.unique(y), X.shape[1], len(knots)
    q, r = len(classes), m-1
    stub = {'knots': knots.tolist(), 'classes': classes.tolist(), 'edges': list(edges),
            'coef': np.zeros((q, d, m)), 'intercept': np.zeros(q),
            'pair_coef': np.zeros((q, len(edges), m, m))}
    _, _, _, _, _, _, raw_edges = validated_model(X, stub)
    retained = [e for e in raw_edges if all(a in fs for a in e)]
    contrast = helmert(m, full=False).T
    Z = memberships(X, knots)@contrast
    blocks = [Z[:, fs].reshape(len(X), -1)]
    for u, v in retained:
        blocks.append((Z[:, u, :, None]*Z[:, v, None, :]).reshape(len(X), r*r)/np.sqrt(pair_penalty))
    model = LogisticRegression(C=C, class_weight='balanced', max_iter=4000,
                               tol=1e-8, random_state=0).fit(np.column_stack(blocks), y)
    coef, bias, classes = linear_parameters(model)
    W, V = np.zeros((q, d, m)), np.zeros((q, len(retained), m, m))
    W[:, fs] = coef[:, :r*len(fs)].reshape(q, len(fs), r)@contrast.T
    for e in range(len(retained)):
        tensor = coef[:, r*len(fs)+r*r*e:r*len(fs)+r*r*(e+1)].reshape(q, r, r)/np.sqrt(pair_penalty)
        V[:, e] = np.einsum('ki,qij,lj->qkl', contrast, tensor, contrast)
    W -= W[:1].copy()
    bias -= bias[:1].copy()
    V -= V[:1].copy()
    return {'knots': knots.tolist(), 'classes': classes.tolist(), 'features': fs, 'edges': retained,
            'coef': W.tolist(), 'pair_coef': V.tolist(), 'intercept': bias.tolist(),
            'C': float(C), 'pair_penalty': float(pair_penalty), 'class_balanced': True,
            'teacher_guards_enforced': False, 'design_columns': r*len(fs)+r*r*len(retained),
            'optimizer_iterations': int(model.n_iter_.max()),
            'optimizer_converged': int(model.n_iter_.max()) < model.max_iter,
            'scope': 'finite_element_fuzzy_forest_classifier_not_certified_reduct_or_prediction_guarantee'}
