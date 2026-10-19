"""Equivalent computation of documented FGS/SFRIP baseline conventions.

Caches label-independent FGS graphs and vectorizes SFRIP radius calculations.
No conventions, hyperparameters, stopping decisions or ties are changed.
This is optimization of existing independent reproductions, not author code.
"""
import numpy as np
from .graph_recent import entropy, graph_select


def sfrip_dependency_fast(distances, y, *, beta=3., k=10, support=3, confidence=.8):
    distances, y = np.asarray(distances, float), np.asarray(y)
    n = len(y)
    if distances.shape != (n, n) or not np.isfinite(distances).all():
        raise ValueError('finite square distance table required')
    if beta <= 1 or k < 1 or int(k) != k or support < 1 or not 0 <= confidence <= 1:
        raise ValueError('invalid SFRIP configuration')
    same = y[:, None] == y[None, :]
    radii = np.sort(np.where(same, np.inf, distances), axis=1, kind='stable')[:, :min(int(k), n)]
    ordered = np.sort(distances, axis=1)
    counts = np.stack([np.searchsorted(ordered[i], radii[i], side='right') for i in range(n)])
    values = counts-beta*np.arange(radii.shape[1])[None]
    values[~np.isfinite(radii)] = -np.inf
    radius = radii[np.arange(n), values.argmax(axis=1)]
    neighbors = distances <= radius[:, None]
    normal, size = (neighbors & same).sum(axis=1), neighbors.sum(axis=1)
    accepted = (size >= support) & (normal/size >= confidence)
    return float(normal[accepted].sum()/(n*n))


def sfrip_paths_grid(X, y, configs):
    X, y = np.asarray(X, float), np.asarray(y)
    n, d = X.shape
    if y.shape != (n,) or n < 2 or not np.isfinite(X).all():
        raise ValueError('invalid training table')
    differences = [(X[:, a, None]-X[None, :, a])**2 for a in range(d)]
    outputs = []
    for config in configs:
        alpha = config.get('alpha', 0.)
        settings = {k: config[k] for k in ('beta', 'confidence', 'k', 'support') if k in config}
        current, value, remaining = np.zeros((n, n)), 0., list(range(d))
        ranking, history, native, stopped = [], [], [], False
        while remaining:
            best, best_value, best_squared = None, -np.inf, None
            for a in remaining:
                candidate = current+differences[a]
                score = sfrip_dependency_fast(np.sqrt(candidate), y, **settings)
                if score > best_value:
                    best, best_value, best_squared = a, score, candidate
            gain = best_value-value
            accepted = gain > alpha+1e-12
            if not stopped and accepted:
                native.append(best)
            else:
                stopped = True
            ranking.append(best)
            remaining.remove(best)
            current, value = best_squared, best_value
            history.append({'feature': int(best), 'dependency': float(value), 'gain': float(gain),
                            'native_acceptance': bool(accepted and not stopped)})
        outputs.append({'native': {'features': native}, 'ranking': ranking, 'history': history,
                        'adaptation': 'continued_greedy_after_native_stopping_not_original_fixed_budget_algorithm'})
    return outputs


def fgs_paths_grid(X, y, configs):
    X, y = np.asarray(X, float), np.asarray(y)
    n, d = X.shape
    if n < 2 or d < 1 or y.shape != (n,) or not np.isfinite(X).all() or X.min() < 0 or X.max() > 1:
        raise ValueError('scaled training table required')
    D = (y[:, None] == y[None, :]).astype(float)
    hd, cache = entropy(D), {}
    outputs = []
    for config in configs:
        beta, xi = config['beta'], config['xi']
        if not 0 <= beta <= 1 or not np.isfinite(xi) or xi <= 0:
            raise ValueError('invalid FGS configuration')
        if xi not in cache:
            relations = []
            for a in range(d):
                difference = abs(X[:, a, None]-X[None, :, a])
                radius = X[:, a].std(ddof=1)/xi
                relations.append(np.where(difference <= radius, 1-difference, 0.))
            h = [entropy(r) for r in relations]
            joint_d = [entropy(np.minimum(r, D)) for r in relations]
            edges = []
            for u in range(d):
                for v in range(u+1, d):
                    pair = np.minimum(relations[u], relations[v])
                    weight = h[u]+h[v]+hd-entropy(pair)-joint_d[u]-joint_d[v]+entropy(np.minimum(pair, D))
                    edges.append((float(weight), u, v))
            # Sorted enemy complements are independent of beta too.
            ordered = [[np.sort((1-r[:, y != label]), axis=1) for label in np.unique(y)] for r in relations]
            sums = [[(1-r[:, y != label]).sum(axis=1) for label in np.unique(y)] for r in relations]
            cache[xi] = (edges, ordered, sums)
        edges, ordered, sums = cache[xi]
        cd = []
        for arrays, totals in zip(ordered, sums):
            lower = []
            for array, total in zip(arrays, totals):
                if array.shape[1] == 0:
                    lower.append(np.ones(n))
                else:
                    s1 = np.minimum(np.maximum(1, np.floor(total*(1-beta)).astype(int)), array.shape[1])
                    lower.append(array[np.arange(n), s1-1])
            cd.append(float(np.max(lower, axis=0).mean()))
        result = graph_select(cd, edges)
        native = {**result, 'cd': cd, 'beta': beta, 'xi': xi,
                  'implementation': 'independent_FGS_RFRAS_numeric_documented_boundary_convention'}
        chosen = set(native['features'])
        ranking = sorted(range(d), key=lambda a: (a not in chosen, -cd[a], a))
        outputs.append({'native': native, 'ranking': ranking,
                        'adaptation': 'native_members_first_then_descending_cd_not_original_budget_algorithm'})
    return outputs
