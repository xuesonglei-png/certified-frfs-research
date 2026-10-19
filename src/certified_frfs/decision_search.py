"""Bounded deterministic local search with explicitly matched objective ablations."""
import numpy as np
from .decision_coverage import certified_decision_coverage


def select_local_decisions(X, y, ranking, *, eps=.01, margin_scale=.05,
                           penalty=.02, budget=24, accuracy_only=False):
    X, y = np.asarray(X, float), np.asarray(y)
    d = X.shape[1]
    ranking = list(ranking)
    if sorted(ranking) != list(range(d)) or budget < 7 or margin_scale <= 0:
        raise ValueError('invalid ranking or search settings')
    if accuracy_only and eps != 0:
        raise ValueError('accuracy ablation requires nominal distances')
    cache = {}
    trace = []

    def evaluate(features):
        key = tuple(sorted(features))
        if key in cache:
            return cache[key]
        cert = certified_decision_coverage(X, y, key, eps=eps)
        if accuracy_only:
            objective = cert['balanced_coverage']-penalty*len(key)/d
        else:
            bounds = np.array([b if b is not None else -np.inf for b in cert['lower_bounds']])
            utility = np.clip(bounds/(margin_scale*len(key)/d), -1., 1.)
            classes = [utility[y == label].mean() for label in np.unique(y)]
            objective = .5*np.mean(classes)+.5*min(classes)-penalty*len(key)/d
        row = {'features': list(key), 'objective': float(objective),
               'balanced_coverage': cert['balanced_coverage'],
               'min_class_coverage': cert['min_class_coverage']}
        cache[key] = row
        trace.append(row)
        return row

    def value(row):
        return (round(row['objective'], 12), -len(row['features']),
                tuple(-a for a in row['features']))

    for fraction in [.1, .2, .35, .5, .65, .8, 1.]:
        evaluate(ranking[:max(1, int(np.ceil(fraction*d)))])
    current = max(trace, key=value)
    initial = current['features'].copy()
    while len(cache) < budget:
        selected = current['features']
        present = set(selected)
        # Weak retained and strong omitted features restrict the neighborhood.
        retained = [a for a in reversed(ranking) if a in present][:4]
        omitted = [a for a in ranking if a not in present][:4]
        moves = []
        if len(selected) > 1:
            moves += [[a for a in selected if a != old] for old in retained]
        moves += [selected+[new] for new in omitted]
        moves += [[a for a in selected if a != old]+[new]
                  for old in retained for new in omitted]
        best = current
        for move in moves:
            key = tuple(sorted(move))
            if key not in cache and len(cache) >= budget:
                break
            candidate = evaluate(move)
            if value(candidate) > value(best):
                best = candidate
        if best is current:
            break
        current = best
    certificate = certified_decision_coverage(X, y, current['features'], eps=eps)
    return {'features': current['features'], 'initial_features': initial,
            'trace': trace, 'evaluations': len(cache), 'budget': budget,
            'certificate': certificate, 'objective': current['objective'],
            'nonvacuous': certificate['min_class_coverage'] > 0,
            'scope': 'finite_table_search_not_global_optimum_or_test_guarantee'}
