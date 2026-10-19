"""Scalar-row equation replay, with no DSNFRS production imports."""
import numpy as np
from audit_forest_compact_transfer import equivalent


def dependency(A, y, features, k, beta):
    A, y = np.asarray(A), np.asarray(y)
    variance = {}
    for c in np.unique(y):
        group = A[y == c]
        v = np.var(group, axis=0, ddof=1) if len(group) > 1 else np.zeros(A.shape[1])
        variance[c] = np.where(v == 0, 1e-6, v)
    lower = []
    for i in range(len(A)):
        distances = np.zeros(len(A))
        for a in sorted(features):
            # Target label determines every denominator; no production matrix.
            den = np.array([variance[c][a] for c in y])
            distances += (A[i, a]-A[:, a])**2/den
        ids = np.flatnonzero(y != y[i])
        ids = ids[np.argsort(distances[ids], kind='stable')][:k]
        supports = [int(np.sum(distances[y == y[i]] < distances[j])) for j in ids]
        scores = [count-beta*t for t, count in enumerate(supports)]
        chosen = max(range(len(ids)), key=lambda t: (scores[t], -t))
        lower.append(-np.expm1(-distances[ids[chosen]]))
    return float(np.mean(lower))


def check_fit(A, y, stored, counts):
    recipe, search = stored['recipe'], stored['search']; conv = search['conventions']
    assert conv['k'] == recipe['k'] and conv['beta'] == recipe['beta']
    assert conv['variance_ddof'] == 1 and conv['zero_variance_lambda'] == 1e-6
    assert conv['singleton_variance'] == 'zero'
    assert search['mode'] == 'native_gain_stopping' and search['requested_count'] is None
    assert search['delta'] == recipe['delta'] and search['empty_subset_dependency'] == 0
    selected, value = [], 0.
    for row in search['trace']:
        assert row['selected_before'] == selected; equivalent(value, row['base_dependency'])
        remaining = [a for a in range(A.shape[1]) if a not in selected]
        assert [q['feature'] for q in row['candidates']] == remaining
        fresh = []
        for q in row['candidates']:
            score = dependency(A, y, selected+[q['feature']], recipe['k'], recipe['beta'])
            equivalent(score, q['dependency']); equivalent(score-value, q['gain'])
            fresh.append({'feature': q['feature'], 'dependency': score, 'gain': score-value})
            counts['independent_DSN_candidate_dependencies'] += 1
        best = min(fresh, key=lambda q: (-q['gain'], q['feature']))
        assert row['chosen_feature'] == best['feature']
        accepted = best['gain'] > recipe['delta']; assert row['accepted'] == accepted
        if not accepted:
            assert row is search['trace'][-1] and search['status'] == 'gain_not_above_delta'
            break
        selected.append(best['feature']); value = best['dependency']
    assert selected == stored['selected_order'] and sorted(selected) == stored['features']
    assert stored['actual_count'] == len(selected)
    equivalent(search['final_dependency'], value)
    if search['status'] == 'attributes_exhausted': assert len(selected) == A.shape[1]
    else: assert search['status'] == 'gain_not_above_delta'
    w = (np.ones(len(selected))*np.sqrt(A.shape[1])/len(selected)).tolist() if selected else []
    equivalent(w, stored['weights']); counts['independent_DSN_native_fits'] += 1
