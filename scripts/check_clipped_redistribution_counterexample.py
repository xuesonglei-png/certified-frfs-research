"""Exact scope diagnostic: clipped dependency is not 5NN correctness.

No training or model selection; this constructed table is not a benchmark.
Fraction arithmetic independently computes nominal memberships and votes.
Existing floating-point certificates are checked separately.
"""
from fractions import Fraction as F
from pathlib import Path
import json
import numpy as np
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.knn_query import certify_knn_queries

ROOT = Path(__file__).resolve().parents[1]


def distance(a, b, weights):
    return sum((w * abs(x-y) for x, y, w in zip(a, b, weights)), F(0))


def memberships(table, labels, weights):
    return [min(F(1), min(distance(x, z, weights)
            for z, c in zip(table, labels) if c != label))
            for x, label in zip(table, labels)]


def votes(table, labels, queries, weights, *, leave_one_out=False):
    result = []
    for i, query in enumerate(queries):
        order = sorted((j for j in range(len(table))
                        if not leave_one_out or i != j),
                       key=lambda j: (distance(query, table[j], weights), j))[:5]
        counts = {c: sum(labels[j] == c for j in order) for c in sorted(set(labels))}
        result.append(min(counts, key=lambda c: (-counts[c], c)))
    return result


def balanced_accuracy(labels, predictions):
    classes = sorted(set(labels))
    return sum((F(sum(a == b and a == c for a, b in zip(labels, predictions)),
                  sum(a == c for a in labels)) for c in classes), F(0))/len(classes)


def run():
    left = (F(0),)*4
    right = (F(1), F(0), F(0), F(0))
    middle = (F(1, 2), F(1), F(1), F(1))
    table = [left, left, right, right, middle, middle, middle, middle]
    labels = [0]*4+[1]*4
    full = [F(1, 2)]*4
    reduced = [F(2), F(0), F(0), F(0)]
    queries = [left, right, middle]
    truth = [0, 0, 1]
    pf, pr = memberships(table, labels, full), memberships(table, labels, reduced)
    assert pf == pr == [F(1)]*8
    loo_full = votes(table, labels, table, full, leave_one_out=True)
    loo_reduced = votes(table, labels, table, reduced, leave_one_out=True)
    query_full = votes(table, labels, queries, full)
    query_reduced = votes(table, labels, queries, reduced)
    assert balanced_accuracy(labels, loo_full) == balanced_accuracy(truth, query_full) == 1
    assert balanced_accuracy(labels, loo_reduced) == balanced_accuracy(truth, query_reduced) == F(1, 2)
    # One shared perturbed table attains every object's 1/25 dependency loss.
    witness = [(F(1, 100), *left[1:])]*2+[(F(99, 100), *right[1:])]*2
    witness += [(F(49, 100), *middle[1:])]*2+[(F(51, 100), *middle[1:])]*2
    assert all(abs(a-b) <= F(1, 100) for x, z in zip(table, witness) for a, b in zip(x, z))
    assert memberships(witness, labels, full) == [F(1)]*8
    assert memberships(witness, labels, reduced) == [F(24, 25)]*8
    assert votes(witness, labels, witness, reduced, leave_one_out=True) == loo_reduced
    X, y, Q = np.array(table, float), np.array(labels), np.array(queries, float)
    guard = weight_certificate(X, y, [0], np.array(full, float), np.array([2.]), eps=.01)
    assert abs(guard['dependency_certificate']['upper_bound']-.04) < 1e-8
    assert guard['retained_counts'] == [0, 4] and guard['required'] == [4, 4]
    assert not guard['feasible'] and not guard['majority_feasible']
    query_checks = {}
    for name, weights, prediction in [('full', full, query_full), ('reduced', reduced, query_reduced)]:
        features = [i for i, w in enumerate(weights) if w]
        check = certify_knn_queries(X, y, Q, np.array(prediction), features,
                                    weights=np.array(weights, float), eps=.01,
                                    k=5, n_friends=4, allowed_enemies=2)
        assert check['certified'] == [True]*3
        query_checks[name] = {**check, 'predictions': prediction,
                             'correct': [a == b for a, b in zip(prediction, truth)]}
    return {'passed': True, 'scope': 'constructed_finite_table_scope_diagnostic_not_empirical_superiority_or_novelty',
            'n': 8, 'd': 4, 'retained': [0], 'full_weights': [str(w) for w in full],
            'reduced_weights': [str(w) for w in reduced],
            'nominal_full_memberships': [str(p) for p in pf],
            'nominal_reduced_memberships': [str(p) for p in pr],
            'full_LOO_BA': '1', 'reduced_LOO_BA': '1/2',
            'full_query_BA': '1', 'reduced_query_BA': '1/2',
            'exact_box_worst_dependency_loss': '1/25',
            'attaining_shared_table': [[str(v) for v in row] for row in witness],
            'production_guard': guard, 'query_checks': query_checks}


if __name__ == '__main__':
    result = run()
    path = ROOT/'research/clipped-redistribution-counterexample.json'
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('passed', 'full_query_BA', 'reduced_query_BA',
                                          'exact_box_worst_dependency_loss')}, indent=2))
