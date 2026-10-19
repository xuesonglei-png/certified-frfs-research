"""Exact margin/prediction/objective replay plus independent forest-energy LP."""
import hashlib
import json
import numpy as np
from scipy.optimize import linprog
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from review_box_iterations import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.triangular_classifier import triangular_memberships, triangular_box_margins
from certified_frfs.fuzzy_interaction_tree import interaction_tree
from certified_frfs.forest_fuzzy_classifier import forest_scores, forest_box_margins


def lp_margin(x, own, rival, W, b, edges, V, eps=.01):
    d = len(x); p = 3*d+9*len(edges)
    lo, hi = np.maximum(0, x-eps), np.minimum(1, x+eps)
    points = np.stack([lo, hi, np.clip(.5, lo, hi)], axis=-1)
    # Independent hat formula, without the message-passing implementation.
    Z = np.stack([np.maximum(0, 1-2*points), 1-np.abs(2*points-1), np.maximum(0, 2*points-1)], axis=-1)
    costs = np.zeros(p); costs[:3*d] = np.einsum('dk,dsk->ds', W[own]-W[rival], Z).ravel()
    equations, rhs = [], []
    for a in range(d):
        row = np.zeros(p); row[3*a:3*a+3] = 1
        equations.append(row); rhs.append(1.)
    for e, (u, v) in enumerate(edges):
        start = 3*d+9*e
        costs[start:start+9] = np.einsum('sk,tl,kl->st', Z[u], Z[v], V[own, e]-V[rival, e]).ravel()
        for s in range(3):
            row = np.zeros(p); row[start+3*s:start+3*s+3] = 1; row[3*u+s] = -1
            equations.append(row); rhs.append(0.)
        for t in range(3):
            row = np.zeros(p); row[start+t:start+9:3] = 1; row[3*v+t] = -1
            equations.append(row); rhs.append(0.)
    solved = linprog(costs, A_eq=np.array(equations), b_eq=np.array(rhs), bounds=(0, None), method='highs')
    assert solved.success, solved.message
    return float(solved.fun+b[own]-b[rival])


def BA(scores, classes, truth):
    pred = classes[scores.argmax(axis=1)]
    return float(np.mean([np.mean(pred[truth == c] == c) for c in np.unique(truth)]))


def main():
    threadpool_limits(1)
    folder = ROOT/'results/centered-forest-C-sensitivity'
    summary = json.loads((folder/'complete.json').read_text())
    counts = {'inner_fits': 0, 'failures': 0, 'objectives': 0, 'LP_probes': 0, 'prediction_pairs': 0}
    for row in summary['rows']:
        name = row['dataset']; saved = json.loads((folder/(name+'.json')).read_text())
        source = json.loads((ROOT/'results/nested-crossfit-classification-rounded'/(name+'.json')).read_text())
        X, y = load_dataset(name)
        for rel, digest in saved['fingerprints'].items():
            assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == digest
        expected = [(s, inner, C) for s in source['splits'] for inner in s['inner'] for C in (.1, 1., 10.)]
        assert len(expected) == len(saved['records'])
        for record, (s, inner, C) in zip(saved['records'], expected):
            assert record['teacher_C'] == C
            assert record['outer_reference_indices'] == s['reference_indices']
            assert record['reference_indices'] == inner['reference_indices'] and record['validation_indices'] == inner['validation_indices']
            ri = np.array(s['reference_indices']); a, z = np.array(record['reference_indices']), np.array(record['validation_indices'])
            assert set(a).isdisjoint(z)
            A, B = preprocess_fit(X[ri[a]], X[ri[z]]); labels, truth = y[ri[a]], y[ri[z]]
            stage = inner['by_C'][str(C)]['stage']; classes = np.array(stage['teacher_classes'])
            assert record['full_additive'] == stage['policies']['1.0']['fit']
            assert record['tree'] == interaction_tree(A, labels)
            fit = record['forest_fit']; assert record['fallback'] == (not fit['feasible'])
            counts['failures'] += record['fallback']
            if 'coef' in fit:
                W, V, bias = np.array(fit['coef']), np.array(fit['pair_coef']), np.array(fit['intercept'])
                assert fit['centered_interactions']
                np.testing.assert_allclose(V.sum(axis=2), 0, atol=1e-7)
                np.testing.assert_allclose(V.sum(axis=3), 0, atol=1e-7)
                original = triangular_box_margins(A, labels, stage['teacher_coef'], stage['teacher_intercept'], classes)
                oracle = forest_box_margins(A, labels, W, bias, classes, fit['edges'], V, witnesses=True)
                np.testing.assert_allclose(original, fit['original_margins'], atol=1e-12)
                np.testing.assert_allclose(oracle['margins'], fit['deployed_margins'], atol=1e-12)
                own = np.argmax(labels[:, None] == classes, axis=1)
                witness_scores = forest_scores(oracle['witness_raw'], W, bias, fit['edges'], V)
                np.testing.assert_allclose(witness_scores[np.arange(len(A)), own]-witness_scores[np.arange(len(A)), oracle['rivals']], oracle['margins'], atol=1e-10)
                if fit['feasible']:
                    mask = original > 1e-10
                    assert np.all(oracle['margins'][mask] >= .5*original[mask]-1e-8) and np.all(oracle['margins'][mask] > 1e-10)
                scores = forest_scores(A, W, bias, fit['edges'], V)
                weights = np.array([len(labels)/(len(classes)*sum(labels == label)) for label in labels])
                loss = weights@(logsumexp(scores, axis=1)-scores[np.arange(len(A)), own])
                coefficients = np.column_stack([W.reshape(len(classes), -1), V.reshape(len(classes), -1)])
                penalty = coefficients[1] if len(classes) == 2 else coefficients-coefficients.mean(axis=0, keepdims=True)
                assert fit['C'] == C
                np.testing.assert_allclose(loss+np.sum(penalty**2)/(2*C), fit['objective'], rtol=1e-8, atol=1e-7)
                counts['objectives'] += 1
                for i in sorted(set([0, len(A)//2, len(A)-1])):
                    for rival in range(len(classes)):
                        if rival == own[i]:
                            continue
                        exact = lp_margin(A[i], own[i], rival, W, bias, fit['edges'], V)
                        np.testing.assert_allclose(exact, oracle['pair_margins'][i, rival], atol=1e-7, rtol=1e-8)
                        counts['LP_probes'] += 1
            rng = np.random.default_rng(record['query_seed'])
            queries = [B]+[np.clip(B+rng.uniform(-.01, .01, B.shape), 0, 1) for _ in range(3)]
            values = {'additive': [], 'forest': []}
            baseline = record['full_additive']
            for query in queries:
                additive = np.einsum('ndk,qdk->nq', triangular_memberships(query), np.array(baseline['coef']))+baseline['intercept']
                forest = forest_scores(query, fit['coef'], fit['intercept'], fit['edges'], fit['pair_coef']) if fit['feasible'] else additive
                values['additive'].append(BA(additive, classes, truth)); values['forest'].append(BA(forest, classes, truth))
                counts['prediction_pairs'] += 1
            for method in values:
                np.testing.assert_allclose(record['metrics'][method]['clean'], values[method][0], atol=1e-12)
                np.testing.assert_allclose(record['metrics'][method]['noise'], np.mean(values[method][1:]), atol=1e-12)
            counts['inner_fits'] += 1
        print(name, 'forest capacity margins and independent LP audited', flush=True)
    assert counts['inner_fits'] == 384 and counts['failures'] == summary['counts']['failures']
    counts['scope'] = 'reference_forest_exact_margin_objective_and_inner_predictions_plus_independent_LP_probes_not_outer_superiority'
    (folder/'complete-audit.json').write_text(json.dumps(counts, indent=2), encoding='utf-8')
    print(counts, flush=True)


if __name__ == '__main__':
    main()
