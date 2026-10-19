"""Fresh fits and independent nodal interpolation/local-polytope LP audit."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.optimize import linprog
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from run_nested_joint_path import ROOT, digest
from screen_multiknot_forest import FOLDER, INPUT, METRICS
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.nested_teacher_regularized import inner_splits
from certified_frfs.fuzzy_interaction_tree import interaction_tree
from certified_frfs.multiknot_forest import fit_multiknot_forest, scores, box_margins
from certified_frfs.orthogonal_forest_logistic import fit_orthogonal_forest
from audit_forest_compact_transfer import equivalent
from review_box_iterations import paired


def independent_scores(X, fit):
    knots = np.array(fit['knots'])
    eye = np.eye(len(knots))
    hats = np.stack([np.interp(X, knots, column) for column in eye.T], axis=-1)
    W, V, b = np.array(fit['coef']), np.array(fit['pair_coef']), np.array(fit['intercept'])
    V = V.reshape(len(b), len(fit['edges']), len(knots), len(knots))
    result = np.einsum('ndk,qdk->nq', hats, W)+b
    for e, (u, v) in enumerate(fit['edges']):
        result += np.einsum('nk,nl,qkl->nq', hats[:, u], hats[:, v], V[:, e])
    return result


def independent_lp(x, own, rival, fit, eps=.01):
    knots = np.array(fit['knots'])
    m, d, edges = len(knots), len(x), fit['edges']
    lo, hi = np.maximum(0., x-eps), np.minimum(1., x+eps)
    points = np.column_stack([lo, hi, np.clip(np.tile(knots[1:-1], (d, 1)), lo[:, None], hi[:, None])])
    eye = np.eye(m)
    Z = np.stack([np.interp(points, knots, column) for column in eye.T], axis=-1)
    W, b = np.array(fit['coef']), np.array(fit['intercept'])
    V = np.array(fit['pair_coef']).reshape(len(b), len(edges), m, m)
    p = m*d+m*m*len(edges)
    costs = np.zeros(p)
    costs[:m*d] = np.einsum('dk,dsk->ds', W[own]-W[rival], Z).ravel()
    equations, rhs = [], []
    for a in range(d):
        row = np.zeros(p)
        row[m*a:m*a+m] = 1
        equations.append(row)
        rhs.append(1.)
    for e, (u, v) in enumerate(edges):
        start = m*d+m*m*e
        costs[start:start+m*m] = np.einsum('sk,tl,kl->st', Z[u], Z[v], V[own, e]-V[rival, e]).ravel()
        for s in range(m):
            row = np.zeros(p)
            row[start+m*s:start+m*s+m] = 1
            row[m*u+s] = -1
            equations.append(row)
            rhs.append(0.)
        for t in range(m):
            row = np.zeros(p)
            row[start+t:start+m*m:m] = 1
            row[m*v+t] = -1
            equations.append(row)
            rhs.append(0.)
    solved = linprog(costs, A_eq=np.array(equations), b_eq=np.array(rhs), bounds=(0, None), method='highs')
    assert solved.success, solved.message
    return float(solved.fun+b[own]-b[rival])


if __name__ == '__main__':
    threadpool_limits(1)
    complete = json.loads((FOLDER/'complete.json').read_text())
    counts, rows = Counter(), []
    maxima = {'fresh_score_error': 0., 'independent_score_error': 0., 'LP_error': 0.}
    hashes = {}
    for item in complete['rows']:
        name = item['dataset']
        artifact = FOLDER/(name+'.json')
        saved = json.loads(artifact.read_text())
        hashes[name] = digest(artifact)
        assert saved['complete'] and not saved['archived_query_labels_used']
        for rel, sha in saved['fingerprints'].items():
            assert digest(ROOT/rel) == sha, rel
        first = json.loads((INPUT/(name+'.json')).read_text())['rows'][0]
        assert first['outer'] == first['inner'] == 0
        assert saved['outer_reference_indices'] == first['outer_reference_indices']
        assert saved['reference_indices'] == first['reference_indices']
        X, y = load_dataset(name)
        ri, a = np.array(saved['outer_reference_indices']), np.array(saved['reference_indices'])
        raw, labels = X[ri[a]], y[ri[a]]
        splits = inner_splits(labels, seed=20261027)
        assert len(splits) == len(saved['folds']) == len(first['folds'])
        values = []
        for fi, ((ca, cb), fold, old) in enumerate(zip(splits, saved['folds'], first['folds'])):
            assert fold['reference_indices'] == old['reference_indices'] == ca.tolist()
            assert fold['validation_indices'] == old['validation_indices'] == cb.tolist()
            assert set(ca).isdisjoint(cb) and len(set(ca)|set(cb)) == len(raw)
            A, B = preprocess_fit(raw[ca], raw[cb])
            train, truth = labels[ca], labels[cb]
            equivalent(fold['tree'], interaction_tree(A, train))
            assert fold['control'] == {**old['single'], 'knots': [0., .5, 1.]}
            fresh3 = fit_orthogonal_forest(A, train, list(range(A.shape[1])), fold['tree']['edges'])
            fresh5 = fit_multiknot_forest(A, train, list(range(A.shape[1])), fold['tree']['edges'], knot_count=5)
            counts['fresh_training_fits'] += 2
            assert fold['ordinary_fallback'] == (not fresh5['optimizer_converged'])
            counts['new_failures'] += fold['ordinary_fallback']
            counts['ordinary_caps'] += not fresh3['optimizer_converged']
            metrics = {}
            for m, fit, fresh in [('three', fold['control'], {**fresh3, 'knots': [0., .5, 1.]}),
                                  ('five', fold['control'] if fold['ordinary_fallback'] else fold['fit'],
                                   {**fresh3, 'knots': [0., .5, 1.]} if fold['ordinary_fallback'] else fresh5)]:
                assert fit['features'] == list(range(A.shape[1])) and fit['edges'] == fold['tree']['edges']
                assert fit['C'] == fit['pair_penalty'] == 1. and fit['class_balanced'] and not fit['teacher_guards_enforced']
                if m == 'five' and not fold['ordinary_fallback']:
                    np.testing.assert_array_equal(fit['knots'], np.linspace(0, 1, 5))
                value, independent, freshvalue = scores(B, fit), independent_scores(B, fit), scores(B, fresh)
                error = float(np.max(abs(value-freshvalue)))
                maxima['fresh_score_error'] = max(maxima['fresh_score_error'], error)
                maxima['independent_score_error'] = max(maxima['independent_score_error'], float(np.max(abs(value-independent))))
                np.testing.assert_allclose(value, freshvalue, atol=1e-8)
                np.testing.assert_allclose(value, independent, atol=1e-10)
                classes = np.array(fit['classes'])
                own = np.searchsorted(classes, truth)
                pred = classes[independent.argmax(axis=1)]
                oracle = box_margins(B, truth, fit, eps=.01, witnesses=True)
                pair = oracle['pair_margins'].copy()
                pair[np.arange(len(B)), own] = 0.
                witness = independent_scores(oracle['witness_raw'], fit)
                np.testing.assert_allclose(witness[np.arange(len(B)), own]-witness[np.arange(len(B)), oracle['rivals']], oracle['margins'], atol=1e-10)
                report = fold['evaluations'][m]
                assert report['predictions'] == pred.tolist()
                np.testing.assert_allclose(report['scores'], independent, atol=1e-10)
                np.testing.assert_allclose(report['pair_margins'], pair, atol=1e-10)
                loss = logsumexp(independent, axis=1)-independent[np.arange(len(B)), own]
                def balanced(array):
                    return float(np.mean([np.mean(array[truth == c]) for c in np.unique(truth)]))
                metrics[m] = {'clean_BA': balanced(pred == truth), 'certified_BA': balanced(oracle['margins'] > 1e-10),
                              'nominal_logloss': balanced(loss)}
                equivalent(report['metrics'], metrics[m])
                counts['score_replays'] += 1
                counts['witness_replays'] += len(B)
                if fi == 0:
                    for r in range(len(classes)):
                        if r == own[0]:
                            continue
                        lp = independent_lp(B[0], own[0], r, fit)
                        err = float(abs(lp-pair[0, r]))
                        np.testing.assert_allclose(lp, pair[0, r], atol=1e-7)
                        maxima['LP_error'] = max(maxima['LP_error'], err)
                        counts['LP_probes'] += 1
            values.append(metrics)
            counts['folds'] += 1
        metrics = {m: {k: float(np.mean([v[m][k] for v in values])) for k in METRICS} for m in ('three', 'five')}
        equivalent(item['metrics'], metrics)
        rows.append({'dataset': name, 'metrics': metrics})
        counts['datasets'] += 1
        counts['positive_clean_datasets'] += metrics['five']['clean_BA'] > metrics['three']['clean_BA']+1e-12
        print(name, 'fresh fitting/scores/witnesses/LP/provenance passed', flush=True)
    for k, v in complete['counts'].items():
        assert counts[k] == v
    means = {m: {k: float(np.mean([r['metrics'][m][k] for r in rows])) for k in METRICS} for m in ('three', 'five')}
    equivalent(means, complete['means'])
    gain = 100*(means['five']['clean_BA']-means['three']['clean_BA'])
    loss_gain = means['three']['nominal_logloss']-means['five']['nominal_logloss']
    equivalent(gain, complete['clean_BA_gain_pp'])
    equivalent(loss_gain, complete['logloss_gain'])
    description = paired([r['metrics']['five']['clean_BA']-r['metrics']['three']['clean_BA'] for r in rows])
    equivalent(description, complete['paired_development_description'])
    gate = gain >= .5 and loss_gain >= 0 and counts['positive_clean_datasets'] >= 8 and counts['new_failures'] == counts['ordinary_caps'] == 0
    assert gate == complete['numeric_resource_gate'] and not complete['goal_achieved']
    result = {'passed': True, 'numeric_resource_gate': bool(gate), 'clean_BA_gain_pp': gain,
              'counts': dict(counts), 'precision_maxima': maxima, 'artifact_hashes': hashes,
              'summary_sha256': digest(FOLDER/'complete.json'), 'auditor_sha256': digest(Path(__file__)),
              'goal_achieved': False, 'scope': 'exposed_internal_model_mechanism_not_reduct_or_SOTA'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(result, flush=True)
