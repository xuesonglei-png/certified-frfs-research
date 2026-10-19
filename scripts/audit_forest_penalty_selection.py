"""Independent replay of isolation, selection, objectives, guards and LP margins."""
import hashlib
import json
from collections import Counter
import numpy as np
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from review_box_iterations import ROOT, paired
from audit_fuzzy_forest_capacity import lp_margin, BA
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.nested_teacher_regularized import inner_splits
from certified_frfs.linear_retention import linear_parameters
from certified_frfs.triangular_classifier import triangular_memberships, triangular_box_margins
from certified_frfs.fuzzy_interaction_tree import interaction_tree
from certified_frfs.forest_fuzzy_classifier import forest_scores, forest_box_margins

OPTIONS = ('additive', '10.0', '3.0', '1.0')
COUNTS = Counter()


def scores(X, fit):
    if 'pair_coef' in fit:
        return forest_scores(X, fit['coef'], fit['intercept'], fit['edges'],
            np.asarray(fit['pair_coef']).reshape(len(fit['coef']), len(fit['edges']), 3, 3))
    return np.einsum('ndk,qdk->nq', triangular_memberships(X), np.array(fit['coef']))+fit['intercept']


def check_fit(A, labels, Wt, bt, classes, fit, *, gamma=None, lp=False):
    COUNTS['checked_fits'] += 1
    if 'coef' not in fit:
        assert not fit['feasible']; return
    W, bias = np.array(fit['coef']), np.array(fit['intercept'])
    original = triangular_box_margins(A, labels, Wt, bt, classes)
    if gamma is None:
        margin = triangular_box_margins(A, labels, W, bias, classes)
    else:
        assert fit['centered_interactions'] and fit['pair_penalty'] == gamma and fit['C'] == 1.
        V = np.array(fit['pair_coef']).reshape(len(classes), len(fit['edges']), 3, 3)
        if fit['feasible']:
            np.testing.assert_allclose(V.sum(axis=2), 0, atol=1e-7)
            np.testing.assert_allclose(V.sum(axis=3), 0, atol=1e-7)
        oracle = forest_box_margins(A, labels, W, bias, classes, fit['edges'], V, witnesses=True)
        margin = oracle['margins']
        own = np.argmax(labels[:, None] == classes, axis=1)
        witness_scores = scores(oracle['witness_raw'], fit)
        np.testing.assert_allclose(witness_scores[np.arange(len(A)), own]-witness_scores[np.arange(len(A)), oracle['rivals']], margin, atol=1e-10)
        if lp:
            for i in sorted(set([0, len(A)//2, len(A)-1])):
                for rival in range(len(classes)):
                    if rival == own[i]: continue
                    independent = lp_margin(A[i], own[i], rival, W, bias, fit['edges'], V)
                    np.testing.assert_allclose(independent, oracle['pair_margins'][i, rival], atol=1e-7, rtol=1e-8)
                    COUNTS['LP_probes'] += 1
    if 'original_margins' in fit:
        np.testing.assert_allclose(original, fit['original_margins'], atol=1e-12)
        np.testing.assert_allclose(margin, fit['deployed_margins'], atol=1e-12)
    if fit['feasible']:
        mask = original > 1e-10
        guards_verified = bool(np.all(margin[mask] >= .5*original[mask]-1e-8) and np.all(margin[mask] > 1e-10))
        if fit.get('teacher_guards_enforced', True):
            assert guards_verified
            COUNTS['verified_guards'] += 1
        else:
            assert fit['full_teacher_guards_verified'] == guards_verified
            COUNTS['explicit_guard_ablation_fits'] += 1
    if 'objective' not in fit: return
    S = scores(A, fit); q = len(classes); own = np.argmax(labels[:, None] == classes, axis=1)
    weights = np.array([len(labels)/(q*sum(labels == label)) for label in labels])
    loss = weights@(logsumexp(S, axis=1)-S[np.arange(len(A)), own])
    main = W[1] if q == 2 else W-W.mean(axis=0, keepdims=True)
    penalty = np.sum(main**2)
    if gamma is not None:
        pair = V[1] if q == 2 else V-V.mean(axis=0, keepdims=True)
        penalty += gamma*np.sum(pair**2)
    np.testing.assert_allclose(loss+penalty/2, fit['objective'], rtol=1e-8, atol=1e-7)
    COUNTS['objectives'] += 1


def main(partial=False):
    threadpool_limits(1)
    folder = ROOT/'results/forest-penalty-selection'
    if partial:
        available = sorted(p for p in folder.glob('*.json') if not p.name.startswith('complete'))
        provisional = []
        for path in available:
            saved = json.loads(path.read_text()); records = saved['records']
            provisional.append({'dataset': saved['dataset'], **{m: {metric: float(np.mean([r['metrics'][m][metric] for r in records]))
                for metric in ('clean', 'noise')} for m in ('additive', 'selected')}})
        summary = {'rows': provisional}
    else:
        summary = json.loads((folder/'complete.json').read_text())
    selections = Counter(); rows = []
    for row in summary['rows']:
        name = row['dataset']; saved = json.loads((folder/(name+'.json')).read_text())
        source = json.loads((ROOT/'results/nested-crossfit-classification-rounded'/(name+'.json')).read_text())
        X, y = load_dataset(name)
        for rel, digest in saved['fingerprints'].items():
            assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == digest
        expected = [(s, inner) for s in source['splits'] for inner in s['inner']]
        assert len(expected) == len(saved['records'])
        metrics = {m: {metric: [] for metric in ('clean', 'noise')} for m in ('additive', 'selected')}
        for record, (s, inner) in zip(saved['records'], expected):
            assert record['outer_reference_indices'] == s['reference_indices']
            assert record['reference_indices'] == inner['reference_indices'] and record['validation_indices'] == inner['validation_indices']
            ri = np.array(s['reference_indices']); a, z = np.array(record['reference_indices']), np.array(record['validation_indices'])
            assert set(a).isdisjoint(z)
            raw, labels = X[ri[a]], y[ri[a]]; selection = record['selection']
            assert selection['seed'] == 20261012
            partitions = inner_splits(labels, seed=20261012)
            assert len(partitions) == len(selection['folds'])
            evidence = {k: [] for k in OPTIONS}
            for fold, (t, v) in zip(selection['folds'], partitions):
                assert fold['reference_indices'] == t.tolist() and fold['validation_indices'] == v.tolist()
                assert set(t).isdisjoint(v) and set(t)|set(v) == set(range(len(raw)))
                A, B = preprocess_fit(raw[t], raw[v]); yt, truth = labels[t], labels[v]
                teacher = LogisticRegression(C=1., max_iter=2000, random_state=0).fit(triangular_memberships(A).reshape(len(A), -1), yt)
                W, bias, classes = linear_parameters(teacher); W = W.reshape(len(classes), A.shape[1], 3)
                np.testing.assert_allclose(W, fold['teacher_coef'], atol=1e-12)
                np.testing.assert_allclose(bias, fold['teacher_intercept'], atol=1e-12)
                np.testing.assert_array_equal(classes, fold['teacher_classes'])
                tree = interaction_tree(A, yt); assert tree == fold['tree']
                additive = fold['additive_fit']; check_fit(A, yt, W, bias, classes, additive)
                if fold['additive_failure'] is not None:
                    assert not fold['additive_failure']['feasible']
                    np.testing.assert_array_equal(additive['coef'], W)
                    np.testing.assert_array_equal(additive['intercept'], bias)
                    check_fit(A, yt, W, bias, classes, fold['additive_failure'])
                    COUNTS['selection_additive_failures'] += 1
                ordinary = scores(B, additive)
                evidence['additive'].append(BA(ordinary, classes, truth))
                for key in OPTIONS[1:]:
                    fit = fold['forest_fits'][key]; assert fit.get('edges', tree['edges']) == tree['edges']
                    check_fit(A, yt, W, bias, classes, fit, gamma=float(key))
                    S = scores(B, fit) if fit['feasible'] else ordinary
                    evidence[key].append(BA(S, classes, truth))
                    COUNTS['selection_forest_failures'] += not fit['feasible']
                    COUNTS['selection_forest_fits'] += 1
                for key in OPTIONS:
                    np.testing.assert_allclose(fold['scores'][key], evidence[key][-1], atol=1e-12)
                COUNTS['selection_folds'] += 1
            means = {k: float(np.mean(evidence[k])) for k in OPTIONS}
            selected = sorted(OPTIONS, key=lambda k: (-round(means[k], 12), OPTIONS.index(k)))[0]
            assert selected == selection['selected']
            for key in OPTIONS:
                np.testing.assert_allclose(means[key], selection['mean_scores'][key], atol=1e-12)
            selections[selected] += 1
            A, B = preprocess_fit(raw, X[ri[z]]); truth = y[ri[z]]
            stage = inner['by_C']['1.0']['stage']; classes = np.array(stage['teacher_classes'])
            assert record['full_additive'] == stage['policies']['1.0']['fit']
            assert record['tree'] == interaction_tree(A, labels)
            teacher = LogisticRegression(C=1., max_iter=2000, random_state=0).fit(triangular_memberships(A).reshape(len(A), -1), labels)
            W, bias, tc = linear_parameters(teacher); W = W.reshape(len(tc), A.shape[1], 3)
            np.testing.assert_allclose(W, stage['teacher_coef'], atol=1e-12)
            np.testing.assert_allclose(bias, stage['teacher_intercept'], atol=1e-12)
            np.testing.assert_array_equal(tc, classes)
            fit = record['forest_fit']; assert (fit is None) == (selected == 'additive')
            assert record['fallback'] == (fit is not None and not fit['feasible'])
            COUNTS['final_forest_failures'] += record['fallback']
            if fit is not None:
                assert fit.get('edges', record['tree']['edges']) == record['tree']['edges']
                check_fit(A, labels, W, bias, classes, fit, gamma=float(selected), lp=True)
                COUNTS['final_forest_attempts'] += 1
            rng = np.random.default_rng(record['query_seed'])
            queries = [B]+[np.clip(B+rng.uniform(-.01, .01, B.shape), 0, 1) for _ in range(3)]
            values = {'additive': [], 'selected': []}
            for query in queries:
                ordinary = scores(query, record['full_additive'])
                S = scores(query, fit) if fit is not None and fit['feasible'] else ordinary
                values['additive'].append(BA(ordinary, classes, truth)); values['selected'].append(BA(S, classes, truth))
                COUNTS['prediction_pairs'] += 1
            for method in values:
                for metric, value in [('clean', values[method][0]), ('noise', np.mean(values[method][1:]))]:
                    np.testing.assert_allclose(record['metrics'][method][metric], value, atol=1e-12)
                    metrics[method][metric].append(value)
            COUNTS['references'] += 1
        replay_row = {'dataset': name, **{m: {metric: float(np.mean(v)) for metric, v in fields.items()} for m, fields in metrics.items()}}
        for m in metrics:
            for metric in metrics[m]:
                np.testing.assert_allclose(row[m][metric], replay_row[m][metric], atol=1e-12)
        rows.append(replay_row)
        print(name, 'selection/guards/objectives/LP/predictions audited', flush=True)
    if partial:
        print({'partial_audit': True, 'datasets': len(rows), **dict(COUNTS)}, flush=True)
        return
    assert COUNTS['references'] == 128 and len(rows) == 15
    assert dict(selections) == summary['selection_counts']
    for key in ('final_forest_failures', 'final_forest_attempts', 'selection_folds', 'selection_forest_failures',
                'selection_forest_fits', 'selection_additive_failures'):
        assert COUNTS[key] == summary['counts'][key]
    for metric in ('clean', 'noise'):
        assert paired([r['selected'][metric]-r['additive'][metric] for r in rows]) == summary[metric+'_gain']
    clean = summary['clean_gain']['mean_pp']
    assert summary['gate_passed'] == (clean >= .5 and COUNTS['final_forest_failures']/max(1, COUNTS['final_forest_attempts']) <= .1)
    result = dict(COUNTS); result['scope'] = 'development_capacity_selection_isolation_and_certification_audit_not_superiority'
    (folder/'complete-audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(result, flush=True)


if __name__ == '__main__':
    import sys
    main(partial='--partial' in sys.argv)
