"""Compact metric certificates and independently verified classifier guards."""
import hashlib
import json
from collections import Counter
import numpy as np
from threadpoolctl import threadpool_limits
from review_box_iterations import ROOT, paired
from audit_fuzzy_forest_capacity import BA
from audit_forest_penalty_selection import check_fit, scores, COUNTS
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.feasible_transport import weight_certificate


def equivalent(actual, expected):
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual: equivalent(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected): equivalent(a, b)
    elif isinstance(actual, (float, int, bool)):
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)
    else:
        assert actual == expected


def main(partial=False):
    threadpool_limits(1)
    folder = ROOT/'results/forest-compact-transfer'
    if partial:
        provisional = []
        for path in sorted(p for p in folder.glob('*.json') if not p.name.startswith('complete')):
            saved = json.loads(path.read_text()); records = saved['records']
            provisional.append({'dataset': saved['dataset'], 'removed': float(np.mean([r['removed'] for r in records])),
                **{m: {metric: float(np.mean([r['metrics'][m][metric] for r in records])) for metric in ('clean', 'noise')}
                    for m in ('additive', 'selected')},
                'full': {m: {metric: float(np.mean([r['full_metrics'][m][metric] for r in records])) for metric in ('clean', 'noise')}
                    for m in ('additive', 'selected')}})
        summary = {'rows': provisional}
    else:
        summary = json.loads((folder/'complete.json').read_text())
        assert (ROOT/'results/forest-penalty-selection/complete-audit.json').exists()
    counts = Counter(); rows = []
    for row in summary['rows']:
        name = row['dataset']; saved = json.loads((folder/(name+'.json')).read_text())
        source = json.loads((ROOT/'results/nested-crossfit-classification-rounded'/(name+'.json')).read_text())
        selected = json.loads((ROOT/'results/forest-penalty-selection'/(name+'.json')).read_text())
        X, y = load_dataset(name)
        for rel, digest in saved['fingerprints'].items():
            assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == digest
        expected = [(s, inner) for s in source['splits'] for inner in s['inner']]
        assert len(expected) == len(saved['records']) == len(selected['records'])
        for record, reference, (s, inner) in zip(saved['records'], selected['records'], expected):
            for key in ('outer_reference_indices', 'reference_indices', 'validation_indices', 'query_seed'):
                assert record[key] == reference[key]
            ri = np.array(s['reference_indices']); a, z = np.array(inner['reference_indices']), np.array(inner['validation_indices'])
            assert set(a).isdisjoint(z)
            A, B = preprocess_fit(X[ri[a]], X[ri[z]]); labels, truth = y[ri[a]], y[ri[z]]
            stage = inner['by_C']['1.0']['stage']; classes = np.array(stage['teacher_classes'])
            policy = stage['policies']['0.8']; fs = policy['candidate']['features']
            assert record['policy'] == policy and record['features'] == fs
            np.testing.assert_allclose(record['removed'], 1-len(fs)/A.shape[1], atol=1e-12)
            weights = np.array(policy['candidate']['weights'])
            cert = weight_certificate(A, labels, fs, np.ones(A.shape[1])/np.sqrt(A.shape[1]), weights[fs])
            equivalent(cert, policy['candidate']['certificate']); assert cert['feasible']
            counts['metric_certificates'] += 1
            key = reference['selection']['selected']; assert record['selected_option'] == key
            fit = record['forest_fit']; assert (fit is None) == (key == 'additive')
            assert record['fallback'] == (fit is not None and not fit['feasible'])
            counts['forest_failures'] += record['fallback']; counts['forest_attempts'] += fit is not None
            check_fit(A, labels, stage['teacher_coef'], stage['teacher_intercept'], classes, policy['fit'])
            if fit is not None:
                expected_edges = [e for e in reference['tree']['edges'] if all(a in fs for a in e)]
                assert fit.get('edges', expected_edges) == expected_edges
                if 'coef' in fit:
                    omitted = [a for a in range(A.shape[1]) if a not in fs]
                    np.testing.assert_allclose(np.array(fit['coef'])[:, omitted], 0, atol=1e-12)
                check_fit(A, labels, stage['teacher_coef'], stage['teacher_intercept'], classes,
                    fit, gamma=float(key), lp=True)
            rng = np.random.default_rng(record['query_seed'])
            queries = [B]+[np.clip(B+rng.uniform(-.01, .01, B.shape), 0, 1) for _ in range(3)]
            values = {'additive': [], 'selected': []}
            for query in queries:
                ordinary = scores(query, policy['fit'])
                S = scores(query, fit) if fit is not None and fit['feasible'] else ordinary
                values['additive'].append(BA(ordinary, classes, truth)); values['selected'].append(BA(S, classes, truth))
                counts['prediction_pairs'] += 1
            for method in values:
                np.testing.assert_allclose(record['metrics'][method]['clean'], values[method][0], atol=1e-12)
                np.testing.assert_allclose(record['metrics'][method]['noise'], np.mean(values[method][1:]), atol=1e-12)
            assert record['full_metrics'] == reference['metrics']; counts['references'] += 1
        records = saved['records']
        replay = {'dataset': name, 'removed': float(np.mean([r['removed'] for r in records])),
            **{m: {metric: float(np.mean([r['metrics'][m][metric] for r in records])) for metric in ('clean', 'noise')}
                for m in ('additive', 'selected')},
            'full': {m: {metric: float(np.mean([r['full_metrics'][m][metric] for r in records])) for metric in ('clean', 'noise')}
                for m in ('additive', 'selected')}}
        equivalent(replay, row); rows.append(replay)
        print(name, 'compact metric/guard/objective/LP/prediction audit passed', flush=True)
    if partial:
        print({'partial_compact_audit': True, 'selection_audit_still_required': True, 'counts': dict(counts), 'classifier_checks': dict(COUNTS)}, flush=True)
        return
    assert counts['references'] == 128 and len(rows) == 15
    for key in summary['counts']: assert counts[key] == summary['counts'][key]
    for metric in ('clean', 'noise'):
        equivalent(paired([r['selected'][metric]-r['additive'][metric] for r in rows]), summary[metric+'_gain'])
    equivalent(paired([r['selected']['clean']-r['full']['selected']['clean'] for r in rows]), summary['vs_selected_full'])
    removed = float(np.mean([r['removed'] for r in rows])); equivalent(removed, summary['removed'])
    assert summary['gate_passed'] == (summary['clean_gain']['mean_pp'] >= .5 and removed >= .1 and counts['forest_failures']/max(1, counts['forest_attempts']) <= .1)
    result = {'counts': dict(counts), 'classifier_checks': dict(COUNTS),
        'scope': 'fixed_compact_development_diagnostic_audit_not_baseline_superiority'}
    (folder/'complete-audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(result, flush=True)


if __name__ == '__main__':
    import sys
    main(partial='--partial' in sys.argv)
