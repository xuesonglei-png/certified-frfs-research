"""Audit disabled guards explicitly, without importing a false certificate."""
import hashlib
import json
from collections import Counter
import numpy as np
from threadpoolctl import threadpool_limits
from review_box_iterations import ROOT, paired
from audit_forest_compact_transfer import equivalent
from audit_forest_penalty_selection import check_fit, scores, COUNTS
from audit_fuzzy_forest_capacity import BA
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.triangular_classifier import triangular_box_margins
from certified_frfs.forest_fuzzy_classifier import forest_box_margins
from certified_frfs.feasible_transport import weight_certificate


def main():
    threadpool_limits(1)
    folder = ROOT/'results/forest-teacher-guard-cost'
    summary = json.loads((folder/'complete.json').read_text()); counts = Counter(); rows = []
    assert (ROOT/'results/forest-aware-support/complete-audit.json').exists()
    for row in summary['rows']:
        name = row['dataset']; saved = json.loads((folder/(name+'.json')).read_text())
        source = json.loads((ROOT/'results/nested-crossfit-classification-rounded'/(name+'.json')).read_text())
        selected = json.loads((ROOT/'results/forest-penalty-selection'/(name+'.json')).read_text())
        compact = json.loads((ROOT/'results/forest-compact-transfer'/(name+'.json')).read_text())
        aware = json.loads((ROOT/'results/forest-aware-support'/(name+'.json')).read_text())
        X, y = load_dataset(name)
        for rel, digest in saved['fingerprints'].items(): assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == digest
        expected = [(s, inner) for s in source['splits'] for inner in s['inner']]
        assert len(expected) == len(saved['records']) == len(aware['records'])
        for r, reference, full, old, (s, inner) in zip(saved['records'], aware['records'], selected['records'], compact['records'], expected):
            for key in ('outer_reference_indices', 'reference_indices', 'validation_indices', 'query_seed', 'features', 'selected_option'):
                assert r[key] == reference[key]
            ri = np.array(s['reference_indices']); a, z = np.array(inner['reference_indices']), np.array(inner['validation_indices'])
            assert set(a).isdisjoint(z)
            A, B = preprocess_fit(X[ri[a]], X[ri[z]]); labels, truth = y[ri[a]], y[ri[z]]
            stage = inner['by_C']['1.0']['stage']; classes = np.array(stage['teacher_classes']); key = r['selected_option']
            assert not r['teacher_guards_enforced']
            fit = r['fit']; assert r['fallback'] == (not fit['feasible'])
            edges = [e for e in full['tree']['edges'] if all(a in r['features'] for a in e)] if key != 'additive' else []
            if 'coef' in fit:
                assert fit['teacher_guards_enforced'] is False and fit['edges'] == edges
                np.testing.assert_allclose(np.array(fit['coef'])[:, [a for a in range(A.shape[1]) if a not in r['features']]], 0, atol=1e-12)
            check_fit(A, labels, stage['teacher_coef'], stage['teacher_intercept'], classes,
                      fit, gamma=float(key) if key != 'additive' else 1., lp=True)
            guarded = reference['fit'] if key != 'additive' else old['policy']['fit']
            deployed = fit if fit['feasible'] else guarded
            V = np.asarray(deployed.get('pair_coef', np.zeros((len(classes), 0, 3, 3))))
            V = V.reshape(len(classes), len(deployed.get('edges', [])), 3, 3)
            original = triangular_box_margins(A, labels, stage['teacher_coef'], stage['teacher_intercept'], classes)
            margin = forest_box_margins(A, labels, deployed['coef'], deployed['intercept'], classes, deployed.get('edges', []), V)
            np.testing.assert_allclose(original, r['original_margins'], atol=1e-12)
            np.testing.assert_allclose(margin, r['deployed_margins'], atol=1e-12)
            protected = original > 1e-10; retained = protected & (margin >= .5*original-1e-8) & (margin > 1e-10)
            if 'coef' in fit:
                assert fit['teacher_protected_count'] == int(protected.sum())
            per_class = [{'class': int(c), 'protected': int(np.sum(protected & (labels == c))),
                          'retained': int(np.sum(retained & (labels == c)))} for c in classes]
            assert per_class == r['per_class_retention']
            assert r['actual_full_teacher_guards_verified'] == bool(np.all(retained[protected]))
            counts['full_teacher_guard_violations'] += not r['actual_full_teacher_guards_verified']
            counts['references_below_90percent_any_protected_class'] += any(c['protected'] and c['retained'] < int(np.ceil(.9*c['protected']-1e-12)) for c in per_class)
            counts['numerical_fit_failures'] += r['fallback']; counts['references'] += 1
            candidate = reference['path']['trace'][-1] if reference['path'] is not None else old['policy']['candidate']
            weights = np.array(candidate['weights']); fs = r['features']
            certificate = weight_certificate(A, labels, fs, np.ones(A.shape[1])/np.sqrt(A.shape[1]), weights[fs])
            assert certificate['feasible']; equivalent(certificate, candidate['certificate'])
            counts['geometry_certificates'] += 1
            rng = np.random.default_rng(r['query_seed'])
            queries = [B]+[np.clip(B+rng.uniform(-.01, .01, B.shape), 0, 1) for _ in range(3)]
            values = [BA(scores(Q, deployed), classes, truth) for Q in queries]
            np.testing.assert_allclose(r['metrics']['clean'], values[0], atol=1e-12)
            np.testing.assert_allclose(r['metrics']['noise'], np.mean(values[1:]), atol=1e-12)
            assert r['guarded_metrics'] == reference['metrics']; np.testing.assert_allclose(r['removed'], reference['removed'], atol=1e-12)
        records = saved['records']
        replay = {'dataset': name, 'removed': float(np.mean([r['removed'] for r in records])),
            **{m: {metric: float(np.mean([r[field][metric] for r in records])) for metric in ('clean', 'noise')}
                for m, field in (('unguarded', 'metrics'), ('guarded', 'guarded_metrics'))}}
        equivalent(replay, row); rows.append(replay)
        print(name, 'explicit ablation/objectives/LP/retained guards/geometry/predictions audited', flush=True)
    assert counts['references'] == 128 and len(rows) == 15
    for key in summary['counts']: assert counts[key] == summary['counts'][key]
    for metric in ('clean', 'noise'):
        equivalent(paired([r['unguarded'][metric]-r['guarded'][metric] for r in rows]), summary[metric+'_gain'])
    assert summary['gate_passed'] == (summary['clean_gain']['mean_pp'] >= .5 and counts['numerical_fit_failures']/128 <= .1)
    result = {'counts': dict(counts), 'classifier_checks': dict(COUNTS),
        'scope': 'counterfactual_guard_cost_audit_not_certified_classification_or_SOTA_superiority'}
    (folder/'complete-audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(result, flush=True)


if __name__ == '__main__': main()
