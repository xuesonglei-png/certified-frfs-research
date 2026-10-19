"""Independent raw-substitution ranking, geometry, refits and query replay."""
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
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.redistributed_loss import transfer_weights
from certified_frfs.capped_sparse import capped_sparse_projection


def main():
    threadpool_limits(1)
    folder = ROOT/'results/forest-aware-support'
    summary = json.loads((folder/'complete.json').read_text()); counts = Counter(); rows = []
    for prerequisite in ('forest-penalty-selection', 'forest-compact-transfer'):
        assert (ROOT/'results'/prerequisite/'complete-audit.json').exists()
    for row in summary['rows']:
        name = row['dataset']; saved = json.loads((folder/(name+'.json')).read_text())
        source = json.loads((ROOT/'results/nested-crossfit-classification-rounded'/(name+'.json')).read_text())
        selected = json.loads((ROOT/'results/forest-penalty-selection'/(name+'.json')).read_text())
        compact = json.loads((ROOT/'results/forest-compact-transfer'/(name+'.json')).read_text())
        X, y = load_dataset(name)
        for rel, digest in saved['fingerprints'].items():
            assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == digest
        expected = [(s, inner) for s in source['splits'] for inner in s['inner']]
        assert len(expected) == len(saved['records']) == len(selected['records']) == len(compact['records'])
        for record, full, old, (s, inner) in zip(saved['records'], selected['records'], compact['records'], expected):
            for key in ('outer_reference_indices', 'reference_indices', 'validation_indices', 'query_seed'):
                assert record[key] == full[key] == old[key]
            ri = np.array(s['reference_indices']); a, z = np.array(inner['reference_indices']), np.array(inner['validation_indices'])
            assert set(a).isdisjoint(z)
            A, B = preprocess_fit(X[ri[a]], X[ri[z]]); labels, truth = y[ri[a]], y[ri[z]]
            stage = inner['by_C']['1.0']['stage']; classes = np.array(stage['teacher_classes']); d = A.shape[1]
            key = full['selection']['selected']; path = record['path']
            assert record['selected_option'] == key
            assert record['target_features'] == old['features'] and (path is None) == (key == 'additive')
            if path is None:
                assert record['features'] == old['features'] and record['fit'] == old['forest_fit']
            else:
                counts['forest_paths'] += 1
                weights = np.array([1/(len(classes)*sum(labels == label)) for label in labels])
                center = weights@A; np.testing.assert_allclose(center, path['center'], atol=1e-12)
                target = scores(A, full['forest_fit']); target -= target.mean(axis=1, keepdims=True)
                original = np.ones(d)/np.sqrt(d); active = list(range(d)); trace_index = 0; attempt_index = 0
                assert path['target_size'] == len(old['features']) and path['ranking_tie_decimals'] == 12
                initial = path['trace'][0]
                assert initial['features'] == active and initial['fit'] == full['forest_fit']
                equivalent(initial['certificate'], weight_certificate(A, labels, active, original, original))
                for ranking in path['rankings']:
                    assert ranking['features'] == active
                    # Direct raw-coordinate substitution: no projection helper.
                    ranked = []
                    for removed in active:
                        fs = [a for a in active if a != removed]
                        replaced = A.copy(); omitted = [a for a in range(d) if a not in fs]
                        replaced[:, omitted] = center[omitted]
                        projected = scores(replaced, full['forest_fit']); projected -= projected.mean(axis=1, keepdims=True)
                        loss = float(weights@np.mean((projected-target)**2, axis=1))
                        ranked.append((loss, removed))
                    ranked.sort(key=lambda r: (round(r[0], 12), r[1]))
                    assert [r['removed'] for r in ranking['choices']] == [r[1] for r in ranked]
                    for reported, (loss, removed) in zip(ranking['choices'], ranked):
                        np.testing.assert_allclose(reported['loss'], loss, atol=1e-12)
                    accepted = None
                    for loss, removed in ranked:
                        attempted = path['attempts'][attempt_index]; attempt_index += 1
                        fs = [a for a in active if a != removed]
                        assert attempted['removed'] == removed and attempted['features'] == fs
                        w = capped_sparse_projection(transfer_weights(A, fs, original)/np.sqrt(d), len(fs), 2/d)*np.sqrt(d)
                        np.testing.assert_allclose(attempted['weights'], w, atol=1e-12)
                        np.testing.assert_allclose(attempted['projection_loss'], loss, atol=1e-12)
                        certificate = weight_certificate(A, labels, fs, original, w)
                        equivalent(certificate, attempted['certificate'])
                        fit = attempted['fit']; assert (fit is not None) == certificate['feasible']
                        if fit is not None:
                            edges = [e for e in full['tree']['edges'] if all(a in fs for a in e)]
                            assert fit.get('edges', edges) == edges
                            if 'coef' in fit:
                                np.testing.assert_allclose(np.array(fit['coef'])[:, [a for a in range(d) if a not in fs]], 0, atol=1e-12)
                            check_fit(A, labels, stage['teacher_coef'], stage['teacher_intercept'], classes, fit, gamma=float(key))
                        valid = certificate['feasible'] and fit['feasible']
                        assert attempted['accepted'] == valid
                        counts['attempts'] += 1; counts['failed_attempts'] += not valid
                        counts['failed_classifier_fits'] += fit is not None and not fit['feasible']
                        if valid:
                            trace_index += 1; accepted = path['trace'][trace_index]
                            assert accepted['features'] == fs and accepted['fit'] == fit
                            embedded = np.zeros(d); embedded[fs] = w
                            np.testing.assert_allclose(accepted['weights'], embedded, atol=1e-12)
                            equivalent(accepted['certificate'], certificate)
                            np.testing.assert_allclose(accepted['projection_loss'], loss, atol=1e-12)
                            active = fs
                            break
                    if accepted is None:
                        assert ranking == path['rankings'][-1]
                assert trace_index+1 == len(path['trace']) and attempt_index == len(path['attempts'])
                final = path['trace'][-1]
                assert record['features'] == active and record['fit'] == final['fit']
                assert final['certificate']['feasible'] and final['fit']['feasible']
                check_fit(A, labels, stage['teacher_coef'], stage['teacher_intercept'], classes, final['fit'], gamma=float(key), lp=True)
            assert record['budget_matched'] == (len(record['features']) == len(old['features']))
            counts['budget_unmatched'] += not record['budget_matched']; counts['references'] += 1
            np.testing.assert_allclose(record['removed'], 1-len(record['features'])/d, atol=1e-12)
            assert record['old_metrics'] == old['metrics']['selected'] and record['full_metrics'] == full['metrics']['selected']
            np.testing.assert_allclose(record['old_removed'], old['removed'], atol=1e-12)
            rng = np.random.default_rng(record['query_seed'])
            queries = [B]+[np.clip(B+rng.uniform(-.01, .01, B.shape), 0, 1) for _ in range(3)]
            fit = record['fit'] if path is not None else old['policy']['fit']
            values = [BA(scores(Q, fit), classes, truth) for Q in queries]
            np.testing.assert_allclose(record['metrics']['clean'], values[0], atol=1e-12)
            np.testing.assert_allclose(record['metrics']['noise'], np.mean(values[1:]), atol=1e-12)
        records = saved['records']
        replay = {'dataset': name, 'removed': float(np.mean([r['removed'] for r in records])),
            'old_removed': float(np.mean([r['old_removed'] for r in records])),
            **{m: {metric: float(np.mean([r[field][metric] for r in records])) for metric in ('clean', 'noise')}
                for m, field in (('new', 'metrics'), ('old', 'old_metrics'), ('full', 'full_metrics'))}}
        equivalent(replay, row); rows.append(replay)
        print(name, 'raw-substitution rankings/geometry/guards/LP/predictions audited', flush=True)
    assert counts['references'] == 128 and len(rows) == 15
    for key in summary['counts']: assert counts[key] == summary['counts'][key]
    for metric in ('clean', 'noise'):
        equivalent(paired([r['new'][metric]-r['old'][metric] for r in rows]), summary[metric+'_gain'])
    equivalent(paired([r['new']['clean']-r['full']['clean'] for r in rows]), summary['vs_full'])
    removed = float(np.mean([r['removed'] for r in rows])); old_removed = float(np.mean([r['old_removed'] for r in rows]))
    equivalent(removed, summary['removed']); equivalent(old_removed, summary['old_removed'])
    assert summary['gate_passed'] == (summary['clean_gain']['mean_pp'] >= .5 and removed >= old_removed-1e-12)
    result = {'counts': dict(counts), 'classifier_checks': dict(COUNTS), 'scope': 'development_support_generator_audit_not_SOTA_superiority'}
    (folder/'complete-audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(result, flush=True)


if __name__ == '__main__': main()
