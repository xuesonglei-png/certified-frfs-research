"""Refit inner baselines and recompute selected metrics/certificates."""
import hashlib
import json
import numpy as np
from threadpoolctl import threadpool_limits
from compare_quantile_granulation import ROOT, FOLDER, GRIDS, SELECTORS, score, utility
from certified_frfs.datasets import load_dataset
from certified_frfs.query_probability import QueryProbabilityObjective
from certified_frfs.validated_pool import preprocess_fit, validation_scores
from certified_frfs.feasible_transport import weight_certificate


def close_tree(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            close_tree(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, z in zip(a, b):
            close_tree(x, z)
    elif isinstance(a, float):
        np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-12)
    else:
        assert a == b


def main():
    threadpool_limits(1)
    complete = json.loads((FOLDER/'complete.json').read_text())
    counts = {'datasets': 0, 'splits': 0, 'inner_selector_refits': 0, 'external_metric_recomputations': 0}
    for row in complete['rows']:
        name = row['dataset']; p = json.loads((FOLDER/f'{name}.json').read_text())
        for relative, digest in p['fingerprints'].items():
            assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() == digest, relative
        X, y = load_dataset(name)
        prior = json.loads((ROOT/'results/long-capped-sparse'/f'{name}.json').read_text())
        for index, s in enumerate(p['splits']):
            old = prior['splits'][index]
            assert s['reference_indices'] == old['reference_indices'] and s['validation_indices'] == old['validation_indices']
            ri, vi = np.array(s['reference_indices']), np.array(s['validation_indices'])
            assert set(ri).isdisjoint(vi)
            raw, labels = X[ri], y[ri]; A, B = preprocess_fit(raw, X[vi], geometry='quantile')
            d = X.shape[1]; mass = np.sqrt(d)
            obj = QueryProbabilityObjective(raw, labels, geometry='quantile')
            assert obj.splits == s['trajectory']['training_splits'] == s['fit']['training_splits']
            c = s['fit']['compact_selected']; w = np.array(c['weights']); fs = c['features']
            assert fs == np.flatnonzero(w > 0).tolist()
            assert np.isfinite(w).all() and np.all(w >= 0)
            assert abs(w.sum()-mass) < 1e-8 and w.max() <= 2*mass/d+1e-8
            certificate = weight_certificate(A, labels, fs, np.ones(d)/mass, w[fs])
            close_tree(certificate, c['certificate']); assert certificate['feasible']
            np.testing.assert_allclose(obj(w/mass)[0], c['objective'], atol=1e-12)
            feasible = [c for c in s['fit']['candidates'] if c['certificate']['feasible']]
            best = min(feasible, key=lambda c: (c['objective'], len(c['features'])))
            chosen = min((c for c in feasible if c['objective'] <= best['objective']+.01), key=lambda c: (len(c['features']), c['objective']))
            assert chosen == s['fit']['compact_selected']
            controls = {'proposed': s['fit']['compact_selected'], 'dense': s['certified_trajectory']['selected'],
                        'unconstrained': s['fit']['unconstrained_selected'],
                        'full': {'features': list(range(d)), 'weights': (np.ones(d)/mass).tolist()}}
            for method, c in controls.items():
                metrics = validation_scores(A, labels, B, y[vi], c['features'], np.array(c['weights']), 144+index, joint=True)
                close_tree(metrics, s['metrics'][method]); counts['external_metric_recomputations'] += 1
            for method, grid in GRIDS.items():
                base = s['baselines'][method]; trace = base['trace']
                assert [c['config'] for c in trace] == grid
                for j, inner in enumerate(obj.splits):
                    ti, qi = np.array(inner['reference']), np.array(inner['query'])
                    assert set(ti).isdisjoint(qi) and len(set(ti).union(qi)) == len(raw)
                    Ai, Bi = preprocess_fit(raw[ti], raw[qi], geometry='quantile')
                    for candidate in trace:
                        selected = SELECTORS[method](Ai, labels[ti], **candidate['config'])
                        assert selected['features'] == candidate['folds'][j]['features']
                        values = score(Ai, labels[ti], Bi, labels[qi], selected['features'], 5000+j)
                        close_tree(values, candidate['folds'][j]['metrics'])
                        counts['inner_selector_refits'] += 1
                for candidate in trace:
                    np.testing.assert_allclose(candidate['utility'], np.mean([utility(f['metrics']) for f in candidate['folds']]), atol=1e-12)
                chosen = max(range(len(trace)), key=lambda j: (trace[j]['utility'], -trace[j]['mean_size'], -j))
                assert chosen == base['chosen_index']
                selected = SELECTORS[method](A, labels, **trace[chosen]['config'])
                close_tree(selected, base['selection'])
                metrics = score(A, labels, B, y[vi], selected['features'], 144+index)
                close_tree(metrics, s['metrics'][method]); counts['external_metric_recomputations'] += 1
            counts['splits'] += 1
        counts['datasets'] += 1
        print(name, 'nested baseline refits, selected certificate and scores verified', flush=True)
    assert counts['datasets'] == 15 and counts['splits'] == 44
    counts['scope'] = 'selected_certificate_and_nested_baseline_replay_not_full_proposed_optimizer_replay_or_interval_rounding_proof'
    (FOLDER/'complete-audit.json').write_text(json.dumps(counts, indent=2), encoding='utf-8')
    print(json.dumps(counts, indent=2), flush=True)


if __name__ == '__main__':
    main()
