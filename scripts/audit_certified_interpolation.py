"""Replay scalar margin certificates, segment coefficients and policy choice."""
import hashlib
import json
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from audit_nested_triangular_repair import audit_stage
from run_nested_triangular import score
from run_common_fuzzy_classifier import train

FOLDER = ROOT / 'results/certified-interpolation-pilot'


def check_segment(raw, y, stage, counts):
    A, _ = preprocess_fit(raw, raw)
    seen = set()
    for record in stage['policies'].values():
        fs = record['candidate']['features']
        if tuple(fs) in seen:
            continue
        seen.add(tuple(fs))
        fit = record['fit']
        assert 0 <= fit['step'] <= 1
        target = train(A, y, fs, stage['teacher_C'])
        np.testing.assert_allclose(target['coef'], fit['target_coef'], atol=1e-12)
        np.testing.assert_allclose(target['intercept'], fit['target_intercept'], atol=1e-12)
        for key, anchor, endpoint in (('coef', 'anchor_coef', 'target_coef'),
                                      ('intercept', 'anchor_intercept', 'target_intercept')):
            expected = np.array(fit[anchor])+fit['step']*(np.array(fit[endpoint])-np.array(fit[anchor]))
            np.testing.assert_allclose(expected, fit[key], atol=1e-12)
        counts['segments'] += 1


def main():
    threadpool_limits(1)
    summary = json.loads((FOLDER/'complete.json').read_text())
    counts = {'outer_splits': 0, 'stages': 0, 'unique_models': 0, 'predictions': 0,
              'fallback_policies': 0, 'segments': 0}
    for row in summary['rows']:
        name = row['dataset']
        p = json.loads((FOLDER/(name+'.json')).read_text())
        X, y = load_dataset(name)
        for rel, digest in p['fingerprints'].items():
            assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == digest, rel
        for i, split in enumerate(p['splits']):
            ri, vi = np.array(split['reference_indices']), np.array(split['validation_indices'])
            assert set(ri).isdisjoint(vi)
            raw, labels = X[ri], y[ri]
            for inner in split['inner']:
                a, b = np.array(inner['reference_indices']), np.array(inner['validation_indices'])
                assert set(a).isdisjoint(b)
                audit_stage(raw[a], labels[a], raw[b], labels[b], inner['stage'], inner['scores'], counts)
                check_segment(raw[a], labels[a], inner['stage'], counts)
            options = [{'fraction': key,
                        'clean': float(np.mean([s['scores'][key]['clean'] for s in split['inner']])),
                        'retained': float(np.mean([s['scores'][key]['retained'] for s in split['inner']]))}
                       for key in ('0.5','0.65','0.8','1.0')]
            assert options == split['options']
            best = max(s['clean'] for s in options)
            selected = min((s for s in options if s['clean'] >= best-.005-1e-12),
                           key=lambda s:(s['retained'], -s['clean'], float(s['fraction'])))
            assert selected['fraction'] == split['chosen']
            values = score(raw, labels, X[vi], y[vi], split['stage'], noise=True, seed=144+i)
            assert values[split['chosen']] == split['metrics']
            assert values['1.0'] == split['full_metrics']
            audit_stage(raw, labels, X[vi], y[vi], split['stage'], values, counts, seed=144+i)
            check_segment(raw, labels, split['stage'], counts)
            counts['outer_splits'] += 1
        print(name, 'segments and independent margins replayed', flush=True)
    assert counts['outer_splits'] == 44
    counts['scope'] = 'cached_anchor_segment_certificates_targets_selection_predictions_not_global_optimality'
    (FOLDER/'complete-audit.json').write_text(json.dumps(counts, indent=2), encoding='utf-8')
    print(counts)


if __name__ == '__main__':
    main()
