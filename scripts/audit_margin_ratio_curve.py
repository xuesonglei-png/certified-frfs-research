"""Scalar margin/certificate/prediction replay at each explicitly stated ratio."""
import hashlib
import json
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.feasible_transport import weight_certificate
from audit_triangular_classifier_diagnostic import manual_margins
from run_common_fuzzy_classifier import score


def main():
    threadpool_limits(1)
    curve = json.loads((ROOT/'results/margin-ratio-curve.json').read_text())
    reports = {}
    for key, summary in curve['ratios'].items():
        ratio = float(key)
        folder = ROOT/'results'/('margin-ratio-'+key)
        counts = {'splits': 0, 'models': 0, 'fallbacks': 0, 'predictions': 0}
        for row in summary['rows']:
            name = row['dataset']
            saved = json.loads((folder/(name+'.json')).read_text())
            assert saved['margin_ratio'] == ratio
            X, y = load_dataset(name)
            for rel, digest in saved['fingerprints'].items():
                assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == digest, rel
            for i, split in enumerate(saved['splits']):
                ri, vi = np.array(split['reference_indices']), np.array(split['validation_indices'])
                assert set(ri).isdisjoint(vi)
                A, B = preprocess_fit(X[ri], X[vi])
                teacher = split['teacher']
                classes = np.array(teacher['classes'])
                original, _ = manual_margins(A, y[ri], np.array(teacher['coef']), np.array(teacher['intercept']), classes)
                mask = original > 1e-10
                for method, result in split['results'].items():
                    fitted = result['fit']
                    W, b = np.array(fitted['coef']), np.array(fitted['intercept'])
                    actual, _ = manual_margins(A, y[ri], W, b, classes)
                    assert np.all(actual[mask] >= ratio*original[mask]-1e-8)
                    assert np.all(actual[mask] > 1e-10)
                    fs = result['actual_features']
                    assert np.all(W[:, np.setdiff1d(np.arange(X.shape[1]), fs)] == 0)
                    assert result['fallback'] == (not result['solver_fit']['feasible'])
                    if result['fallback']:
                        assert fitted == teacher and len(fs) == X.shape[1]
                        counts['fallbacks'] += 1
                    else:
                        assert result['solver_fit']['ratio'] == ratio
                        np.testing.assert_allclose(actual, result['solver_fit']['deployed_margins'], atol=1e-10)
                    if method == 'proposed' and not result['fallback']:
                        c = split['candidate']
                        assert c['features'] == fs
                        v = np.array(c['weights'])
                        assert np.isclose(v.sum(), np.sqrt(X.shape[1]))
                        assert weight_certificate(A, y[ri], fs, np.ones(X.shape[1])/np.sqrt(X.shape[1]), v[fs])['feasible']
                    rng = np.random.default_rng(144+i)
                    values = [score(B, y[vi], fitted)]+[score(np.clip(B+rng.uniform(-.01, .01, B.shape), 0, 1), y[vi], fitted) for _ in range(3)]
                    np.testing.assert_allclose(values[0], result['clean'], atol=1e-12)
                    np.testing.assert_allclose(np.mean(values[1:]), result['noise'], atol=1e-12)
                    counts['models'] += 1
                    counts['predictions'] += 4
                counts['splits'] += 1
            print(key, name, 'stated-fraction scalar audit passed', flush=True)
        assert counts['splits'] == 44
        reports[key] = counts
    (ROOT/'results/margin-ratio-curve-audit.json').write_text(json.dumps({
        'ratios': reports, 'scope': 'scalar_guarantee_FRS_fallback_prediction_not_optimizer_optimality'}, indent=2), encoding='utf-8')
    print(reports)


if __name__ == '__main__':
    main()
