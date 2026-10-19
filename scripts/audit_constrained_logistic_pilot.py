"""Independent raw-breakpoint margins, metric certificates and predictions."""
import hashlib
import json
from collections import Counter
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.feasible_transport import weight_certificate
from audit_triangular_classifier_diagnostic import manual_margins
from run_common_fuzzy_classifier import score

FOLDER = ROOT/'results/constrained-logistic-pilot'


def main():
    threadpool_limits(1)
    summary = json.loads((FOLDER/'complete.json').read_text())
    counts = {'splits': 0, 'verified_models': 0, 'fallbacks': 0, 'predictions': 0}
    statuses = Counter()
    for row in summary['rows']:
        name = row['dataset']
        saved = json.loads((FOLDER/(name+'.json')).read_text())
        X, y = load_dataset(name)
        for rel, digest in saved['fingerprints'].items():
            assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == digest, rel
        for i, split in enumerate(saved['splits']):
            ri, vi = np.array(split['reference_indices']), np.array(split['validation_indices'])
            assert set(ri).isdisjoint(vi)
            A, B = preprocess_fit(X[ri], X[vi])
            teacher = split['teacher']
            classes = np.array(teacher['classes'])
            original, _ = manual_margins(A, y[ri], np.array(teacher['coef']),
                                          np.array(teacher['intercept']), classes)
            protected = original > 1e-10
            for method, result in split['results'].items():
                fit = result['fit']
                coef, intercept = np.array(fit['coef']), np.array(fit['intercept'])
                fs = result['actual_features']
                assert np.all(coef[:, np.setdiff1d(np.arange(A.shape[1]), fs)] == 0)
                margins, _ = manual_margins(A, y[ri], coef, intercept, classes)
                assert np.all(margins[protected] >= .5*original[protected]-1e-8)
                assert np.all(margins[protected] > 1e-10)
                solver = result['solver_fit']
                statuses[solver['status']] += 1
                assert result['fallback'] == (not solver['feasible'])
                if result['fallback']:
                    assert len(fs) == A.shape[1] and fit == teacher
                    counts['fallbacks'] += 1
                else:
                    np.testing.assert_allclose(margins, solver['deployed_margins'], atol=1e-10)
                if method == 'proposed' and not result['fallback']:
                    c = split['candidate']
                    assert c['features'] == fs
                    assert weight_certificate(A, y[ri], fs, np.ones(A.shape[1])/np.sqrt(A.shape[1]),
                                              np.array(c['weights'])[fs])['feasible']
                rng = np.random.default_rng(144+i)
                values = [score(B, y[vi], fit)]+[score(np.clip(B+rng.uniform(-.01, .01, B.shape), 0, 1), y[vi], fit) for _ in range(3)]
                np.testing.assert_allclose(values[0], result['clean'], atol=1e-12)
                np.testing.assert_allclose(np.mean(values[1:]), result['noise'], atol=1e-12)
                counts['verified_models'] += 1
                counts['predictions'] += 4
            counts['splits'] += 1
        print(name, 'scalar certificate and prediction audit passed', flush=True)
    assert counts['splits'] == 44
    counts['solver_statuses'] = dict(statuses)
    counts['scope'] = 'scalar_certification_and_prediction_replay_not_conic_optimality_verification'
    (FOLDER/'complete-audit.json').write_text(json.dumps(counts, indent=2), encoding='utf-8')
    print(counts)


if __name__ == '__main__':
    main()
