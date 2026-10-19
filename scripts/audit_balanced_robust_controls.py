"""Scalar certification and prediction audit for both shared loss objectives."""
import hashlib
import json
import numpy as np
from collections import Counter
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.feasible_transport import weight_certificate
from audit_triangular_classifier_diagnostic import manual_margins
from run_common_fuzzy_classifier import score

FOLDER = ROOT/'results/balanced-robust-controls-pilot'
SOURCE = ROOT/'results/nested-joint-group'


def main():
    threadpool_limits(1)
    summary = json.loads((FOLDER/'complete.json').read_text())
    counts = {'splits': 0, 'models': 0, 'predictions': 0, 'requested_count_matches': 0}
    failures = Counter()
    for row in summary['rows']:
        name = row['dataset']
        saved = json.loads((FOLDER/(name+'.json')).read_text())
        original = json.loads((SOURCE/(name+'.json')).read_text())
        X, y = load_dataset(name)
        for rel, digest in saved['fingerprints'].items():
            assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == digest, rel
        for i, (split, prior) in enumerate(zip(saved['splits'], original['splits'])):
            ri, vi = np.array(split['reference_indices']), np.array(split['validation_indices'])
            assert set(ri).isdisjoint(vi)
            assert split['reference_indices'] == prior['reference_indices']
            assert split['validation_indices'] == prior['validation_indices']
            A, B = preprocess_fit(X[ri], X[vi])
            teacher = split['teacher']
            classes = np.array(teacher['classes'])
            original_margin, _ = manual_margins(A, y[ri], np.array(teacher['coef']), np.array(teacher['intercept']), classes)
            protected = original_margin > 1e-10
            for mode, outcomes in split['results'].items():
                sizes = [len(outcomes[m]['requested_features']) for m in ('proposed', 'fgs2023', 'sfrip2025')]
                assert sizes == [split['requested_size']]*3
                counts['requested_count_matches'] += 3
                for method, record in outcomes.items():
                    fitted = record['fit']
                    W, b = np.array(fitted['coef']), np.array(fitted['intercept'])
                    fs = record['actual_features']
                    assert np.all(W[:, np.setdiff1d(np.arange(X.shape[1]), fs)] == 0)
                    margins, _ = manual_margins(A, y[ri], W, b, classes)
                    assert np.all(margins[protected] >= .5*original_margin[protected]-1e-8)
                    assert np.all(margins[protected] > 1e-10)
                    solver = record['solver_fit']
                    assert record['fallback'] == (not solver['feasible'])
                    if record['fallback']:
                        failures[mode+':'+method] += 1
                        assert fs == list(range(X.shape[1]))
                        assert fitted == outcomes['full']['fit']
                    else:
                        assert solver['class_balanced'] == (mode == 'balanced')
                        assert fs == record['requested_features']
                        np.testing.assert_allclose(margins, solver['deployed_margins'], atol=1e-10)
                        # Replay the objective from deployed coefficients, including
                        # class weights, to detect an ordinary/balanced mix-up.
                        from scipy.special import logsumexp
                        from certified_frfs.triangular_classifier import triangular_memberships
                        scores = np.einsum('ndk,qdk->nq', triangular_memberships(A), W)+b
                        own = np.argmax(y[ri, None] == classes[None, :], axis=1)
                        losses = logsumexp(scores, axis=1)-scores[np.arange(len(ri)), own]
                        weights = np.array([len(ri)/(len(classes)*sum(y[ri] == label)) for label in y[ri]]) if mode == 'balanced' else np.ones(len(ri))
                        penalty_W = W[1] if len(classes) == 2 else W-W.mean(axis=0, keepdims=True)
                        objective = weights@losses+np.sum(penalty_W**2)/(2*split['teacher_C'])
                        np.testing.assert_allclose(objective, solver['objective'], rtol=1e-8, atol=1e-7)
                    if method == 'proposed' and not record['fallback']:
                        candidate = prior['stage']['policies'][prior['chosen']]['candidate']
                        assert fs == candidate['features']
                        assert weight_certificate(A, y[ri], fs, np.ones(X.shape[1])/np.sqrt(X.shape[1]), np.array(candidate['weights'])[fs])['feasible']
                    rng = np.random.default_rng(144+i)
                    values = [score(B, y[vi], fitted)]+[score(np.clip(B+rng.uniform(-.01, .01, B.shape), 0, 1), y[vi], fitted) for _ in range(3)]
                    np.testing.assert_allclose(values[0], record['clean'], atol=1e-12)
                    np.testing.assert_allclose(np.mean(values[1:]), record['noise'], atol=1e-12)
                    counts['models'] += 1
                    counts['predictions'] += 4
            counts['splits'] += 1
        print(name, 'shared objectives and scalar margins verified', flush=True)
    assert counts['splits'] == 44 and counts['models'] == 352
    counts['fallbacks'] = dict(failures)
    counts['scope'] = 'scalar_margins_objective_replay_FRS_and_predictions_not_optimizer_optimality'
    (FOLDER/'complete-audit.json').write_text(json.dumps(counts, indent=2), encoding='utf-8')
    print(counts)


if __name__ == '__main__':
    main()
