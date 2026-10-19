"""Independent scalar certificates and all three budget deployment checks."""
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

FOLDER = ROOT/'results/joint-group-fuzzy-pilot'


def verify_fit(A, labels, fit, classes, original):
    margins, _ = manual_margins(A, labels, np.array(fit['coef']), np.array(fit['intercept']), classes)
    protected = original > 1e-10
    assert np.all(margins[protected] >= .5*original[protected]-1e-8)
    assert np.all(margins[protected] > 1e-10)
    if 'deployed_margins' in fit:
        np.testing.assert_allclose(margins, fit['deployed_margins'], atol=1e-10)


def main():
    threadpool_limits(1)
    summary = json.loads((FOLDER/'complete.json').read_text())
    counts = {'splits': 0, 'policies': 0, 'joint_failed': 0, 'candidate_fallbacks': 0,
              'repaired_to_full': 0, 'compact_deployments': 0}
    statuses = Counter()
    for row in summary['rows']:
        name = row['dataset']
        saved = json.loads((FOLDER/(name+'.json')).read_text())
        X, y = load_dataset(name)
        for rel, digest in saved['fingerprints'].items():
            assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == digest, rel
        for split in saved['splits']:
            ri, vi = np.array(split['reference_indices']), np.array(split['validation_indices'])
            assert set(ri).isdisjoint(vi)
            A, B = preprocess_fit(X[ri], X[vi])
            teacher = split['teacher']
            classes = np.array(teacher['classes'])
            original, _ = manual_margins(A, y[ri], np.array(teacher['coef']), np.array(teacher['intercept']), classes)
            joint = split['joint']
            statuses['group:'+joint['status']] += 1
            if joint['feasible']:
                verify_fit(A, y[ri], joint, classes, original)
                W = np.array(joint['coef'])
                norms = np.linalg.norm(W-W.mean(axis=0, keepdims=True), axis=(0, 2))
                np.testing.assert_allclose(norms, joint['group_norms'], atol=1e-12)
                ranking = sorted(range(X.shape[1]), key=lambda a:(-norms[a], a))
                assert ranking == joint['ranking']
            else:
                ranking = list(range(X.shape[1]))
                counts['joint_failed'] += 1
            verify_fit(A, y[ri], split['full_fit'], classes, original)
            np.testing.assert_allclose(score(B, y[vi], split['full_fit']), split['full_clean'], atol=1e-12)
            np.testing.assert_allclose(score(B, y[vi], teacher), split['teacher_clean'], atol=1e-12)
            for key, policy in split['policies'].items():
                expected = sorted(ranking[:int(np.ceil(float(key)*X.shape[1]))])
                assert expected == policy['initial_features']
                c = policy['candidate']
                assert set(expected).issubset(c['features'])
                fs = c['features']
                v = np.array(c['weights'])
                assert np.isclose(v.sum(), np.sqrt(X.shape[1]))
                assert v.max() <= 2/np.sqrt(X.shape[1])+1e-8
                assert weight_certificate(A, y[ri], fs, np.ones(X.shape[1])/np.sqrt(X.shape[1]), v[fs])['feasible']
                fitted = policy['solver_fit']
                assert policy['fallback'] == (not (joint['feasible'] and fitted['feasible']))
                if policy['fallback']:
                    assert policy['fit'] == split['full_fit']
                    assert policy['actual_features'] == list(range(X.shape[1]))
                else:
                    assert policy['actual_features'] == fs
                actual = policy['fit']
                verify_fit(A, y[ri], actual, classes, original)
                assert np.all(np.array(actual['coef'])[:, np.setdiff1d(np.arange(X.shape[1]), policy['actual_features'])] == 0)
                np.testing.assert_allclose(score(B, y[vi], actual), policy['clean'], atol=1e-12)
                counts['policies'] += 1
                counts['candidate_fallbacks'] += policy['fallback']
                counts['repaired_to_full'] += len(fs) == X.shape[1]
                counts['compact_deployments'] += len(policy['actual_features']) < X.shape[1]
            counts['splits'] += 1
        print(name, 'all grouped budgets independently audited', flush=True)
    assert counts['splits'] == 44 and counts['policies'] == 132
    counts = {key:int(value) for key,value in counts.items()}
    counts['joint_solver_statuses'] = dict(statuses)
    counts['scope'] = 'scalar_certification_ranking_and_prediction_replay_not_conic_optimality_or_FRS_global_optimum'
    (FOLDER/'complete-audit.json').write_text(json.dumps(counts, indent=2), encoding='utf-8')
    print(counts)


if __name__ == '__main__':
    main()
