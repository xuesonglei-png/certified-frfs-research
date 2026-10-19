"""Independent scalar replay of the weaker reference-only anchor guarantee."""
import hashlib
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.triangular_classifier import triangular_memberships
from certified_frfs.linear_retention import linear_parameters
from certified_frfs.feasible_transport import weight_certificate
from audit_triangular_classifier_diagnostic import manual_margins
from run_common_fuzzy_classifier import train, score

FOLDER = ROOT/'results/crossfit-anchor-pilot'


def main():
    threadpool_limits(1)
    summary = json.loads((FOLDER/'complete.json').read_text())
    counts = {'splits': 0, 'crossfit_teachers': 0, 'excluded_positive': 0,
              'protected': 0, 'empty_masks': 0, 'compact_deployments': 0,
              'newly_accepted_vs_all_positive': 0}
    for row in summary['rows']:
        name = row['dataset']
        saved = json.loads((FOLDER/(name+'.json')).read_text())
        X, y = load_dataset(name)
        source = json.loads((ROOT/'results/nested-triangular-repair'/(name+'.json')).read_text())
        for rel, digest in saved['fingerprints'].items():
            assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == digest, rel
        for split, prior in zip(saved['splits'], source['splits']):
            ri, vi = np.array(split['reference_indices']), np.array(split['validation_indices'])
            assert set(ri).isdisjoint(vi)
            raw, labels = X[ri], y[ri]
            A, B = preprocess_fit(raw, X[vi])
            stable = np.zeros(len(ri), bool)
            visits = np.zeros(len(ri), int)
            for fold in split['stability']['folds']:
                a, b = np.array(fold['reference_indices']), np.array(fold['validation_indices'])
                assert set(a).isdisjoint(b)
                TA, TB = preprocess_fit(raw[a], raw[b])
                teacher = LogisticRegression(C=split['teacher_C'], max_iter=2000, random_state=0).fit(
                    triangular_memberships(TA).reshape(len(TA), -1), labels[a])
                W, bias, classes = linear_parameters(teacher)
                W = W.reshape(len(classes), X.shape[1], 3)
                np.testing.assert_allclose(W, fold['coef'], atol=1e-12)
                np.testing.assert_allclose(bias, fold['intercept'], atol=1e-12)
                margins, _ = manual_margins(TB, labels[b], W, bias, classes)
                np.testing.assert_allclose(margins, fold['validation_margins'], atol=1e-10)
                stable[b] = margins > 1e-10
                visits[b] += 1
                counts['crossfit_teachers'] += 1
            classes, sizes = np.unique(labels, return_counts=True)
            assert np.all(visits <= 1)
            assert np.array_equal(visits == 0, np.isin(labels, classes[sizes == 1]))
            assert stable.tolist() == split['stability']['mask']
            stage = prior['stage']
            original, _ = manual_margins(A, labels, np.array(stage['teacher_coef']),
                                         np.array(stage['teacher_intercept']), np.array(stage['teacher_classes']))
            np.testing.assert_allclose(original, split['original_margins'], atol=1e-10)
            protected = stable & (original > 1e-10)
            assert protected.tolist() == split['protected_mask']
            candidate = split['candidate']
            fs = candidate['features']
            target = train(A, labels, fs, split['teacher_C'])
            margins, _ = manual_margins(A, labels, np.array(target['coef']),
                                         np.array(target['intercept']), np.array(target['classes']))
            np.testing.assert_allclose(margins, split['target_margins'], atol=1e-10)
            accepted = bool(np.all(margins[protected] >= .5*original[protected]-1e-8)
                            and np.all(margins[protected] > 1e-10))
            assert accepted == split['accepted']
            all_positive = original > 1e-10
            old_accepted = bool(np.all(margins[all_positive] >= .5*original[all_positive]-1e-8)
                                and np.all(margins[all_positive] > 1e-10))
            counts['newly_accepted_vs_all_positive'] += accepted and not old_accepted
            full = train(A, labels, list(range(X.shape[1])), split['teacher_C'])
            actual, deployed = (target, fs) if accepted else (full, list(range(X.shape[1])))
            assert deployed == split['features']
            np.testing.assert_allclose(actual['coef'], split['fit']['coef'], atol=1e-12)
            np.testing.assert_allclose(actual['intercept'], split['fit']['intercept'], atol=1e-12)
            if accepted:
                assert weight_certificate(A, labels, fs, np.ones(X.shape[1])/np.sqrt(X.shape[1]),
                                          np.array(candidate['weights'])[fs])['feasible']
            np.testing.assert_allclose(score(B, y[vi], actual), split['clean'], atol=1e-12)
            np.testing.assert_allclose(score(B, y[vi], full), split['full_clean'], atol=1e-12)
            counts['excluded_positive'] += int((all_positive & ~protected).sum())
            counts['protected'] += int(protected.sum())
            counts['empty_masks'] += not protected.any()
            counts['compact_deployments'] += len(deployed) < X.shape[1]
            counts['splits'] += 1
        print(name, 'crossfit teachers and subset guarantee audited', flush=True)
    assert counts['splits'] == 44
    counts = {key: int(value) for key, value in counts.items()}
    counts['scope'] = 'scalar_subset_margin_certificate_prediction_replay_not_all_positive_guarantee'
    (FOLDER/'complete-audit.json').write_text(json.dumps(counts, indent=2), encoding='utf-8')
    print(counts)


if __name__ == '__main__':
    main()
