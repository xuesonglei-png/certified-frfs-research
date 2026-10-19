"""Raw-design crossfit class-balanced error replay and original FRS checks."""
import hashlib
import json
import numpy as np
from threadpoolctl import threadpool_limits
from review_box_iterations import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.nested_teacher_regularized import inner_splits
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.triangular_classifier import triangular_memberships
from certified_frfs.crossfit_classification_backward import crossfit_classification_path
from certified_frfs.feasible_transport import weight_certificate


def main():
    threadpool_limits(1)
    folder = ROOT/'results/crossfit-classification-screen'
    summary = json.loads((folder/'complete.json').read_text())
    counts = {'references': 0, 'ranking_costs': 0, 'certificate_attempts': 0, 'common_counts': 0}
    for row in summary['rows']:
        saved = json.loads((folder/(row['dataset']+'.json')).read_text())
        score = json.loads((ROOT/'results/nested-conditional-feasible'/(row['dataset']+'.json')).read_text())
        mse = json.loads((ROOT/'results/crossfit-label-screen'/(row['dataset']+'.json')).read_text())
        X, y = load_dataset(row['dataset'])
        for rel, digest in saved['fingerprints'].items():
            assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == digest
        for record, s, t in zip(saved['references'], score['splits'], mse['references']):
            assert record['reference_indices'] == s['reference_indices'] == t['reference_indices']
            ri = np.array(record['reference_indices']); raw, labels = X[ri], y[ri]
            A, _ = preprocess_fit(raw, raw); d = A.shape[1]
            path = record['classification_path']
            assert path == crossfit_classification_path(raw, labels)
            classes, own = np.unique(labels, return_inverse=True)
            target = np.eye(len(classes))[own]-1/len(classes)
            folds = []
            for a, b in inner_splits(labels, seed=20261011):
                P, Q = preprocess_fit(raw[a], raw[b])
                w = 1/(len(classes)*np.bincount(own[a], minlength=len(classes))[own[a]])
                folds.append((triangular_memberships(P), triangular_memberships(Q), target[a], target[b], w, own[b]))
            cache = {}
            def loss(fs):
                key = tuple(fs)
                if key not in cache:
                    errors, values = [], []
                    for P, Q, train_target, query_target, w, truth in folds:
                        D = np.column_stack([np.ones(len(P)), P[:, fs].reshape(len(P), -1)])
                        E = np.column_stack([np.ones(len(Q)), Q[:, fs].reshape(len(Q), -1)])
                        penalty = np.eye(D.shape[1])*.001; penalty[0, 0] = 0
                        beta = np.linalg.solve(D.T@(w[:, None]*D)+penalty, D.T@(w[:, None]*train_target))
                        scores = E@beta; pred = scores.argmax(axis=1)
                        errors.append(float(np.mean([np.mean(pred[truth == c] != c) for c in np.unique(truth)])))
                        residual = np.mean((scores-query_target)**2, axis=1)
                        values.append(float(np.mean([residual[truth == c].mean() for c in np.unique(truth)])))
                    cache[key] = (float(np.mean(errors)), float(np.mean(values)))
                return cache[key]
            for ranking in path['rankings']:
                active = ranking['features']
                expected = {a: loss([b for b in active if b != a]) for a in active}
                choices = ranking['choices']
                assert set(expected) == {r['removed'] for r in choices}
                direct = [expected[r['removed']] for r in choices]
                np.testing.assert_allclose(direct, [(r['balanced_error'], r['MSE']) for r in choices], rtol=1e-8, atol=1e-10)
                for left, right in zip(direct, direct[1:]):
                    assert left[0] <= right[0]+1e-12
                    if abs(left[0]-right[0]) <= 1e-12:
                        assert left[1] <= right[1]+1e-10
                counts['ranking_costs'] += len(choices)
            for attempt in path['attempts']:
                fs = attempt['features']
                cert = weight_certificate(A, labels, fs, np.ones(d)/np.sqrt(d), attempt['weights'])
                assert cert == attempt['certificate'] and cert['feasible'] == attempt['accepted']
                np.testing.assert_allclose(loss(fs), [attempt['crossfit_balanced_error'], attempt['reconstruction_loss']], rtol=1e-8, atol=1e-10)
                counts['certificate_attempts'] += 1
            paths = {'classification': path, 'teacher_score': s['stage']['conditional_path'], 'crossfit_MSE': t['crossfit_path']}
            common = sorted(set.intersection(*(set(len(r['features']) for r in p['trace']) for p in paths.values())))
            assert common == [r['size'] for r in record['same_count_losses']]
            for match in record['same_count_losses']:
                for method, p in paths.items():
                    fs = next(r['features'] for r in p['trace'] if len(r['features']) == match['size'])
                    np.testing.assert_allclose([match[method]['balanced_error'], match[method]['MSE']], loss(fs), rtol=1e-8, atol=1e-10)
                counts['common_counts'] += 1
            counts['references'] += 1
        print(row['dataset'], 'direct crossfit classification costs and FRS audited', flush=True)
    assert counts['references'] == 44
    counts['scope'] = 'raw_fold_design_balanced_error_ranking_and_FRS_replay_not_test_advantage'
    (folder/'complete-audit.json').write_text(json.dumps(counts, indent=2), encoding='utf-8')
    print(counts, flush=True)


if __name__ == '__main__':
    main()
