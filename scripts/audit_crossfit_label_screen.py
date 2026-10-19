"""Direct fold-design costs, preprocessing isolation and FRS path replay."""
import hashlib
import json
import numpy as np
from threadpoolctl import threadpool_limits
from review_box_iterations import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.nested_teacher_regularized import inner_splits
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.triangular_classifier import triangular_memberships
from certified_frfs.feasible_transport import weight_certificate


def main():
    threadpool_limits(1)
    folder = ROOT/'results/crossfit-label-screen'
    summary = json.loads((folder/'complete.json').read_text())
    counts = {'references': 0, 'folds': 0, 'ranking_costs': 0, 'certificate_attempts': 0, 'common_counts': 0}
    for row in summary['rows']:
        saved = json.loads((folder/(row['dataset']+'.json')).read_text())
        score = json.loads((ROOT/'results/nested-conditional-feasible'/(row['dataset']+'.json')).read_text())
        label = json.loads((ROOT/'results/nested-conditional-label'/(row['dataset']+'.json')).read_text())
        X, y = load_dataset(row['dataset'])
        for rel, digest in saved['fingerprints'].items():
            assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == digest
        for record, s, t in zip(saved['references'], score['splits'], label['splits']):
            assert record['reference_indices'] == s['reference_indices'] == t['reference_indices']
            ri = np.array(record['reference_indices']); raw, labels = X[ri], y[ri]
            A, _ = preprocess_fit(raw, raw); d = A.shape[1]
            classes, own = np.unique(labels, return_inverse=True)
            target = np.eye(len(classes))[own]-1/len(classes)
            folds = []
            partitions = inner_splits(labels, seed=20261011)
            expected = [{'reference_indices': a.tolist(), 'validation_indices': b.tolist()} for a, b in partitions]
            path = record['crossfit_path']
            assert expected == path['crossfit_splits']
            for a, b in partitions:
                assert set(a).isdisjoint(b) and set(a)|set(b) == set(range(len(labels)))
                P, Q = preprocess_fit(raw[a], raw[b])
                w = 1/(len(classes)*np.bincount(own[a], minlength=len(classes))[own[a]])
                v = 1/(len(np.unique(labels[b]))*np.bincount(own[b], minlength=len(classes))[own[b]])
                folds.append((triangular_memberships(P), triangular_memberships(Q), target[a], target[b], w, v))
                counts['folds'] += 1
            cache = {}
            def loss(fs):
                identity = tuple(fs)
                if identity not in cache:
                    values = []
                    for P, Q, train_target, query_target, w, v in folds:
                        D = np.column_stack([np.ones(len(P)), P[:, fs].reshape(len(P), -1)])
                        E = np.column_stack([np.ones(len(Q)), Q[:, fs].reshape(len(Q), -1)])
                        penalty = np.eye(D.shape[1])*.001; penalty[0, 0] = 0
                        beta = np.linalg.solve(D.T@(w[:, None]*D)+penalty, D.T@(w[:, None]*train_target))
                        values.append(float(v@np.mean((E@beta-query_target)**2, axis=1)))
                    cache[identity] = float(np.mean(values))
                return cache[identity]
            active = list(range(d)); attempt_i = 0; trace_i = 1
            for ranking in path['rankings']:
                assert ranking['features'] == active
                direct = {a: loss([b for b in active if b != a]) for a in active}
                choices = ranking['choices']
                assert set(direct) == {r['removed'] for r in choices}
                ordered = [direct[r['removed']] for r in choices]
                np.testing.assert_allclose(ordered, [r['loss'] for r in choices], rtol=1e-8, atol=1e-10)
                assert np.all(np.diff(ordered) >= -1e-10)
                counts['ranking_costs'] += len(choices)
                accepted = False
                for choice in choices:
                    attempt = path['attempts'][attempt_i]; attempt_i += 1
                    fs = [a for a in active if a != choice['removed']]
                    assert attempt['features'] == fs and attempt['removed'] == choice['removed']
                    cert = weight_certificate(A, labels, fs, np.ones(d)/np.sqrt(d), attempt['weights'])
                    assert cert == attempt['certificate'] and cert['feasible'] == attempt['accepted']
                    np.testing.assert_allclose(attempt['reconstruction_loss'], loss(fs), rtol=1e-8, atol=1e-10)
                    counts['certificate_attempts'] += 1
                    if cert['feasible']:
                        step = path['trace'][trace_i]; trace_i += 1
                        assert step['features'] == fs
                        np.testing.assert_allclose(np.array(step['weights'])[fs], attempt['weights'], atol=1e-12)
                        active = fs; accepted = True
                        break
                if not accepted:
                    assert ranking == path['rankings'][-1]
            assert attempt_i == len(path['attempts']) and trace_i == len(path['trace'])
            paths = {'crossfit': path, 'teacher_score': s['stage']['conditional_path'],
                     'training_label': t['stage']['conditional_path']}
            common = sorted(set.intersection(*(set(len(r['features']) for r in p['trace']) for p in paths.values())))
            assert common == [r['size'] for r in record['same_count_losses']]
            for match in record['same_count_losses']:
                for method, p in paths.items():
                    fs = next(r['features'] for r in p['trace'] if len(r['features']) == match['size'])
                    np.testing.assert_allclose(match[method], loss(fs), rtol=1e-8, atol=1e-10)
                counts['common_counts'] += 1
            counts['references'] += 1
        print(row['dataset'], 'direct crossfit costs and certificates audited', flush=True)
    assert counts['references'] == 44
    counts['scope'] = 'fold_local_preprocessing_direct_costs_candidate_order_and_original_FRS_not_prediction_advantage'
    (folder/'complete-audit.json').write_text(json.dumps(counts, indent=2), encoding='utf-8')
    print(counts, flush=True)


if __name__ == '__main__':
    main()
