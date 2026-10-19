"""Frozen selected-support label surrogate transfer; diagnostic, no tuning."""
import json
import numpy as np
from threadpoolctl import threadpool_limits
from review_box_iterations import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.triangular_classifier import triangular_memberships


def main():
    threadpool_limits(1)
    folder = ROOT/'results/nested-conditional-label'
    summary = json.loads((folder/'complete.json').read_text())
    records = []
    for row in summary['rows']:
        saved = json.loads((folder/(row['dataset']+'.json')).read_text())
        X, y = load_dataset(row['dataset'])
        for i, split in enumerate(saved['splits']):
            ri, vi = np.array(split['reference_indices']), np.array(split['validation_indices'])
            A, B = preprocess_fit(X[ri], X[vi])
            labels, truth = y[ri], y[vi]
            classes, own, counts = np.unique(labels, return_inverse=True, return_counts=True)
            assert set(truth).issubset(classes)
            support = split['stage']['policies'][split['chosen']]['candidate']['features']
            designs = [np.column_stack([np.ones(len(Q)), triangular_memberships(Q[:, support]).reshape(len(Q), -1)]) for Q in (A, B)]
            target = np.eye(len(classes))[own]-1/len(classes)
            weights = 1/(len(classes)*counts[own])
            P = np.eye(designs[0].shape[1])*.001; P[0, 0] = 0
            beta = np.linalg.solve(designs[0].T@(weights[:, None]*designs[0])+P, designs[0].T@(weights[:, None]*target))
            training = float(weights@np.mean((designs[0]@beta-target)**2, axis=1))
            test_own = np.argmax(truth[:, None] == classes, axis=1)
            test_target = np.eye(len(classes))[test_own]-1/len(classes)
            test_counts = np.bincount(test_own, minlength=len(classes))
            test_weights = 1/(len(np.unique(truth))*test_counts[test_own])
            validation = float(test_weights@np.mean((designs[1]@beta-test_target)**2, axis=1))
            records.append({'dataset': row['dataset'], 'split': i, 'features': support,
                'training_label_MSE': training, 'query_label_MSE': validation,
                'selected_inner_BA': split['selected_config']['score'],
                'query_BA': split['metrics']['clean'], 'full_query_BA': split['full_metrics']['clean'],
                'query_gap_pp': 100*(split['metrics']['clean']-split['full_metrics']['clean'])})
    assert len(records) == 44
    payload = {'records': records,
        'scope': 'posthoc_frozen_support_surrogate_transfer_failure_diagnostic_no_tuning_or_selection',
        'interpretation': 'Training label reconstruction and exact reference certification do not guarantee unseen classification.'}
    (folder/'surrogate-transfer-diagnostic.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(sorted(records, key=lambda r:r['query_gap_pp'])[:5], flush=True)


if __name__ == '__main__':
    main()
