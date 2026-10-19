"""Posthoc frozen-support ridge/logistic mismatch; never selects a method."""
import json
import numpy as np
from threadpoolctl import threadpool_limits
from review_box_iterations import ROOT, paired
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.triangular_classifier import triangular_memberships


def BA(pred, truth):
    return float(np.mean([np.mean(pred[truth == c] == c) for c in np.unique(truth)]))


def main():
    threadpool_limits(1)
    folder = ROOT/'results/nested-crossfit-classification-rounded'
    summary = json.loads((folder/'complete.json').read_text())
    rows, records = [], []
    for row in summary['rows']:
        saved = json.loads((folder/(row['dataset']+'.json')).read_text())
        X, y = load_dataset(row['dataset']); group = []
        for i, split in enumerate(saved['splits']):
            ri, vi = np.array(split['reference_indices']), np.array(split['validation_indices'])
            A, B = preprocess_fit(X[ri], X[vi]); labels, truth = y[ri], y[vi]
            stage = split['stage']; chosen = stage['policies'][split['chosen']]
            fs = chosen['candidate']['features']
            classes, own, counts = np.unique(labels, return_inverse=True, return_counts=True)
            target = np.eye(len(classes))[own]-1/len(classes)
            weights = 1/(len(classes)*counts[own]); results = {}
            ridge_predictions = {}
            for method, support in (('selected_ridge', fs), ('full_ridge', list(range(A.shape[1])))):
                D = np.column_stack([np.ones(len(A)), triangular_memberships(A[:, support]).reshape(len(A), -1)])
                Q = np.column_stack([np.ones(len(B)), triangular_memberships(B[:, support]).reshape(len(B), -1)])
                P = np.eye(D.shape[1])*.001; P[0, 0] = 0
                beta = np.linalg.solve(D.T@(weights[:, None]*D)+P, D.T@(weights[:, None]*target))
                train_pred, query_pred = classes[(D@beta).argmax(axis=1)], classes[(Q@beta).argmax(axis=1)]
                results[method] = {'reference_BA': BA(train_pred, labels), 'query_BA': BA(query_pred, truth)}
                ridge_predictions[method] = (train_pred, query_pred)
            fit = chosen['fit']
            preds = [np.array(stage['teacher_classes'])[(np.einsum('ndk,qdk->nq', triangular_memberships(Q),
                     np.array(fit['coef']))+np.array(fit['intercept'])).argmax(axis=1)] for Q in (A, B)]
            np.testing.assert_allclose(BA(preds[1], truth), split['metrics']['clean'], atol=1e-12)
            path_step = next(r for r in stage['conditional_path']['trace'] if r['features'] == fs)
            compact_ridge, full_ridge = results['selected_ridge']['query_BA'], results['full_ridge']['query_BA']
            record = {'dataset': row['dataset'], 'split': i, 'features': fs, 'ridge': results,
                'internal_selected_ridge_error': path_step['crossfit_balanced_error'],
                'internal_full_ridge_error': stage['conditional_path']['trace'][0]['crossfit_balanced_error'],
                'certified_query_BA': split['metrics']['clean'], 'certified_full_query_BA': split['full_metrics']['clean'],
                'ridge_support_query_gain': compact_ridge-full_ridge,
                'certified_support_query_gain': split['metrics']['clean']-split['full_metrics']['clean'],
                'selected_classifier_query_difference': split['metrics']['clean']-compact_ridge,
                'reference_prediction_disagreement': float(np.mean(preds[0] != ridge_predictions['selected_ridge'][0])),
                'query_prediction_disagreement': float(np.mean(preds[1] != ridge_predictions['selected_ridge'][1]))}
            records.append(record); group.append(record)
        rows.append({'dataset': row['dataset'], **{key: float(np.mean([r[key] for r in group])) for key in
            ('ridge_support_query_gain', 'certified_support_query_gain', 'selected_classifier_query_difference',
             'reference_prediction_disagreement', 'query_prediction_disagreement')}})
    assert len(records) == 44
    result = {'rows': rows, 'records': records,
        'ridge_support_query_gain': paired([r['ridge_support_query_gain'] for r in rows]),
        'certified_support_query_gain': paired([r['certified_support_query_gain'] for r in rows]),
        'selected_classifier_difference': paired([r['selected_classifier_query_difference'] for r in rows]),
        'scope': 'posthoc_frozen_support_classifier_alignment_diagnostic_no_selection_or_risk_claim'}
    (folder/'classifier-alignment-diagnostic.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print({key: value for key, value in result.items() if key not in ('rows', 'records')}, flush=True)


if __name__ == '__main__':
    main()
