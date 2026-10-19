"""Independent class means plus original-certified guarded repair verification."""
import json
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_classwise_repair_admission import FOLDER, BASE, digest
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.feasible_transport import weight_certificate
from audit_flexible_majority_repair_admission import check_repair, independent_protected
from audit_forest_compact_transfer import equivalent

ROOT = Path(__file__).resolve().parents[1]


def independent_class_certificate(y, cert):
    dep = cert['dependency_certificate']
    values = np.array(dep.get('redistributed_object_bounds', np.zeros(len(y))))
    if 'redistributed_object_bounds' not in dep:
        assert dep['upper_bound'] == dep['raw_redistributed_bound'] == 0.
    classes, counts = np.unique(y, return_counts=True)
    raw = np.array([sum(values[y == c])/sum(y == c) for c in classes])
    upper = np.minimum(1., raw+1e-10*(1+abs(raw))) if np.any(values) else np.zeros(len(classes))
    return {'classes': classes.tolist(), 'counts': counts.tolist(), 'upper_bounds': upper.tolist(),
            'max_upper_bound': float(max(upper)), 'eta': .1, 'feasible_all_classes': bool(max(upper) <= .1),
            'scope': 'same_finite_reference_shared_box_class_prior_mixtures_not_population_or_classifier_risk'}


def audit(name):
    threadpool_limits(1)
    saved = json.loads((FOLDER/(name+'.json')).read_text())
    assert saved['complete'] and not saved['query_labels_or_accuracy_used']
    for rel, sha in saved['fingerprints'].items():
        assert digest(ROOT/rel) == sha, rel
    prior = json.loads((BASE/(name+'.json')).read_text())
    assert prior['reference_indices'] == saved['reference_indices']
    X, y = load_dataset(name)
    ri = np.array(saved['reference_indices'])
    labels = y[ri]
    equivalent(saved['ordinary_classwise_certificate'], independent_class_certificate(labels, prior['outer']['stage']['candidate']['certificate']))
    counts = Counter(datasets=1)
    if not saved['attempts']:
        assert saved['actual_count'] == saved['original_count']
        return counts
    A, _ = preprocess_fit(X[ri], X[ri])
    full = np.ones(saved['d'])/np.sqrt(saved['d'])
    protected = independent_protected(A, labels, full)
    choices = prior['outer']['stage']['path']['rankings'][0]['choices']
    for choice, record in zip(choices, saved['attempts']):
        assert choice['removed'] == record['removed'] and choice['loss'] == record['fixed_OOF_loss']
        info = record['fit']
        assert info['features'] == [a for a in range(saved['d']) if a != record['removed']]
        assert info['time_limit'] == 10.
        accepted = False
        if 'weights' in info:
            weights = np.array(info['weights'])
            fresh = weight_certificate(A, labels, info['features'], full, weights, protected=protected)
            equivalent(fresh, info['certificate'])
            classes = independent_class_certificate(labels, fresh)
            equivalent(classes, info['classwise_dependency_certificate'])
            original_passed = info['status'] == 'original_certificate_passed'
            assert original_passed == fresh['feasible']
            assert info['classwise_guard_rejected'] == (original_passed and not classes['feasible_all_classes'])
            accepted = original_passed and classes['feasible_all_classes']
            if original_passed:
                check_repair(A, labels, info, counts)
            counts['independent_class_aggregations'] += 1
            counts['classwise_rejected_original_repairs'] += info['classwise_guard_rejected']
        assert info['guarded_repair_accepted'] == accepted
        counts['attempts'] += 1
        if accepted:
            assert record is saved['attempts'][-1]
            assert saved['status'] == 'original_and_classwise_certified_repair_selected'
            assert saved['actual_count'] == saved['target_count']
            equivalent(saved['weights'], info['weights'])
            counts['guarded_repairs'] += 1
            break
    else:
        assert len(saved['attempts']) == len(choices) and saved['actual_count'] == saved['original_count']
    return counts


if __name__ == '__main__':
    saved = json.loads((FOLDER/'complete.json').read_text())
    counts = Counter()
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(audit, r['dataset']) for r in saved['rows']]):
            counts.update(future.result())
    assert counts['guarded_repairs'] == saved['repaired_blocked_references']
    assert saved['admission_gate_passed'] == (counts['guarded_repairs'] >= 1)
    report = {'passed': True, 'counts': dict(counts), 'summary_sha256': digest(FOLDER/'complete.json'),
              'auditor_sha256': digest(Path(__file__)), 'goal_achieved': False,
              'scope': 'independent_original_FRS_and_class_guard_candidate_admission_not_whole_path_or_prediction'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(report, flush=True)
