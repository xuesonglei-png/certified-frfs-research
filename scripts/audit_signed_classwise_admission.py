"""All signed scalar witness maxima/classes/admissions; stars already audited."""
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from diagnose_signed_classwise_admission import ROOT, SOURCE, STARS, FOLDER, digest
from independent_signed_classwise import SignedWitnessValidator
from continue_soft_topk_knn_pilot import verify
from audit_forest_compact_transfer import equivalent
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit


def audit(name):
    threadpool_limits(1); saved = json.loads((FOLDER/(name+'.json')).read_text()); counts = Counter(datasets=1)
    assert saved['complete'] and not saved['classification_scored']
    for rel, sha in saved['fingerprints'].items():
        assert digest(ROOT/rel) == sha, rel
    assert verify(STARS)['artifact_hashes'][name] == digest(STARS/(name+'.json'))
    assert verify(SOURCE)['artifact_hashes'][name] == digest(SOURCE/(name+'.json'))
    source = json.loads((SOURCE/(name+'.json')).read_text()); stars = json.loads((STARS/(name+'.json')).read_text())
    assert saved['reference_indices'] == source['reference_indices'] == stars['reference_indices']
    X, y = load_dataset(name); ri = np.asarray(saved['reference_indices']); raw, labels = X[ri], y[ri]
    for b, old, sb in zip(saved['inner'], source['inner'], stars['inner'], strict=True):
        assert b['reference_indices'] == old['reference_indices'] == sb['reference_indices']
        ca = np.asarray(b['reference_indices']); A, _ = preprocess_fit(raw[ca], raw[ca]); train_y = labels[ca]
        full = np.ones(A.shape[1])/np.sqrt(A.shape[1]); validator = SignedWitnessValidator(A, train_y, full)
        classes, class_counts = np.unique(train_y, return_counts=True)
        assert len(b['checks']) == len(old['stage']['certificate_checks']) == len(sb['checks'])
        for check, guard, prior in zip(b['checks'], old['stage']['certificate_checks'], sb['checks'], strict=True):
            assert check['guard_id'] == guard['id'] == prior['guard_id'] and check['features'] == guard['features']
            equivalent(check['weights'], guard['weights']); assert check['old_accepted'] == guard['accepted']
            assert check['star_feasible'] == prior['new_feasible'] and not check['classification_scored']
            report = check['certificate']; signed = report['signed_object_report']; previous = prior['certificate']
            equivalent(signed['base_dependency'], guard['certificate']['dependency_certificate'])
            equivalent(report['original_class_certificate'], guard['class_certificate'])
            equivalent(report['star_class_certificate'], previous['class_certificate'])
            values = []
            if signed['objects']:
                assert len(signed['objects']) == len(A)
                for i, recorded in enumerate(signed['objects']):
                    fresh = validator.object(i, check['features'], check['weights']); equivalent(fresh, recorded)
                    equivalent(max(0., fresh['raw_signed_upper_bound']), guard['certificate']['dependency_certificate']['redistributed_object_bounds'][i])
                    values.append(fresh['upper_bound']); counts['independent_signed_object_witness_bounds'] += 1
            else:
                assert len(classes) < 2 or (check['features'] == list(range(A.shape[1])) and np.array_equal(check['weights'], full))
                values = [0.]*len(A)
            equivalent(values, signed['signed_object_upper_bounds']); values = np.asarray(values); bounds = []
            assert len(report['class_reports']) == len(classes)
            for ci, (c, r) in enumerate(zip(classes, report['class_reports'], strict=True)):
                own = train_y == c; raw_independent = float(values[own].mean()); raw_star = raw_independent
                if previous['class_reports']:
                    pr = previous['class_reports'][ci]; selected = np.asarray(pr['selected_indices'], int)
                    remaining = own.copy(); remaining[selected] = False
                    raw_star = float((sum(a['maximum'] for a in pr['coordinates'])+values[remaining].sum())/class_counts[ci])
                padded = lambda v: float(np.clip(v+1e-10*(1+abs(v)), 0, 1))
                bound = min(previous['class_certificate']['upper_bounds'][ci], padded(raw_independent), padded(raw_star))
                equivalent(r, {'class': c.item(), 'count': int(class_counts[ci]), 'signed_objectwise_raw_mean': raw_independent,
                              'shared_star_plus_signed_remainder_raw': raw_star, 'upper_bound': bound,
                              'negative_object_bounds': int(sum(values[own] < -1e-12))})
                bounds.append(bound); counts['independent_signed_class_bounds'] += 1
            cls = report['class_certificate']; equivalent(cls['upper_bounds'], bounds)
            assert cls['classes'] == classes.tolist() and cls['counts'] == class_counts.tolist() and cls['eta'] == .1
            equivalent(cls['max_upper_bound'], max(bounds)); assert cls['feasible_all_classes'] == all(v <= .1 for v in bounds)
            global_bound = min(previous['global_upper_bound'], float(class_counts@bounds/len(A)))
            equivalent(global_bound, report['global_upper_bound'])
            feasible = bool(guard['certificate']['majority_feasible'] and global_bound <= .1 and all(v <= .1 for v in bounds))
            assert check['new_feasible'] == feasible and check['new_admission'] == bool(feasible and not guard['accepted'])
            assert check['beyond_star_admission'] == bool(feasible and not prior['new_feasible'])
            counts['guard_checks'] += 1; counts['new_admissions'] += check['new_admission']
            counts['beyond_star_admissions'] += check['beyond_star_admission']; counts['negative_object_bounds'] += int(sum(values < -1e-12))
            counts['class_bounds_tightened_beyond_star'] += int(sum(np.asarray(bounds) < np.asarray(previous['class_certificate']['upper_bounds'])-1e-12))
        counts['inner_folds'] += 1
        print(name, 'independent signed-neighbor/knots/saturation/class/admission audit', len(b['checks']), flush=True)
    return {'dataset': name, 'counts': dict(counts), 'sha256': digest(FOLDER/(name+'.json'))}


def main():
    summary = json.loads((FOLDER/'complete.json').read_text()); reports = []; counts = Counter()
    with ProcessPoolExecutor(max_workers=2) as executor:
        for f in as_completed([executor.submit(audit, r['dataset']) for r in summary['rows']]):
            r = f.result(); reports.append(r); counts.update(r['counts'])
    assert counts['datasets'] == summary['datasets'] == 15 and counts['inner_folds'] == summary['inner_folds'] == 44
    for key, value in summary['counts'].items():
        assert counts[key] == value, key
    hashes = {r['dataset']: r['sha256'] for r in reports}; equivalent(hashes, summary['artifact_hashes'])
    out = {'passed': True, 'counts': dict(counts), 'artifact_hashes': hashes, 'summary_sha256': digest(FOLDER/'complete.json'),
           'auditor_sha256': digest(Path(__file__)), 'goal_achieved': False, 'classification_scored': False,
           'scope': 'independent_signed_scalar_witness_upper_and_class_aggregation_not_prediction'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(out, indent=2), encoding='utf-8'); print(json.dumps(out, indent=2), flush=True)


if __name__ == '__main__':
    main()
