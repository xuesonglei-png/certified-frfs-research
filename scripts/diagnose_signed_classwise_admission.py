"""All528guard metrics: signed losses before class aggregation, no new scores."""
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_simplex_soft5_knn_pilot import ROOT, digest
from continue_soft_topk_knn_pilot import verify
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from signed_classwise_dependency import signed_classwise_dependency

SOURCE = ROOT/'results/majority-hinge-inner-all15'
STARS = ROOT/'results/classwise-star-admission'
FOLDER = ROOT/'results/signed-classwise-admission'
FREEZE = ROOT/'research/signed-classwise-admission-source-freeze.json'


def run(name):
    threadpool_limits(1); FOLDER.mkdir(exist_ok=True)
    freeze = json.loads(FREEZE.read_text())
    for rel, sha in freeze['fingerprints'].items():
        assert digest(ROOT/rel) == sha, rel
    hashes = {**freeze['fingerprints'], (STARS/'complete-audit.json').relative_to(ROOT).as_posix(): digest(STARS/'complete-audit.json'),
              (STARS/(name+'.json')).relative_to(ROOT).as_posix(): digest(STARS/(name+'.json')),
              (SOURCE/(name+'.json')).relative_to(ROOT).as_posix(): digest(SOURCE/(name+'.json'))}
    assert verify(STARS)['artifact_hashes'][name] == digest(STARS/(name+'.json'))
    assert verify(SOURCE)['artifact_hashes'][name] == digest(SOURCE/(name+'.json'))
    path = FOLDER/(name+'.json')
    if path.exists():
        saved = json.loads(path.read_text()); assert saved['fingerprints'] == hashes
        if saved['complete']:
            return
    source = json.loads((SOURCE/(name+'.json')).read_text()); stars = json.loads((STARS/(name+'.json')).read_text())
    X, y = load_dataset(name); ri = np.asarray(source['reference_indices']); raw, labels = X[ri], y[ri]; inner = []
    for fi, (b, sb) in enumerate(zip(source['inner'], stars['inner'], strict=True)):
        assert b['reference_indices'] == sb['reference_indices']; ca = np.asarray(b['reference_indices'])
        A, _ = preprocess_fit(raw[ca], raw[ca]); train_y = labels[ca]; full = np.ones(A.shape[1])/np.sqrt(A.shape[1]); checks = []
        for old, prior in zip(b['stage']['certificate_checks'], sb['checks'], strict=True):
            assert old['id'] == prior['guard_id']
            report = signed_classwise_dependency(A, train_y, old['features'], full, old['weights'],
                       base_dependency=old['certificate']['dependency_certificate'], star_report=prior['certificate'])
            feasible = bool(old['certificate']['majority_feasible'] and report['global_upper_bound'] <= .1
                            and report['class_certificate']['feasible_all_classes'])
            assert not prior['new_feasible'] or feasible
            checks.append({'guard_id': old['id'], 'features': old['features'], 'weights': old['weights'],
                'old_accepted': old['accepted'], 'star_feasible': prior['new_feasible'], 'new_feasible': feasible,
                'new_admission': bool(feasible and not old['accepted']), 'beyond_star_admission': bool(feasible and not prior['new_feasible']),
                'certificate': report, 'classification_scored': False})
        inner.append({'reference_indices': ca.tolist(), 'checks': checks})
        print(name, 'inner', fi+1, 'signed checks', len(checks), 'new admissions', sum(r['new_admission'] for r in checks),
              'beyond star', sum(r['beyond_star_admission'] for r in checks), flush=True)
    saved = {'dataset': name, 'reference_indices': ri.tolist(), 'fingerprints': hashes, 'inner': inner,
             'complete': True, 'goal_achieved': False, 'classification_scored': False,
             'scope': 'signed_classwise_FRS_certificate_tightness_and_admission_only'}
    path.write_text(json.dumps(saved, separators=(',', ':')), encoding='utf-8')


def main():
    names = sorted(verify(STARS)['artifact_hashes']); assert len(names) == 15
    with ProcessPoolExecutor(max_workers=2) as executor:
        for f in as_completed([executor.submit(run, n) for n in names]):
            f.result()
    rows = []
    for n in names:
        saved = json.loads((FOLDER/(n+'.json')).read_text()); checks = [r for b in saved['inner'] for r in b['checks']]
        rows.append({'dataset': n, 'guard_checks': len(checks), 'new_admissions': int(sum(r['new_admission'] for r in checks)),
            'beyond_star_admissions': int(sum(r['beyond_star_admission'] for r in checks)),
            'negative_object_bounds': int(sum(sum(np.asarray(r['certificate']['signed_object_report']['signed_object_upper_bounds']) < -1e-12) for r in checks)),
            'class_bounds_tightened_beyond_star': int(sum(np.sum(np.asarray(r['certificate']['class_certificate']['upper_bounds'])
                < np.asarray(r['certificate']['star_class_certificate']['upper_bounds'])-1e-12) for r in checks))})
    summary = {'rows': rows, 'counts': {k: int(sum(r[k] for r in rows)) for k in rows[0] if k != 'dataset'},
        'datasets': len(names), 'inner_folds': sum(len(json.loads((FOLDER/(n+'.json')).read_text())['inner']) for n in names),
        'artifact_hashes': {n: digest(FOLDER/(n+'.json')) for n in names}, 'independent_audit_required': True,
        'goal_achieved': False, 'classification_scored': False, 'scope': 'signed_class_mean_certificate_no_prediction_claim'}
    (FOLDER/'complete.json').write_text(json.dumps(summary, indent=2), encoding='utf-8'); print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
