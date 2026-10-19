"""All-dataset training-only tighter majority-certificate admission diagnosis."""
import json
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_recent_pool_certificate_admission import ROOT, FOLDER as PARENT, digest
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.core import interval_bounds
from audit_flexible_majority_repair_admission import independent_protected
from shared_anchor_enemy_triple import refine_anchor

FOLDER = ROOT/'results/shared-anchor-enemy-triple-diagnosis'


def run(name):
    threadpool_limits(1)
    audit = json.loads((PARENT/'complete-audit.json').read_text())
    source = PARENT/(name+'.json'); assert audit['passed'] and audit['artifact_hashes'][name] == digest(source)
    saved = json.loads(source.read_text())
    files = [source, PARENT/'complete-audit.json', Path(__file__), ROOT/'scripts/shared_anchor_enemy_triple.py',
             ROOT/'research/shared-anchor-enemy-triple-protocol.md', ROOT/'data/processed'/(name+'.npz')]
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in files}
    out = FOLDER/(name+'.json')
    if out.exists():
        old = json.loads(out.read_text()); assert old['fingerprints'] == hashes
        if old['complete']: return old
    X, y = load_dataset(name); ri = np.array(saved['reference_indices'])
    A, _ = preprocess_fit(X[ri], X[ri]); labels = y[ri]
    full = np.ones(A.shape[1])/np.sqrt(A.shape[1])
    protected = independent_protected(A, labels, full)
    records = []
    for parent in saved['pool']:
        fs = parent['features']
        if len(fs) == A.shape[1]: continue
        ordinary = parent['ordinary']; w = np.array(ordinary['weights'])[fs]
        Z = A[:, fs]; lo, hi = interval_bounds(Z, .01)
        nominal = abs(Z[:, None]-Z[None, :])@w
        np.fill_diagonal(nominal, np.inf)
        correct = np.zeros(len(labels), bool); targets = []
        for i in np.flatnonzero(protected):
            friends = np.flatnonzero((labels == labels[i]) & (np.arange(len(labels)) != i))
            friends = friends[np.lexsort((friends, nominal[i, friends]))][:3]
            enemies = np.flatnonzero(labels != labels[i])
            if len(friends) < 3: continue
            result = refine_anchor(lo, hi, int(i), friends, enemies, w)
            correct[i] = result['certified']
            if result['status'] != 'old_pairwise_certificate': targets.append(result)
        counts = [int(np.sum(correct & protected & (labels == c))) for c in np.unique(labels)]
        original = ordinary['certificate']
        oldcounts = original['retained_counts']
        assert all(a >= b for a, b in zip(counts, oldcounts))
        majority = all(c >= r for c, r in zip(counts, original['required']))
        dependency = original['dependency_certificate']['upper_bound'] <= .1 and parent['ordinary_class_certificate']['feasible_all_classes']
        records.append({'features': fs, 'origins': parent['origins'], 'weights': w.tolist(),
                        'old_retained_counts': oldcounts, 'new_retained_counts': counts,
                        'required': original['required'], 'new_majority_feasible': majority,
                        'global_and_class_dependency_feasible': dependency,
                        'old_joint_feasible': original['feasible'] and parent['ordinary_class_certificate']['feasible_all_classes'],
                        'new_joint_feasible': majority and dependency,
                        'targets': targets})
        print(name, 'support', len(records), 'old/new', oldcounts, counts, 'joint', majority and dependency, flush=True)
    output = {'dataset': name, 'reference_indices': ri.tolist(), 'records': records, 'fingerprints': hashes,
              'complete': True, 'query_labels_or_accuracy_used': False, 'independent_audit_required': True,
              'scope': 'same_reference_majority_contract_tighter_computation_not_old_test_pass_or_prediction'}
    assert all(digest(ROOT/rel) == sha for rel, sha in hashes.items())
    out.write_text(json.dumps(output, separators=(',', ':')), encoding='utf-8')
    return output


if __name__ == '__main__':
    FOLDER.mkdir(exist_ok=True)
    names = sorted(json.loads((PARENT/'complete-audit.json').read_text())['artifact_hashes'])
    reports = []
    with ProcessPoolExecutor(max_workers=2) as ex:
        for future in as_completed([ex.submit(run, name) for name in names]): reports.append(future.result())
    counts = Counter(); rows = []
    for report in sorted(reports, key=lambda r: r['dataset']):
        added = 0
        for r in report['records']:
            counts['nonfull_supports'] += 1
            counts['new_joint_admissions'] += r['new_joint_feasible'] and not r['old_joint_feasible']
            added += r['new_joint_feasible'] and not r['old_joint_feasible']
            counts['added_protected_objects'] += sum(r['new_retained_counts'])-sum(r['old_retained_counts'])
            for t in r['targets']:
                counts[t['status']] += 1; counts['LP_checks'] += len(t['checks'])
        rows.append({'dataset': report['dataset'], 'new_joint_admissions': added})
    summary = {'rows': rows, 'counts': dict(counts), 'goal_achieved': False, 'independent_audit_required': True,
               'query_labels_or_accuracy_used': False, 'scope': 'all15_training_admission_not_predictive_evidence'}
    (FOLDER/'complete.json').write_text(json.dumps(summary, indent=2), encoding='utf-8'); print(summary, flush=True)
