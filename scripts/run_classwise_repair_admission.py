"""All15 reference-stage report of frozen flexible-weight repair admission."""
import hashlib
import json
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from flexible_majority_repair import fit_flexible_majority_repair
from certified_frfs.classwise_dependency_certificate import classwise_dependency_certificate

BASE = ROOT/'results/crossfit-label-loss-recent-pilot'
FOLDER = ROOT/'results/classwise-repair-admission'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(name):
    threadpool_limits(1)
    artifact = BASE/(name+'.json')
    audit = json.loads((BASE/'complete-audit.json').read_text())
    assert audit['passed'] and digest(artifact) == audit['artifact_hashes'][name]
    saved = json.loads(artifact.read_text())
    for rel, sha in saved['fingerprints'].items():
        assert digest(ROOT/rel) == sha
    sources = [Path(__file__), ROOT/'scripts/flexible_majority_repair.py',
               ROOT/'research/classwise-repair-admission-protocol.md', ROOT/'scripts/prepare_classwise_repair_admission.py', artifact,
               BASE/'complete-audit.json']
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in sources}
    hashes.update({p.relative_to(ROOT).as_posix(): digest(p) for p in sorted((ROOT/'src/certified_frfs').glob('*.py'))})
    output = FOLDER/(name+'.json')
    if output.exists():
        prior = json.loads(output.read_text())
        assert prior['fingerprints'] == hashes
        if prior['complete']:
            return prior
    stage, d = saved['outer']['stage'], saved['d']
    _, all_labels = load_dataset(name)
    original_classes = classwise_dependency_certificate(all_labels[np.array(saved['reference_indices'])], stage['candidate']['certificate'])
    result = {'dataset': name, 'd': d, 'target_count': stage['target_count'],
              'original_count': stage['actual_count'], 'reference_indices': saved['reference_indices'],
              'fingerprints': hashes, 'attempts': [], 'complete': False, 'ordinary_classwise_certificate': original_classes,
              'query_labels_or_accuracy_used': False}
    if stage['target_count'] == d:
        result.update({'status': 'target_full_no_repair', 'actual_count': d})
    elif stage['actual_count'] < d:
        result.update({'status': 'existing_compact_support_preserved', 'actual_count': stage['actual_count']})
    else:
        ranking = stage['path']['rankings'][0]
        assert ranking['features'] == list(range(d))
        assert len(stage['path']['attempts']) == d and all(not a['accepted'] for a in stage['path']['attempts'])
        X, y = load_dataset(name)
        ri = np.array(saved['reference_indices'])
        A, _ = preprocess_fit(X[ri], X[ri])
        result.update({'status': 'all_repairs_rejected', 'actual_count': d})
        for entry in ranking['choices']:
            removed = entry['removed']
            fs = [a for a in range(d) if a != removed]
            start = time.perf_counter()
            weights, info = fit_flexible_majority_repair(A, y[ri], fs, time_limit=10.)
            if 'weights' in info:
                info['classwise_dependency_certificate'] = classwise_dependency_certificate(y[ri], info['certificate'])
                info['classwise_guard_rejected'] = weights is not None and not info['classwise_dependency_certificate']['feasible_all_classes']
                if info['classwise_guard_rejected']:
                    weights = None
            info['guarded_repair_accepted'] = weights is not None
            result['attempts'].append({'removed': removed, 'fixed_OOF_loss': entry['loss'],
                                       'fit': info, 'wall_seconds': time.perf_counter()-start})
            output.write_text(json.dumps(result, indent=2), encoding='utf-8')
            print(name, 'removed', removed, info['status'], 'seconds', round(result['attempts'][-1]['wall_seconds'], 2), flush=True)
            if weights is not None:
                result.update({'status': 'original_and_classwise_certified_repair_selected', 'actual_count': len(fs),
                               'features': fs, 'weights': weights.tolist()})
                break
    result['complete'] = True
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    FOLDER.mkdir(exist_ok=True)
    audit = json.loads((BASE/'complete-audit.json').read_text())
    names = sorted(audit['artifact_hashes'])
    rows = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run, name) for name in names]):
            rows.append(future.result())
    repaired = sum(r['status'] == 'original_and_classwise_certified_repair_selected' for r in rows)
    report = {'rows': sorted(rows, key=lambda r: r['dataset']), 'repaired_blocked_references': repaired,
              'admission_gate_passed': repaired >= 1, 'goal_achieved': False,
              'mean_removal': float(np.mean([1-r['actual_count']/r['d'] for r in rows])),
              'independent_audit_required': True, 'query_labels_or_accuracy_used': False,
              'scope': 'all15_reference_only_classwise_guarded_repair_admission_not_whole_path_or_prediction_guarantee'}
    (FOLDER/'complete.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print({k: v for k, v in report.items() if k != 'rows'}, flush=True)
