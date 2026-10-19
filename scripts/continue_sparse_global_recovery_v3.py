"""Memory-bounded sparse-global recovery using streaming legacy artifacts.

The failed v1/v2 study is immutable.  This version reads only the legacy
fields needed to refit the sparse arm and independently audits the new fit;
fixed/global legacy traces are not copied or silently treated as audited.
"""
import ctypes
import hashlib
import ijson
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from fit_sparse_global_transition_frs_v2 import fit_sparse_global_transition_target
from audit_sparse_global_transition_frs_v2 import check_sparse_global_transition_fit
from sparse_global_frs_training_ablation import STARTS, CappedThirdEnemyHinge
from run_sparse_global_frs_training import prerequisites, DIAGNOSTIC

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "results" / "sparse-global-frs-training"
OUT = ROOT / "results" / "sparse-global-frs-training-recovery-v3"
FREEZE = ROOT / "research" / "sparse-global-frs-training-recovery-v3-source-freeze.json"
STATUS = ROOT / "results" / "sparse-global-frs-training-recovery-v3-chain-status.json"


class MemoryStatus(ctypes.Structure):
    _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def set_status(phase, **fields):
    STATUS.write_text(json.dumps({'phase': phase, 'updated_unix': time.time(),
                                  'goal_achieved': False, **fields}, indent=2), encoding='utf-8')


def read_item(path, prefix):
    """Read one JSON prefix without materialising unrelated sibling values."""
    with Path(path).open('rb') as stream:
        iterator = ijson.items(stream, prefix)
        try:
            return next(iterator)
        except StopIteration as exc:
            raise KeyError(prefix) from exc


def old_sparse_metadata(path):
    return {
        'reference_indices': read_item(path, 'reference_indices'),
        'training_indices': read_item(path, 'training_indices'),
        'fingerprints': read_item(path, 'fingerprints'),
        'features': read_item(path, 'block.features'),
        'anchor': read_item(path, 'block.anchor'),
        'starts': read_item(path, 'block.starts'),
        'sparse_arm': read_item(path, 'block.arms.global_signed_sparse'),
    }


def wait_memory():
    required = 2 * 1024**3
    while True:
        state = MemoryStatus(); state.dwLength = ctypes.sizeof(state)
        assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state))
        if state.ullAvailPhys >= required:
            return
        set_status('waiting_memory', required_bytes=required, available_bytes=state.ullAvailPhys)
        time.sleep(5.)


def run(name, frozen):
    output = OUT / (name + '.json'); audit_path = OUT / (name + '.audit.json')
    if output.exists() and audit_path.exists():
        report = json.loads(audit_path.read_text())
        assert report['passed'] and report['sha256'] == digest(output)
        return report
    wait_memory(); set_status('streaming_legacy_sparse_metadata', dataset=name)
    source_path = OLD / (name + '.json'); source_sha = digest(source_path)
    meta = old_sparse_metadata(source_path)
    old_arm = meta['sparse_arm']; counts = Counter(datasets=1)
    X, y = load_dataset(name)
    ri = np.asarray(meta['reference_indices']); ca = np.asarray(meta['training_indices'])
    A, _ = preprocess_fit(X[ri][ca], X[ri][ca]); labels = y[ri][ca]
    targets = {}
    if old_arm['available']:
        set_status('refitting_sparse_arm', dataset=name)
        for start in STARTS:
            if name == 'ecoli':
                evidence = json.loads((ROOT / 'results/sparse-global-v2-ecoli-recovery/complete-audit.json').read_text())
                path = ROOT / 'results/sparse-global-v2-ecoli-recovery' / (start + '.json')
                assert evidence['passed'] and evidence['source_sha256'] == source_sha
                assert next(r['sha256'] for r in evidence['rows'] if r['start'] == start) == digest(path)
                saved = json.loads(path.read_text()); target, fit = np.asarray(saved['target']), saved['fit']
            else:
                target, fit = fit_sparse_global_transition_target(
                    A, labels, meta['features'], meta['anchor'],
                    full_reference=np.ones(A.shape[1]) / np.sqrt(A.shape[1]),
                    reference_scope='original_full', initialization=np.asarray(meta['starts'][start]),
                    initialization_scope=start)
            record = {'features': meta['features'], 'anchor': meta['anchor'],
                      'target': target.tolist(), 'fit': fit, 'classification_scored': False}
            check_sparse_global_transition_fit(A, labels, record, counts)
            targets[start] = record
            print(name, start, 'sparse v3 independent audit PASS', flush=True)
    sparse_target = targets.get(old_arm.get('selected_start'))
    if sparse_target is None and targets:
        sparse_target = min(targets.values(), key=lambda r: r['fit']['training_objective_final'])
    evaluation = None
    if sparse_target is not None:
        mass = np.sqrt(A.shape[1]); hinge = CappedThirdEnemyHinge(A[:, meta['features']], labels,
                                                                    np.asarray(meta['anchor']) / mass)
        evaluation = {'training_error_initial': float(hinge.alpha @ np.minimum(1., hinge.losses(np.asarray(meta['anchor']) / mass))),
                      'training_error_final': float(hinge.alpha @ np.minimum(1., hinge.losses(np.asarray(sparse_target['target']) / mass))),
                      'optimizer_failures': sum(r['fit']['subproblem_failures'] for r in targets.values()),
                      'available': True, 'classification_scored': False}
    packet = {'dataset': name, 'reference_indices': ri.tolist(), 'training_indices': ca.tolist(),
              'features': meta['features'], 'anchor': meta['anchor'], 'starts': meta['starts'],
              'legacy_source_sha256': source_sha, 'legacy_fingerprints': meta['fingerprints'],
              'sparse_arm': old_arm, 'recovered_sparse_targets': targets, 'evaluation': evaluation,
              'complete': True, 'classification_scored': False, 'goal_achieved': False,
              'scope': 'streamed_sparse_arm_only_dual_path_recovery_no_inherited_fixed_global_claim'}
    output.write_text(json.dumps(packet, separators=(',', ':')), encoding='utf-8')
    report = {'passed': True, 'dataset': name, 'sha256': digest(output), 'source_sha256': source_sha,
              'counts': dict(counts), 'classification_scored': False, 'goal_achieved': False}
    audit_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main():
    threadpool_limits(1); OUT.mkdir(exist_ok=True); audits = prerequisites()
    names = sorted(audits['capped-frs-training']['artifact_hashes'])
    files = sorted((ROOT / 'scripts').glob('*.py')) + sorted((ROOT / 'src/certified_frfs').glob('*.py'))
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in files}
    if not FREEZE.exists():
        FREEZE.write_text(json.dumps({'created_unix': time.time(), 'fingerprints': hashes,
                                      'datasets': names, 'streaming_legacy_reader': True}, indent=2), encoding='utf-8')
    frozen = json.loads(FREEZE.read_text())['fingerprints']
    for rel, sha in frozen.items():
        assert digest(ROOT / rel) == sha, rel
    reports = [run(name, frozen) for name in names]
    counts = Counter(); rows = []
    for report in reports:
        counts.update(report['counts']); packet = json.loads((OUT / (report['dataset'] + '.json')).read_text())
        rows.append({'dataset': report['dataset'], 'available': packet['evaluation'] is not None,
                     'evaluation': packet['evaluation']})
    summary = {'datasets': len(rows), 'rows': rows, 'counts': dict(counts),
               'classification_scored': False, 'goal_achieved': False,
               'streaming_legacy_reader': True, 'independent_audit_passed': True}
    summary_path = OUT / 'complete.json'; summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    complete_audit = {'passed': True, 'datasets': len(rows), 'summary_sha256': digest(summary_path),
                      'artifact_hashes': {r['dataset']: digest(OUT / (r['dataset'] + '.json')) for r in rows},
                      'goal_achieved': False}
    (OUT / 'complete-audit.json').write_text(json.dumps(complete_audit, indent=2), encoding='utf-8')
    set_status('complete_audited_training_only', audit_sha256=digest(OUT / 'complete-audit.json'))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        set_status('failed', error=f'{type(exc).__name__}: {exc}')
        raise
