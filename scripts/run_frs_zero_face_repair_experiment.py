"""Frozen ALL15 training diagnostic; preserve every raw-zero repair outcome."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_simplex_soft5_knn_pilot import ROOT, digest
from continue_soft_topk_knn_pilot import verify
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from frs_zero_face_repair import repair_diagnostic
from frs_coordinate_minimum import MODES

SOURCE = ROOT/'results/frs-coordinate-diagnostic-v2'
FOLDER = ROOT/'results/frs-zero-face-repair'
FREEZE = ROOT/'research/frs-zero-face-repair-source-freeze.json'
PROTOCOL = ROOT/'research/FRS-zero-face-repair-protocol.md'


def prerequisites():
    audit = verify(SOURCE)
    review = json.loads((ROOT/'research/FRS-coordinate-minimum-review-v2.json').read_text())
    assert review['audit_sha256'] == digest(SOURCE/'complete-audit.json')
    assert len(audit['artifact_hashes']) == len(review['all_dataset_rows']) == 15
    assert not review['classification_scored']
    return audit


def frozen():
    hashes = json.loads(FREEZE.read_text())['fingerprints']
    for rel, sha in hashes.items():
        assert digest(ROOT/rel) == sha, rel
    return hashes


def builtin(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def statistics(saved):
    """Per-vector zeros only: never merge witnesses from different candidates."""
    p = len(saved['original_support'])
    out = {'dataset':saved['dataset'], 'available':saved['available'],
           'original_support_size':p, 'fit_seconds':saved['fit_seconds'], 'modes':{}}
    for mode in MODES:
        values = [] if saved['repair'] is None else saved['repair']['modes'][mode]
        accepted = [r for r in values if r['target'] is not None]
        old = [r for r in accepted if r['reused_accepted']]
        zeros = lambda rows: [sum(float(v) == 0. for v in r['target']) for r in rows]
        statuses = {s:sum(r['status'] == s for r in values) for s in sorted({r['status'] for r in values})}
        out['modes'][mode] = {'raw_zero_candidates':len(values),
            'original_accepted':len(old), 'attempted_repairs':sum(not r['reused_accepted'] for r in values),
            'new_repaired_accepted':sum(r['repaired_accepted'] for r in values),
            'total_accepted_vectors':len(accepted), 'status_counts':statuses,
            'old_max_joint_exact_zeros_in_one_vector':max(zeros(old), default=0),
            'new_max_joint_exact_zeros_in_one_vector':max(zeros(accepted), default=0),
            'accepted_vector_exact_support_sizes':[p-z for z in zeros(accepted)]}
    return out


def run(name):
    threadpool_limits(1); parent = prerequisites(); hashes = frozen()
    path = FOLDER/(name+'.json'); FOLDER.mkdir(exist_ok=True)
    if path.exists():
        saved = json.loads(path.read_text()); assert saved['fingerprints'] == hashes
        assert saved['source_sha256'] == digest(SOURCE/(name+'.json'))
        if saved['complete']:
            return
        raise RuntimeError('Incomplete artifact retained; use a new version rather than overwrite')
    source_path = SOURCE/(name+'.json')
    assert parent['artifact_hashes'][name] == digest(source_path)
    source = json.loads(source_path.read_text())
    X, y = load_dataset(name)
    ri, ca = np.asarray(source['reference_indices']), np.asarray(source['training_indices'])
    A, _ = preprocess_fit(X[ri][ca], X[ri][ca]); start = time.perf_counter()
    repair = None if source['diagnostic'] is None else repair_diagnostic(A, y[ri][ca], source['diagnostic'])
    saved = {'dataset':name, 'd':source['d'], 'fold':0, 'available':source['strict_cell_available'],
        'reference_indices':source['reference_indices'], 'training_indices':source['training_indices'],
        'original_support':source['original_support'], 'original_anchor':source['original_anchor'],
        'source_sha256':digest(source_path), 'fingerprints':hashes, 'repair':repair,
        'fit_seconds':time.perf_counter()-start, 'complete':True,
        'classification_scored':False, 'goal_achieved':False,
        'scope':'training_only_numerical_repair_no_prediction_no_merging_individual_deletions'}
    path.write_text(json.dumps(saved, default=builtin, separators=(',', ':')), encoding='utf-8')
    print(name, json.dumps(statistics(saved)['modes']), flush=True)


def summarize(names):
    rows = [statistics(json.loads((FOLDER/(n+'.json')).read_text())) for n in names]
    keys = ('raw_zero_candidates', 'original_accepted', 'attempted_repairs',
            'new_repaired_accepted', 'total_accepted_vectors')
    counts = {mode:{k:sum(r['modes'][mode][k] for r in rows) for k in keys} for mode in MODES}
    summary = {'rows':rows, 'datasets':len(names), 'counts':counts,
        'artifact_hashes':{n:digest(FOLDER/(n+'.json')) for n in names},
        'classification_scored':False, 'goal_achieved':False, 'independent_audit_required':True,
        'limits':['Numerical repair of exposed first training folds is not predictive superiority.',
                  'Only zeros in the same verified vector form a joint deletion witness.',
                  'No-positive-slack failure is not global impossibility of valid reduction.',
                  'All original constraints and deployed coordinates remain unchanged.']}
    (FOLDER/'complete.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(counts, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--dataset'); args = p.parse_args()
    if args.dataset:
        run(args.dataset)
    else:
        names = sorted(prerequisites()['artifact_hashes'])
        for name in names:
            run(name)
        summarize(names)


if __name__ == '__main__':
    main()
