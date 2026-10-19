"""Independent zero-face repair reconstruction, source alignment and full summary."""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_frs_zero_face_repair_experiment import ROOT, SOURCE, FOLDER, frozen, prerequisites, digest
from audit_frs_zero_face_repair import check_diagnostic
from audit_forest_compact_transfer import equivalent
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit

MODES = ('majority_only', 'stable_cell_only', 'one_sided', 'pair_transition')


def independent_statistics(saved):
    p = len(saved['original_support'])
    row = {'dataset':saved['dataset'], 'available':saved['available'],
        'original_support_size':p, 'fit_seconds':saved['fit_seconds'], 'modes':{}}
    for mode in MODES:
        records = [] if saved['repair'] is None else saved['repair']['modes'][mode]
        statuses = Counter(); supports, old_zeros = [], []
        old_count = new_count = 0
        for record in records:
            statuses[record['status']] += 1
            if record['target'] is None:
                assert not record['reused_accepted'] and not record['repaired_accepted']
                continue
            size = int(np.count_nonzero(np.asarray(record['target'])))
            supports.append(size)
            if record['reused_accepted']:
                old_count += 1; old_zeros.append(p-size)
            if record['repaired_accepted']:
                new_count += 1
        row['modes'][mode] = {'raw_zero_candidates':len(records), 'original_accepted':old_count,
            'attempted_repairs':len(records)-old_count, 'new_repaired_accepted':new_count,
            'total_accepted_vectors':len(supports), 'status_counts':dict(sorted(statuses.items())),
            'old_max_joint_exact_zeros_in_one_vector':max(old_zeros, default=0),
            'new_max_joint_exact_zeros_in_one_vector':p-min(supports) if supports else 0,
            'accepted_vector_exact_support_sizes':supports}
    return row


def audit(name):
    threadpool_limits(1); parent = prerequisites(); hashes = frozen(); counts = Counter(datasets=1)
    path = FOLDER/(name+'.json'); saved = json.loads(path.read_text())
    source_path = SOURCE/(name+'.json'); source = json.loads(source_path.read_text())
    assert saved['fingerprints'] == hashes and saved['complete'] and saved['fold'] == 0
    assert not saved['classification_scored'] and not saved['goal_achieved']
    assert saved['source_sha256'] == parent['artifact_hashes'][name] == digest(source_path)
    assert saved['available'] == source['strict_cell_available']
    for k in ('reference_indices', 'training_indices', 'original_support', 'original_anchor', 'd'):
        equivalent(saved[k], source[k])
    X, y = load_dataset(name); ri = np.asarray(saved['reference_indices']); ca = np.asarray(saved['training_indices'])
    A, _ = preprocess_fit(X[ri][ca], X[ri][ca])
    if saved['available']:
        check_diagnostic(A, y[ri][ca], source['diagnostic'], saved['repair'], counts)
    else:
        assert source['diagnostic'] is None and saved['repair'] is None
    print(name, 'independent original constraints/zero face/primal-dual repair audit', dict(counts), flush=True)
    return {'dataset':name, 'row':independent_statistics(saved), 'counts':dict(counts), 'sha256':digest(path)}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--dataset'); args = p.parse_args()
    if args.dataset:
        r = audit(args.dataset)
        (FOLDER/(args.dataset+'.smoke-audit.json')).write_text(json.dumps({'passed':True, **r}, indent=2), encoding='utf-8')
        return
    summary = json.loads((FOLDER/'complete.json').read_text()); reports = []; counts = Counter()
    names = sorted(prerequisites()['artifact_hashes'])
    assert [r['dataset'] for r in summary['rows']] == names
    for name in names:
        r = audit(name); reports.append(r); counts.update(r['counts'])
    rows = [r['row'] for r in reports]; equivalent(rows, summary['rows'])
    sums = {m:{k:sum(r['modes'][m][k] for r in rows) for k in summary['counts'][m]} for m in MODES}
    equivalent(sums, summary['counts'])
    hashes = {r['dataset']:r['sha256'] for r in reports}; equivalent(hashes, summary['artifact_hashes'])
    assert counts['datasets'] == summary['datasets'] == 15
    assert counts['zero_repair_training_folds'] == sum(r['available'] for r in rows)
    out = {'passed':True, 'counts':dict(counts), 'artifact_hashes':hashes,
        'summary_sha256':digest(FOLDER/'complete.json'), 'auditor_sha256':digest(Path(__file__)),
        'goal_achieved':False, 'classification_scored':False}
    (FOLDER/'complete-audit.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2), flush=True)


if __name__ == '__main__':
    main()
