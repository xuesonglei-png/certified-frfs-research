"""Full cohort independent common-budget optimizer/LP audit."""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_sparse_global_frs_training import (ROOT, CAPPED, DIAGNOSTIC, FOLDER, ARMS,
    frozen, prerequisites, digest)
from audit_sparse_global_training_block import audit_block
from audit_forest_compact_transfer import equivalent
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit


def audit(name):
    threadpool_limits(1); prerequisites(); hashes = frozen(); counts = Counter(datasets=1)
    path = FOLDER/(name+'.json'); saved = json.loads(path.read_text())
    source = json.loads((CAPPED/(name+'.json')).read_text()); old = source['inner_training'][0]
    diag = json.loads((DIAGNOSTIC/(name+'.json')).read_text())
    assert saved['fingerprints'] == hashes and saved['complete'] and not saved['classification_scored']
    assert saved['reference_indices'] == source['reference_indices'] == diag['reference_indices']
    assert saved['training_indices'] == old['reference_indices'] == diag['training_indices']
    assert saved['fold'] == 0 and saved['strict_cell_available'] == old['strict_cell_available']
    if saved['strict_cell_available']:
        b = saved['block']; assert b['features'] == old['features']; equivalent(b['anchor'], old['anchor'])
        equivalent(b['capped'], old['fits'][old['training_selected_start']]['target'])
        X, y = load_dataset(name); ri, ca = np.asarray(saved['reference_indices']), np.asarray(saved['training_indices'])
        A, _ = preprocess_fit(X[ri][ca], X[ri][ca]); audit_block(A, y[ri][ca], b, diag['diagnostic'], counts)
    else:
        assert saved['block'] is None
    row = {'dataset':name, 'available_original_cell':saved['strict_cell_available'],
        'evaluation':None if saved['block'] is None else saved['block']['evaluation'],
        'common_zero_feature':None if saved['block'] is None else saved['block']['common_zero_feature']}
    print(name, 'three arms/common starts/loss/sparsity/duals independently audited', flush=True)
    return {'dataset':name, 'row':row, 'counts':dict(counts), 'sha256':digest(path)}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--dataset'); args = p.parse_args()
    if args.dataset:
        r = audit(args.dataset)
        (FOLDER/(args.dataset+'.smoke-audit.json')).write_text(json.dumps({'passed':True, **r}, indent=2), encoding='utf-8'); return
    summary = json.loads((FOLDER/'complete.json').read_text()); reports = []; counts = Counter()
    for row in summary['rows']:
        r = audit(row['dataset']); reports.append(r); counts.update(r['counts'])
    rows = [r['row'] for r in reports]; equivalent(rows, summary['rows'])
    assert counts['datasets'] == summary['datasets'] == summary['training_folds'] == 15
    available = [r for r in rows if r['evaluation'] is not None]
    metrics = {k:float(np.mean([r['evaluation'][k] for r in available])) for k in summary['dataset_balanced_metrics']}
    failures = {a:sum(r['evaluation']['optimizer_failures'][a] for r in available) for a in ARMS}
    regions = {a:sum(r['evaluation']['available'][a] for r in available) for a in ARMS}
    zeros = {a:sum(r['evaluation']['exact_zero_weights'][a] > 0 for r in available) for a in ARMS}
    equivalent(metrics, summary['dataset_balanced_metrics']); equivalent(failures, summary['optimizer_failures'])
    equivalent(regions, summary['available_regions']); equivalent(zeros, summary['folds_with_exact_zero_weights'])
    hashes = {r['dataset']:r['sha256'] for r in reports}; equivalent(hashes, summary['artifact_hashes'])
    report = {'passed':True, 'counts':dict(counts), 'artifact_hashes':hashes,
        'summary_sha256':digest(FOLDER/'complete.json'), 'auditor_sha256':digest(Path(__file__)),
        'goal_achieved':False, 'classification_scored':False,
        'scope':'independent_matched_starts_global_and_fixed_signed_cells_sparse_tangents_LPduals_training_only'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
