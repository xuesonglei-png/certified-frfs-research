"""Audit all coordinate LPs independently; no accepted witness may be forged."""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_frs_coordinate_diagnostic_v2 import ROOT, SOURCE, FOLDER, frozen, prerequisites, digest
from audit_frs_coordinate_minimum import audit_diagnostic, MODES
from audit_forest_compact_transfer import equivalent
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit


def audit(name):
    threadpool_limits(1); parent = prerequisites(); hashes = frozen(); counts = Counter(datasets=1)
    saved = json.loads((FOLDER/(name+'.json')).read_text()); source = json.loads((SOURCE/(name+'.json')).read_text())
    assert parent['artifact_hashes'][name] == digest(SOURCE/(name+'.json'))
    assert saved['fingerprints'] == hashes and saved['complete'] and not saved['classification_scored']
    assert saved['fold'] == 0; old = source['inner_training'][0]
    assert saved['reference_indices'] == source['reference_indices']
    assert saved['training_indices'] == old['reference_indices']
    assert saved['strict_cell_available'] == old['strict_cell_available']
    assert saved['original_support'] == old['features']; equivalent(saved['original_anchor'], old['anchor'])
    X, y = load_dataset(name); ri, ca = np.asarray(saved['reference_indices']), np.asarray(saved['training_indices'])
    A, _ = preprocess_fit(X[ri][ca], X[ri][ca]); labels = y[ri][ca]
    if saved['strict_cell_available']:
        d = saved['diagnostic']; assert d['geometry']['features'] == old['features']
        equivalent(d['geometry']['anchor'], old['anchor']); audit_diagnostic(A, labels, d, counts)
    else:
        assert saved['diagnostic'] is None
    regimes = {}
    if saved['diagnostic'] is not None:
        for m in MODES:
            r = saved['diagnostic']['regimes'][m]; ok = [v for v in r['coordinates'] if v['success']]
            regimes[m] = {'coordinate_LPs':len(r['coordinates']), 'failures':r['failures'],
                'accepted_individual_zero_witnesses':sum(v['accepted_individual_zero_witness'] for v in ok),
                'raw_exact_zero_candidates':sum(v['exact_zero_candidate'] for v in ok),
                'actual_coordinate_rejections':sum(not v['original_constraints_passed_at_actual_coordinates'] for v in ok),
                'numerically_positive_minimum_coordinates':sum(v['actual_minimum_weight'] > 1e-6 for v in ok),
                'actual_minima':{str(v['feature']):v['actual_minimum_weight'] for v in ok}}
    row = {'dataset':name, 'available':saved['strict_cell_available'], 'support_size':len(saved['original_support']),
        'fit_seconds':saved['fit_seconds'], 'regimes':regimes}
    print(name, 'independent coordinate LP/cell/dual diagnostic audited', flush=True)
    return {'dataset':name, 'row':row, 'counts':dict(counts), 'sha256':digest(FOLDER/(name+'.json'))}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--dataset'); args = p.parse_args()
    if args.dataset:
        r = audit(args.dataset)
        (FOLDER/(args.dataset+'.smoke-audit.json')).write_text(json.dumps({'passed':True, **r}, indent=2), encoding='utf-8'); return
    summary = json.loads((FOLDER/'complete.json').read_text()); counts = Counter(); reports = []
    for row in summary['rows']:
        r = audit(row['dataset']); reports.append(r); counts.update(r['counts'])
    rows = [r['row'] for r in reports]; equivalent(rows, summary['rows'])
    assert counts['datasets'] == summary['datasets'] == summary['training_folds'] == 15
    assert counts['diagnostic_training_folds'] == summary['available_cells']
    sums = {m:{k:sum(r['regimes'][m][k] for r in rows if r['available']) for k in summary['counts'][m]} for m in MODES}
    equivalent(sums, summary['counts'])
    hashes = {r['dataset']:r['sha256'] for r in reports}; equivalent(hashes, summary['artifact_hashes'])
    report = {'passed':True, 'counts':dict(counts), 'artifact_hashes':hashes,
        'summary_sha256':digest(FOLDER/'complete.json'), 'auditor_sha256':digest(Path(__file__)),
        'goal_achieved':False, 'classification_scored':False,
        'scope':'independent_geometry_and_per_coordinate_LP_primal_dual_diagnostic_no_prediction_or_global_reduct_necessity'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
