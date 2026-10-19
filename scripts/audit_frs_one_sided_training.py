"""Independent same-anchor training replay; classification never scored."""
import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_frs_one_sided_training import ROOT, SOURCE, FOLDER, digest, frozen
from continue_soft_topk_knn_pilot import verify
from audit_frs_cell_fit_stored import check_cell_fit
from audit_frs_one_sided_fit_stored import check_one_sided_fit
from independent_frs_one_sided_stored import reconstruct_cell, scalar_cell_evaluation as one_sided_evaluation
from independent_frs_cell_stored_coordinates import scalar_cell_evaluation as symmetric_evaluation
from audit_forest_compact_transfer import equivalent
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit


def audit(name):
    threadpool_limits(1); saved = json.loads((FOLDER/(name+'.json')).read_text()); counts = Counter(datasets=1)
    assert saved['complete'] and not saved['classification_scored']
    for rel, sha in saved['fingerprints'].items():
        assert digest(ROOT/rel) == sha, rel
    assert verify(SOURCE)['artifact_hashes'][name] == digest(SOURCE/(name+'.json'))
    source = json.loads((SOURCE/(name+'.json')).read_text())
    assert saved['reference_indices'] == source['reference_indices']
    X, y = load_dataset(name); ri = np.asarray(saved['reference_indices']); raw, labels = X[ri], y[ri]
    for b, old in zip(saved['inner_training'], source['inner'], strict=True):
        assert b['reference_indices'] == old['reference_indices']
        assert 'validation_indices' not in b
        ca = np.asarray(b['reference_indices']); A, _ = preprocess_fit(raw[ca], raw[ca])
        base = old['options']['proposed'][source['selection']['selected']['proposed']]['metric']
        assert b['features'] == base['features']; equivalent(b['anchor'], base['weights'])
        probe = b['anchor_probe']; reconstruct_cell(A, labels[ca], probe, counts)
        assert b['strict_cell_available'] == probe['anchor_strictly_feasible']
        assert probe['full_weights'] == (np.ones(A.shape[1])/np.sqrt(A.shape[1])).tolist()
        counts['available_cells'] += b['strict_cell_available']
        if b['strict_cell_available']:
            assert set(b['fits']) == {'symmetric', 'one_sided'}
            for family, checker in [('symmetric', check_cell_fit), ('one_sided', check_one_sided_fit)]:
                record = b['fits'][family]
                assert record['features'] == b['features']; equivalent(record['anchor'], b['anchor'])
                assert record['fit']['reference_scope'] == 'original_full' and record['fit']['constrained']
                checker(A, labels[ca], record, counts)
                old_bound = symmetric_evaluation(record['fit']['class_cell'], record['target'])
                new_bound = one_sided_evaluation(record['fit']['class_cell'], record['target'])
                equivalent(old_bound, record['symmetric_bound_at_target']); equivalent(new_bound, record['one_sided_bound_at_target'])
                assert all(a <= b+1e-12 for a,b in zip(new_bound['class_upper_bounds'], old_bound['class_upper_bounds']))
        else:
            assert not b['fits']
        counts['inner_training_folds'] += 1
        print(name, 'training fold', counts['inner_training_folds'], 'independent cells/DC/duals replayed', flush=True)
    return {'dataset': name, 'sha256': digest(FOLDER/(name+'.json')), 'counts': dict(counts)}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--dataset'); args = parser.parse_args(); frozen()
    if args.dataset:
        report = audit(args.dataset)
        (FOLDER/(args.dataset+'.smoke-audit.json')).write_text(json.dumps({'passed': True, **report}, indent=2), encoding='utf-8'); return
    summary = json.loads((FOLDER/'complete.json').read_text()); counts = Counter(); reports = []
    with ProcessPoolExecutor(max_workers=2) as executor:
        for f in as_completed([executor.submit(audit, r['dataset']) for r in summary['rows']]):
            r = f.result(); reports.append(r); counts.update(r['counts'])
    assert counts['datasets'] == summary['datasets'] == 15
    assert counts['inner_training_folds'] == summary['inner_training_folds'] == 44
    assert counts['available_cells'] == summary['available_cells']
    assert counts['target_optimizer_failures'] == summary['optimizer_failures']
    hashes = {r['dataset']: r['sha256'] for r in reports}; equivalent(hashes, summary['artifact_hashes'])
    rows = []
    for sr in summary['rows']:
        name = sr['dataset']; saved = json.loads((FOLDER/(name+'.json')).read_text()); comparisons = []
        for fi, b in enumerate(saved['inner_training']):
            if not b['strict_cell_available']:
                continue
            old, new = b['fits']['symmetric'], b['fits']['one_sided']
            comparisons.append({'fold': fi, 'old_objective': old['fit']['training_objective_final'],
                'new_objective': new['fit']['training_objective_final'],
                'objective_decrease_vs_symmetric': old['fit']['training_objective_final']-new['fit']['training_objective_final'],
                'new_target_outside_symmetric_cell': bool(new['one_sided_bound_at_target']['feasible'] and not new['symmetric_bound_at_target']['feasible']),
                'failures': old['fit']['subproblem_failures']+new['fit']['subproblem_failures']})
        rows.append({'dataset': name, 'comparisons': comparisons,
            'mean_training_objective_decrease_vs_symmetric': float(np.mean([r['objective_decrease_vs_symmetric'] for r in comparisons])) if comparisons else None})
    equivalent(rows, summary['rows'])
    report = {'passed': True, 'counts': dict(counts), 'artifact_hashes': hashes,
        'summary_sha256': digest(FOLDER/'complete.json'), 'auditor_sha256': digest(Path(__file__)),
        'classification_scored': False, 'goal_achieved': False, 'scope': 'independent_training_geometry_DC_LP_bound_containment_not_prediction'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8'); print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
