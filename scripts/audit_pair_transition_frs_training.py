"""Independent replay of geometry, strict-solver controls and all rows."""
import argparse
import json
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_pair_transition_frs_training import ROOT, SOURCE, FOLDER, frozen, prerequisite, digest
from audit_pair_transition_frs_fit import check_pair_transition_fit
from audit_frs_one_sided_fit_stored import check_one_sided_fit
from independent_frs_pair_transition_cell import reconstruct_cell, scalar_cell_evaluation as pair_evaluation
from independent_frs_one_sided_stored import scalar_cell_evaluation as monotone_evaluation
from audit_forest_compact_transfer import equivalent
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit


def audit(name):
    threadpool_limits(1); source_audit = prerequisite(); counts = Counter(datasets=1)
    saved = json.loads((FOLDER/(name+'.json')).read_text())
    assert saved['complete'] and not saved['classification_scored']
    for rel, sha in saved['fingerprints'].items():
        assert digest(ROOT/rel) == sha, rel
    assert digest(SOURCE/(name+'.json')) == source_audit['artifact_hashes'][name]
    source = json.loads((SOURCE/(name+'.json')).read_text()); assert saved['reference_indices'] == source['reference_indices']
    X, y = load_dataset(name); ri = np.asarray(saved['reference_indices']); raw, labels = X[ri], y[ri]; comparisons = []
    for fi, (b, old) in enumerate(zip(saved['inner_training'], source['inner_training'], strict=True)):
        assert b['reference_indices'] == old['reference_indices'] and 'validation_indices' not in b
        assert b['features'] == old['features']; equivalent(b['anchor'], old['anchor'])
        assert b['strict_cell_available'] == old['strict_cell_available']
        ca = np.asarray(b['reference_indices']); A, _ = preprocess_fit(raw[ca], raw[ca])
        reconstruct_cell(A, labels[ca], b['anchor_pair_probe'], counts)
        if b['strict_cell_available']:
            assert set(b['fits']) == {'one_sided', 'pair_transition'}
            for arm, checker in (('one_sided', check_one_sided_fit), ('pair_transition', check_pair_transition_fit)):
                r = b['fits'][arm]; assert r['features'] == b['features']; equivalent(r['anchor'], b['anchor'])
                fit = r['fit']; assert fit['constrained'] and fit['reference_scope'] == 'original_full'
                assert fit['solver_primal_feasibility_tolerance'] == fit['solver_dual_feasibility_tolerance'] == 1e-9
                checker(A, labels[ca], r, counts)
                equivalent(monotone_evaluation(b['anchor_pair_probe'], r['target']), r['one_sided_bound_at_target'])
                equivalent(pair_evaluation(b['anchor_pair_probe'], r['target']), r['pair_transition_bound_at_target'])
                assert all(z <= a+1e-12 for z, a in zip(r['pair_transition_bound_at_target']['raw_upper_bounds_with_inflation'],
                    r['one_sided_bound_at_target']['raw_upper_bounds_with_inflation'], strict=True))
                counts[arm+'_optimizer_failures'] += fit['subproblem_failures']
            a, z = b['fits']['one_sided'], b['fits']['pair_transition']
            class_n = [c['count'] for c in b['anchor_pair_probe']['classes']]
            old_b = a['one_sided_bound_at_target']['raw_upper_bounds_with_inflation']; new_b = a['pair_transition_bound_at_target']['raw_upper_bounds_with_inflation']
            comparisons.append({'fold': fi, 'objective_decrease': a['fit']['training_objective_final']-z['fit']['training_objective_final'],
                'clipped_training_bound_decrease_pp': 100*(a['fit']['training_error_upper_bound_final']-z['fit']['training_error_upper_bound_final']),
                'pair_bound_tightening_at_old_target_pp': 100*sum(n*(u-v) for n,u,v in zip(class_n,old_b,new_b,strict=True))/sum(class_n),
                'new_target_outside_one_sided_cell': bool(z['pair_transition_bound_at_target']['feasible'] and not z['one_sided_bound_at_target']['feasible']),
                'old_zero_weights': int(sum(w == 0 for w in a['target'])), 'new_zero_weights': int(sum(w == 0 for w in z['target'])),
                'fit_failures': {arm: int(b['fits'][arm]['fit']['subproblem_failures']) for arm in ('one_sided','pair_transition')},
                'fit_seconds': {arm: b['fits'][arm]['fit_seconds'] for arm in ('one_sided','pair_transition')},
                'pair_enemy_rows': int(sum(len(r['enemies']) for r in b['anchor_pair_probe']['pair_transition_objects']))})
        else:
            assert not b['fits']
        counts['inner_training_folds'] += 1; counts['available_cells'] += b['strict_cell_available']
        print(name, 'training fold', fi+1, 'paired residual cells/DC/duals replayed', flush=True)
    keys = ('objective_decrease', 'clipped_training_bound_decrease_pp', 'pair_bound_tightening_at_old_target_pp')
    row = {'dataset': name, 'comparisons': comparisons,
        'mean_fold_metrics': {k: float(np.mean([r[k] for r in comparisons])) if comparisons else None for k in keys}}
    return {'dataset': name, 'counts': dict(counts), 'row': row, 'sha256': digest(FOLDER/(name+'.json'))}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--dataset'); args = parser.parse_args(); frozen()
    if args.dataset:
        report = audit(args.dataset)
        (FOLDER/(args.dataset+'.smoke-audit.json')).write_text(json.dumps({'passed': True, **report}, indent=2), encoding='utf-8'); return
    summary = json.loads((FOLDER/'complete.json').read_text()); reports = []; counts = Counter()
    with ProcessPoolExecutor(max_workers=2) as executor:
        for future in as_completed([executor.submit(audit, r['dataset']) for r in summary['rows']]):
            report = future.result(); reports.append(report); counts.update(report['counts'])
    reports.sort(key=lambda r:r['dataset']); rows = [r['row'] for r in reports]; equivalent(rows, summary['rows'])
    hashes = {r['dataset']:r['sha256'] for r in reports}; equivalent(hashes, summary['artifact_hashes'])
    assert counts['datasets'] == summary['datasets'] == 15 and counts['inner_training_folds'] == summary['inner_training_folds'] == 44
    assert counts['available_cells'] == summary['available_cells']
    for arm in ('one_sided', 'pair_transition'):
        assert counts[arm+'_optimizer_failures'] == summary['optimizer_failures'][arm]
    metrics = {k: float(np.mean([r['mean_fold_metrics'][k] for r in rows if r['comparisons']])) for k in summary['dataset_balanced_metrics']}
    equivalent(metrics, summary['dataset_balanced_metrics'])
    report = {'passed': True, 'counts': dict(counts), 'artifact_hashes': hashes, 'summary_sha256': digest(FOLDER/'complete.json'),
        'auditor_sha256': digest(Path(__file__)), 'goal_achieved': False, 'classification_scored': False,
        'scope': 'independent_residual_transition_training_geometry_DC_LP_not_prediction_or_originality'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8'); print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
