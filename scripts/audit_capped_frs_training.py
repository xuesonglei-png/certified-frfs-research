"""Independent capped DC/dual/initialization training-only cohort audit."""
import argparse
import json
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_capped_frs_training import ROOT, SOURCE, FOLDER, digest, frozen, prerequisite
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from audit_capped_frs_warm_fit import check_capped_warm_fit
from audit_forest_compact_transfer import equivalent


def audit(name):
    threadpool_limits(1); saved = json.loads((FOLDER/(name+'.json')).read_text()); prerequisite(name)
    source = json.loads((SOURCE/(name+'.json')).read_text()); counts = Counter(datasets=1)
    assert saved['complete'] and not saved['classification_scored']
    for rel, sha in saved['fingerprints'].items():
        assert digest(ROOT/rel) == sha, rel
    assert saved['reference_indices'] == source['reference_indices']
    X, y = load_dataset(name); ri = np.asarray(saved['reference_indices']); raw, labels = X[ri], y[ri]
    for b, old in zip(saved['inner_training'], source['inner_training'], strict=True):
        assert b['reference_indices'] == old['reference_indices'] and 'validation_indices' not in b
        assert b['features'] == old['features']; equivalent(b['anchor'], old['anchor'])
        assert b['strict_cell_available'] == old['strict_cell_available']; counts['available_cells'] += b['strict_cell_available']
        ca = np.asarray(b['reference_indices']); A, _ = preprocess_fit(raw[ca], raw[ca])
        if b['strict_cell_available']:
            assert set(b['fits']) == {'anchor', 'warm'}
            for family in ('anchor', 'warm'):
                r = b['fits'][family]; assert r['features'] == b['features']; equivalent(r['anchor'], b['anchor'])
                fit = r['fit']; assert fit['constrained'] and fit['reference_scope'] == 'original_full'
                assert fit['training_witnesses']['decomposition'] == 'S3_minus_max_of_S2_and_S3_minus1'
                assert fit['class_cell']['metric_envelope'] == 'Q_times_coordinate_weight_decreases_only'
                expected = b['anchor'] if family == 'anchor' else old['fits']['one_sided']['target']
                equivalent(fit['requested_initial_weights'], expected)
                assert fit['initialization_scope'] == ('anchor' if family == 'anchor' else 'training_one_sided_unbounded_endpoint')
                check_capped_warm_fit(A, labels[ca], r, counts)
            chosen = min(('anchor', 'warm'), key=lambda family: (b['fits'][family]['fit']['training_objective_final'], 0 if family == 'anchor' else 1))
            assert chosen == b['training_selected_start']; counts['warm_start_selected'] += chosen == 'warm'
        else:
            assert not b['fits'] and b['training_selected_start'] is None
        counts['inner_training_folds'] += 1
        print(name, 'training fold', counts['inner_training_folds'], 'capped faces/LP/duals/starts replayed', flush=True)
    return {'dataset': name, 'sha256': digest(FOLDER/(name+'.json')), 'counts': dict(counts)}


def independent_comparison(saved, source):
    comparisons = []
    for fi, (b, old) in enumerate(zip(saved['inner_training'], source['inner_training'], strict=True)):
        if not b['strict_cell_available']:
            continue
        fit = b['fits'][b['training_selected_start']]['fit']; prior = old['fits']['one_sided']; old_fit = prior['fit']
        reg = .01*sum(abs(w-v) for w,v in zip(prior['target'],b['anchor']))/fit['mass']
        before, after = old_fit['training_error_upper_bound_final'], fit['training_error_upper_bound_final']
        comparisons.append({'fold': fi, 'selected_start': b['training_selected_start'],
            'prior_regularized_clipped_objective': before+reg, 'new_regularized_clipped_objective': fit['training_objective_final'],
            'clipped_objective_decrease_vs_previous': before+reg-fit['training_objective_final'],
            'prior_clipped_training_bound': before, 'new_clipped_training_bound': after,
            'clipped_training_bound_decrease_pp': 100*(before-after),
            'anchor_to_start_objective_decrease': fit['training_objective_initial']-fit['optimization_start_objective'],
            'start_to_final_objective_decrease': fit['optimization_start_objective']-fit['training_objective_final'],
            'new_exact_zero_weights': int(sum(w == 0 for w in fit['weights'])),
            'failures': sum(r['fit']['subproblem_failures'] for r in b['fits'].values())})
    return {'dataset': saved['dataset'], 'comparisons': comparisons,
        'mean_clipped_training_bound_decrease_pp': float(np.mean([r['clipped_training_bound_decrease_pp'] for r in comparisons])) if comparisons else None,
        'mean_clipped_objective_decrease': float(np.mean([r['clipped_objective_decrease_vs_previous'] for r in comparisons])) if comparisons else None}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--dataset'); args = parser.parse_args(); frozen()
    if args.dataset:
        report = audit(args.dataset)
        (FOLDER/(args.dataset+'.smoke-audit.json')).write_text(json.dumps({'passed': True, **report}, indent=2), encoding='utf-8'); return
    summary = json.loads((FOLDER/'complete.json').read_text()); reports = []; counts = Counter()
    with ProcessPoolExecutor(max_workers=2) as executor:
        for f in as_completed([executor.submit(audit, r['dataset']) for r in summary['rows']]):
            report = f.result(); reports.append(report); counts.update(report['counts'])
    assert counts['datasets'] == summary['datasets'] == 15 and counts['inner_training_folds'] == summary['inner_training_folds'] == 44
    assert counts['available_cells'] == summary['available_cells'] and counts['target_optimizer_failures'] == summary['optimizer_failures']
    hashes = {r['dataset']:r['sha256'] for r in reports}; equivalent(hashes, summary['artifact_hashes']); rows = []
    for r in summary['rows']:
        name = r['dataset']; saved = json.loads((FOLDER/(name+'.json')).read_text()); source = json.loads((SOURCE/(name+'.json')).read_text())
        rows.append(independent_comparison(saved, source))
    equivalent(rows, summary['rows'])
    equivalent(float(np.mean([r['mean_clipped_training_bound_decrease_pp'] for r in rows if r['comparisons']])), summary['mean_dataset_clipped_training_bound_decrease_pp'])
    equivalent(float(np.mean([r['mean_clipped_objective_decrease'] for r in rows if r['comparisons']])), summary['mean_dataset_clipped_objective_decrease'])
    out = {'passed': True, 'counts': dict(counts), 'artifact_hashes': hashes,
        'summary_sha256': digest(FOLDER/'complete.json'), 'auditor_sha256': digest(Path(__file__)),
        'goal_achieved': False, 'classification_scored': False, 'scope': 'independent_capped_DC_training_not_prediction_superiority'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(out, indent=2), encoding='utf-8'); print(json.dumps(out, indent=2), flush=True)


if __name__ == '__main__':
    main()
