"""Full independent scalar geometry/dual/paired-start and summary replay."""
import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_latent_frs_training import ROOT, SOURCE, FOLDER, frozen, prerequisite, digest
from audit_latent_frs_training_block import check_block
from audit_forest_compact_transfer import equivalent
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit


def audit(name):
    threadpool_limits(1); source_audit = prerequisite()
    saved = json.loads((FOLDER/(name+'.json')).read_text()); counts = Counter(datasets=1)
    assert saved['complete'] and not saved['classification_scored']
    for rel, sha in saved['fingerprints'].items():
        assert digest(ROOT/rel) == sha, rel
    assert digest(SOURCE/(name+'.json')) == source_audit['artifact_hashes'][name]
    source = json.loads((SOURCE/(name+'.json')).read_text())
    assert saved['reference_indices'] == source['reference_indices']
    X, y = load_dataset(name); ri = np.asarray(saved['reference_indices']); raw, labels = X[ri], y[ri]
    evaluations = []
    for fi, (b, old) in enumerate(zip(saved['inner_training'], source['inner_training'], strict=True)):
        assert b['reference_indices'] == old['reference_indices'] and 'validation_indices' not in b
        assert b['features'] == old['features']; equivalent(b['anchor'], old['anchor'])
        assert b['strict_cell_available'] == old['strict_cell_available']
        counts['available_cells'] += b['strict_cell_available']; counts['inner_training_folds'] += 1
        if b['strict_cell_available']:
            prior = old['fits'][old['training_selected_start']]
            equivalent(b['capped_endpoint'], prior['target'])
            assert b['prior_selected_start'] == old['training_selected_start']
            assert b['prior_selected_fit_optimizer_success'] == prior['fit']['optimizer_success']
            assert b['prior_fit_failures'] == sum(r['fit']['subproblem_failures'] for r in old['fits'].values())
            ca = np.asarray(b['reference_indices']); A, _ = preprocess_fit(raw[ca], raw[ca])
            evaluation = check_block(A, labels[ca], b, counts)
            evaluations.append({'fold': fi, **evaluation})
            for a in ('fixed_three', 'latent_five'):
                counts[a+'_optimizer_failures'] += evaluation['optimizer_failures'][a]
        else:
            assert not b['arms'] and b['evaluation'] is None and b['capped_endpoint'] is None
        print(name, 'training fold', fi+1, 'matched latent/fixed geometry/LP/duals replayed', flush=True)
    # Independent summary implementation; no production evaluator/aggregator.
    keys = ('selected_training_bound_decrease_vs_fixed_three_pp', 'same_weight_tightening_at_fixed_three_endpoint_pp',
        'latent_objective_improvement_beyond_fixed_three_endpoint', 'regularized_training_objective_decrease_vs_fixed_three')
    row = {'dataset': name, 'folds': evaluations,
        'mean_fold_metrics': {k: float(np.mean([r[k] for r in evaluations])) if evaluations else None for k in keys},
        'optimizer_failures': {a: int(sum(r['optimizer_failures'][a] for r in evaluations)) for a in ('fixed_three', 'latent_five')}}
    return {'dataset': name, 'counts': dict(counts), 'sha256': digest(FOLDER/(name+'.json')), 'row': row}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--dataset'); args = parser.parse_args(); frozen()
    if args.dataset:
        report = audit(args.dataset)
        (FOLDER/(args.dataset+'.smoke-audit.json')).write_text(json.dumps({'passed': True, **report}, indent=2), encoding='utf-8')
        return
    summary = json.loads((FOLDER/'complete.json').read_text()); reports = []; counts = Counter()
    with ProcessPoolExecutor(max_workers=2) as executor:
        for future in as_completed([executor.submit(audit, r['dataset']) for r in summary['rows']]):
            report = future.result(); reports.append(report); counts.update(report['counts'])
    reports.sort(key=lambda r: r['dataset']); rows = [r['row'] for r in reports]
    equivalent(rows, summary['rows'])
    assert counts['datasets'] == summary['datasets'] == 15
    assert counts['inner_training_folds'] == summary['inner_training_folds'] == 44
    assert counts['available_cells'] == summary['available_cells']
    hashes = {r['dataset']:r['sha256'] for r in reports}; equivalent(hashes, summary['artifact_hashes'])
    for arm in ('fixed_three', 'latent_five'):
        assert counts[arm+'_optimizer_failures'] == summary['optimizer_failures'][arm]
    metrics = {k: float(np.mean([r['mean_fold_metrics'][k] for r in rows if r['folds']])) for k in summary['dataset_balanced_metrics']}
    equivalent(metrics, summary['dataset_balanced_metrics'])
    out = {'passed': True, 'counts': dict(counts), 'artifact_hashes': hashes,
        'summary_sha256': digest(FOLDER/'complete.json'), 'auditor_sha256': digest(Path(__file__)),
        'classification_scored': False, 'goal_achieved': False,
        'scope': 'matched_training_scalar_geometry_LP_duals_not_directed_interval_or_test_superiority'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2), flush=True)


if __name__ == '__main__':
    main()
