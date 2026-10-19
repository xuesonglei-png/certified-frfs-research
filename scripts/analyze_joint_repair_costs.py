"""Complete-only actual support/repair cost audit; excludes cached fit timing."""
import json
import numpy as np
from run_decision_iteration import ROOT
from analyze_joint_majority_iteration42 import DATASETS


def main():
    folder = ROOT/'results/predictive-iteration42'; rows = []
    for name in DATASETS:
        for fold in range(3):
            path = folder/f'{name}-s101-f{fold}.json'
            if not path.exists(): raise RuntimeError(f'incomplete study: {path.name}')
            p = json.loads(path.read_text()); r = next(q for q in p['rows'] if q['method'] == 'certified_anchored')
            fit = r['fit']; repair = fit.get('repair', {}); raw = next(q for q in p['rows'] if q['method'] == 'sparse')
            rows.append({'dataset': name, 'fold': fold, 'd': p['d'], 'actual_features': len(fit['features']),
                'initial_fraction_config': fit['config']['fraction'], 'selected_config': fit['config'],
                'restored_budget': repair.get('restored_budget', 0), 'repair_steps': len(repair.get('trace', [])),
                'original_full_fallback': repair.get('original_full_fallback', False),
                'selected_original_full_config': fit['config']['ridge'] == 'original',
                'full_actual_support': len(fit['features']) == p['d'],
                'metric_vs_uniform_same_support_clean_gain': r['metrics']['clean']-p['uniform_on_selected_support']['metrics']['clean'],
                'certified_vs_unconstrained_clean_gain': r['metrics']['clean']-raw['metrics']['clean'],
                'repair_and_scoring_seconds': p['seconds_repair_and_scoring_only']})
    datasets = []
    for name in DATASETS:
        group = [r for r in rows if r['dataset'] == name]
        datasets.append({'dataset': name, 'removed_fraction': float(np.mean([1-r['actual_features']/r['d'] for r in group])),
            'mean_support': float(np.mean([r['actual_features'] for r in group])),
            'mean_metric_vs_uniform_support_gain': float(np.mean([r['metric_vs_uniform_same_support_clean_gain'] for r in group])),
            'full_support_folds': sum(r['full_actual_support'] for r in group),
            'mean_repair_and_scoring_seconds_excluding_original_fit': float(np.mean([r['repair_and_scoring_seconds'] for r in group]))})
    out = folder/'complete-repair-cost-analysis.json'
    if out.exists(): raise FileExistsError(out)
    payload = {'scope': 'complete45task_repair_costs_and_matched_support_control_not_acquisition_fees_or_inference_speed',
        'rows': rows, 'datasets': datasets, 'full_support_tasks': sum(r['full_actual_support'] for r in rows),
        'explicit_original_full_fallbacks': sum(r['original_full_fallback'] for r in rows),
        'selected_original_full_configs': sum(r['selected_original_full_config'] for r in rows),
        'mean_removed_fraction': float(np.mean([r['removed_fraction'] for r in datasets])),
        'mean_metric_vs_uniform_support_gain': float(np.mean([r['mean_metric_vs_uniform_support_gain'] for r in datasets])),
        'timing_limit': 'repair uses cached41training fits;elapsed seconds exclude shared metric fitting and include controls;not total training/inference time'}
    out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps({k: payload[k] for k in ['full_support_tasks', 'explicit_original_full_fallbacks', 'selected_original_full_configs', 'mean_removed_fraction', 'mean_metric_vs_uniform_support_gain']}), flush=True)


if __name__ == '__main__': main()
