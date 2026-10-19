"""Retain all development outcomes and paired dataset-level comparisons."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]


def summarize(directory):
    files = sorted(directory.glob('*-s*-f*.json'))
    if any(json.loads(path.read_text()).get('status','').startswith('invalid_') for path in files):
        print(json.dumps({'iteration':directory.name,'status':'invalid_run_excluded'}))
        return
    grouped = {}
    for path in files:
        obj = json.loads(path.read_text())
        for row in obj['rows']:
            grouped.setdefault(row['method'], {}).setdefault(obj['dataset'], []).append({
                'accuracy': row['metrics']['0.0']['balanced_accuracy'],
                'noise05': row['metrics']['0.05']['balanced_accuracy'],
                'removed': 1-len(row['features'])/obj['d'],
                **({'upper': row['certificate_upper']} if 'certificate_upper' in row else
                   {'balanced_coverage': row['coverage']['balanced_coverage'] if row['coverage'] else 0.,
                    'min_class_coverage': row['coverage']['min_class_coverage'] if row['coverage'] else 0.})})
    summary = {'status': 'development_only', 'completed_splits': len(files),
               'methods': {}, 'pairwise': []}
    for method, datasets in grouped.items():
        means = {name: {key: float(np.mean([r[key] for r in rows]))
                        for key in rows[0]} for name, rows in datasets.items()}
        summary['methods'][method] = {
            'n_datasets': len(datasets), 'datasets': means,
            'means': {key: float(np.mean([m[key] for m in means.values()]))
                      for key in next(iter(means.values()))}}
    for target in ['predictive_certified', 'ranking_certified_repair', 'certified_metric', 'decision_robust', 'decision_margin', 'decision_robust_local', 'decision_robust_stable', 'decision_query_aware', 'decision_robust_balanced_query', 'decision_robust_crossfit', 'decision_robust_constrained', 'decision_robust_screen', 'decision_robust_knn_majority']:
        if target not in grouped:
            continue
        for comparator in grouped:
            if comparator == target:
                continue
            common = sorted(set(grouped[target]) & set(grouped[comparator]))
            delta = np.array([summary['methods'][target]['datasets'][n]['accuracy']
                              - summary['methods'][comparator]['datasets'][n]['accuracy']
                              for n in common])
            summary['pairwise'].append({
                'target': target, 'comparator': comparator, 'datasets': common,
                'mean_delta': float(delta.mean()),
                'wins_ties_losses': [int((delta>1e-12).sum()), int((abs(delta)<=1e-12).sum()),
                                    int((delta < -1e-12).sum())],
                'raw_wilcoxon': float(wilcoxon(delta).pvalue) if np.any(delta) else 1.})
    comparisons = summary['pairwise']
    order = sorted(range(len(comparisons)), key=lambda i: comparisons[i]['raw_wilcoxon'])
    maximum = 0.
    for rank, i in enumerate(order):
        maximum = max(maximum, min(1., (len(order)-rank)*comparisons[i]['raw_wilcoxon']))
        comparisons[i]['holm_p'] = maximum
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({'iteration': directory.name, 'splits': len(files),
          'means': {m: v['means'] for m,v in summary['methods'].items()},
          'pairwise': comparisons}, indent=2))


if __name__ == '__main__':
    for directory in sorted((ROOT / 'results').glob('predictive-iteration*')):
        summarize(directory)
