"""Exploratory joint accuracy/coverage tests, with one Holm family per iteration."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]


def analyze(iteration):
    directory = ROOT/f'results/predictive-iteration{iteration}'
    summary = json.loads((directory/'summary.json').read_text())
    target = 'decision_robust' if iteration == 4 else 'decision_margin'
    rows = []
    for comparator in summary['methods']:
        if comparator == target:
            continue
        a = summary['methods'][target]['datasets']
        b = summary['methods'][comparator]['datasets']
        for metric in ['accuracy', 'noise05', 'removed', 'balanced_coverage', 'min_class_coverage']:
            names = sorted(set(a)&set(b))
            delta = np.array([a[n][metric]-b[n][metric] for n in names])
            rows.append({'comparator': comparator, 'metric': metric,
                         'mean_delta': float(delta.mean()),
                         'wins_ties_losses': [int((delta>1e-12).sum()), int((abs(delta)<=1e-12).sum()), int((delta< -1e-12).sum())],
                         'raw_p': float(wilcoxon(delta).pvalue) if np.any(delta) else 1.})
    order = sorted(range(len(rows)), key=lambda i: rows[i]['raw_p'])
    maximum = 0.
    for rank, i in enumerate(order):
        maximum = max(maximum, min(1., (len(order)-rank)*rows[i]['raw_p']))
        rows[i]['holm_p'] = maximum
    result = {'status': 'exploratory_development_posthoc_metrics_not_confirmation',
              'family': 'all_35_tests_across_five_metrics_and_seven_comparators',
              'comparisons': rows}
    (directory/'joint-metric-comparisons.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'iteration': iteration, 'adjusted_positive_results':
        [r for r in rows if r['holm_p'] < .05 and r['mean_delta'] > 0]}, indent=2))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--iteration', type=int, choices=[4, 5], default=4)
    analyze(ap.parse_args().iteration)
