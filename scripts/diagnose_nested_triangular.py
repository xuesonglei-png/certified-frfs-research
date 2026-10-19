"""Describe every dataset, failure and paired reduction effect without selection."""
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'results/nested-triangular'


def diagnose():
    summary = json.loads((FOLDER / 'complete.json').read_text())
    rows = []
    failures = Counter()
    for row in summary['rows']:
        source = json.loads((FOLDER / (row['dataset'] + '.json')).read_text())
        splits = source['splits']
        counts = Counter()
        for split in splits:
            stages = [entry['stage'] for inner in split['inner']
                      for entry in inner['by_C'].values()]
            stages += list(split['outer_stages'].values())
            for stage in stages:
                seen = set()
                for record in stage['policies'].values():
                    failure = record['failure']
                    if failure is None:
                        continue
                    # Multiple fractions can map to the same actual candidate.
                    key = json.dumps(failure, sort_keys=True)
                    if key in seen:
                        continue
                    seen.add(key)
                    fit = failure.get('requested_fit', failure)
                    counts[str(fit.get('status', 'unknown'))] += 1
        failures.update(counts)
        rows.append({
            'dataset': row['dataset'], 'balanced_accuracy': row['clean'],
            'matched_full_balanced_accuracy': row['full'],
            'paired_gain_pp': 100 * (row['clean'] - row['full']),
            'removed_fraction': row['removed'],
            'noise_gain_pp': 100 * (row['noise'] - row['full_noise']),
            'selected_full_count': sum(s['metrics']['retained'] == 1 for s in splits),
            'selected_fallback_count': row['selected_fallbacks'],
            'split_count': len(splits), 'failure_status_counts': dict(counts),
            'training_selection_seconds': row['seconds'],
        })
    report = {
        'rows': rows, 'failure_status_counts': dict(failures),
        'selected_full_count': sum(r['selected_full_count'] for r in rows),
        'mean_noise_gain_pp': float(np.mean([r['noise_gain_pp'] for r in rows])),
        'scope': 'all15_all44_exposed_development_descriptive_not_confirmation',
        'interpretation': 'Full supports and fallbacks count toward actual compression. '
                          'Training and selection seconds are not inference timing. '
                          'Solver status 1 is a limit, not evidence of infeasibility.',
    }
    (FOLDER / 'diagnostics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True)
    ids = np.arange(len(rows))
    axes[0].barh(ids, [100*r['removed_fraction'] for r in rows], color='#168477')
    gains = [r['paired_gain_pp'] for r in rows]
    axes[1].barh(ids, gains, color=['#168477' if g >= 0 else '#c34a4a' for g in gains])
    axes[1].axvline(0, color='black', linewidth=.8)
    axes[0].set_yticks(ids, [r['dataset'] for r in rows])
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Removed raw attributes (%)')
    axes[1].set_xlabel('Balanced accuracy change vs matched full (pp)')
    fig.suptitle('All 15 development datasets: actual reduction and prediction trade-off')
    fig.tight_layout()
    fig.savefig(FOLDER / 'reduction-prediction.png', dpi=160)
    plt.close(fig)
    print(json.dumps({k:v for k,v in report.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', default='nested-triangular')
    FOLDER = ROOT / 'results' / parser.parse_args().folder
    diagnose()
