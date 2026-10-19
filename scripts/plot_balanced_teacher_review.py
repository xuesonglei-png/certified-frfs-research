"""Teacher-alignment result with actual compression and matched controls."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from review_box_iterations import ROOT


def main():
    folder = ROOT/'results/nested-balanced-teacher-backward'
    report = json.loads((folder/'teacher-alignment-review.json').read_text())
    budget = json.loads((folder.parent/(folder.name+'-common-budget')/'complete.json').read_text())
    rows = report['rows']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    locations = {}
    for row in rows:
        x = 100*row['removed']
        y = 100*(row['clean']-row['full'])
        key = (round(x, 10), round(y, 10))
        locations.setdefault(key, []).append(row['dataset'])
    for (x, y), names in locations.items():
        axes[0].scatter(x, y, color='#2166ac', s=22)
        label = '\n'.join(names)
        axes[0].annotate(label, (x, y), xytext=(3, 5), textcoords='offset points', fontsize=7)
    axes[0].axhline(0, color='#777777', linestyle='--', lw=1)
    axes[0].set_xlabel('Actual attributes removed (%)')
    axes[0].set_ylabel('BA difference vs matched full (percentage points)')
    axes[0].set_title('Certified deployment, all 15 datasets', fontsize=11)
    comparisons = [report['vs_balanced_native_controls']['fgs2023_native'],
                   report['vs_balanced_native_controls']['sfrip2025_native'],
                   budget['comparisons']['fgs2023'], budget['comparisons']['sfrip2025']]
    labels = ['FGS native support', 'SFRIP native support', 'FGS same count / same classifier',
              'SFRIP same count / same classifier']
    for i, r in enumerate(comparisons):
        color = '#636363' if i < 2 else '#238b45'
        axes[1].plot(r['bootstrap95_pp'], [i, i], color=color, lw=2)
        axes[1].scatter([r['mean_pp']], [i], color=color, s=25)
    axes[1].axvline(0, color='#777777', linestyle='--', lw=1)
    axes[1].set_yticks(range(4), labels, fontsize=8)
    axes[1].set_xlabel('Proposed minus control (percentage points)')
    axes[1].set_title('Native pipelines and support ablations differ', fontsize=11)
    for axis in axes:
        axis.spines[['top', 'right']].set_visible(False)
        axis.margins(.18)
    fig.suptitle('Balanced teacher; exposed development, descriptive intervals', fontsize=11)
    fig.savefig(folder/'alignment-comparison.png', dpi=180)
    fig.savefig(folder/'alignment-comparison.pdf')
    print(folder/'alignment-comparison.png')


if __name__ == '__main__':
    main()
