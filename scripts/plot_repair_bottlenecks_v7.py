"""Static scientific mechanism diagnosis; all six development datasets shown."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    diagnosis = json.loads((ROOT/'research/repair-bottlenecks-inner-v7.json').read_text())
    nominal = json.loads((ROOT/'results/nominal-dependency-headroom-v7/summary.json').read_text())
    assert diagnosis['complete'] and nominal['complete'] and diagnosis['datasets'] == nominal['datasets'] == 6
    rows = diagnosis['rows']; x = np.arange(6)
    labels = [r['dataset'].replace('heart_statlog', 'heart') for r in rows]
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.1), layout='constrained')
    values = [[], [], [], []]
    for row in rows:
        cnt = nominal['dataset_counts'][row['dataset']]
        admitted = sum(c.get('feasible_candidates', 0) for c in row['family_training_counts'].values())
        majority_only = sum(c.get('majority_only_block', 0) for c in row['family_training_counts'].values())
        parts = [admitted, majority_only, cnt['nominal_violation_witnesses'],
                 cnt['dependency_blocked_without_nominal_violation']]
        assert sum(parts) == cnt['unique_candidates']
        for arr, v in zip(values, parts, strict=True):
            arr.append(100*v/cnt['unique_candidates'])
    base = np.zeros(6)
    for vals, label, color in zip(values,
            ('Already certified', 'Majority-only rejection', 'Nominal dependency violation',
             'Dependency rejection; nominal inconclusive'),
            ('#228833', '#EECC66', '#CC6677', '#4477AA'), strict=True):
        axes[0].bar(x, vals, bottom=base, color=color, label=label)
        base += vals
    axes[0].set_ylabel('Candidate models (%)')
    axes[0].set_ylim(0, 100)
    axes[0].set_title('What blocks certification?')
    axes[0].legend(frameon=False, fontsize=8, loc='upper left', bbox_to_anchor=(0, -.24))
    w = .35
    for off, method, title, color in (
            (-w/2, 'fgs_native_capped_transfer', 'Capped-transfer FGS', '#AA4499'),
            (w/2, 'fgs_native_local_soft', 'Local-soft FGS adaptation', '#66CCEE')):
        rr = {r['dataset']: r for r in diagnosis['contrasts'][method]['dataset_rows']}
        vals = [rr[r['dataset']]['best_existing_pool_minus_native_pp'] for r in rows]
        axes[1].bar(x+off, vals, width=w, color=color, label=title)
    axes[1].axhline(0, color='#444444', lw=.8)
    axes[1].set_title('Best existing pool versus tuned native FGS')
    axes[1].set_ylabel('Inner balanced accuracy difference (pp)')
    axes[1].legend(frameon=False, fontsize=8, loc='upper left', bbox_to_anchor=(0, -.24))
    for ax in axes:
        ax.set_xticks(x, labels, rotation=35, ha='right')
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    fig.suptitle('Exposed training-side diagnosis; selected inner scores are optimistic', fontsize=12)
    folder = ROOT/'results/repair-iterations-diagnostics'; folder.mkdir(exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(folder/('certification-bottlenecks-v7.'+ext), dpi=200)
    plt.close(fig)
    print(folder/'certification-bottlenecks-v7.png')


if __name__ == '__main__':
    main()
