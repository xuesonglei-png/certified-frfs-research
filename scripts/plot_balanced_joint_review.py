"""Scientific figure separating deployment and support-only comparisons."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from run_decision_iteration import ROOT


def main():
    folder = ROOT/'results/nested-balanced-joint'
    proposed = json.loads((folder/'complete-review.json').read_text())
    aligned = json.loads((folder.parent/'nested-balanced-joint-aligned-repair/complete-review.json').read_text())
    backward = json.loads((folder.parent/'nested-joint-backward/complete-review.json').read_text())
    controls = json.loads((folder.parent/'nested-balanced-joint-robust-controls/complete.json').read_text())
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    labels = ['Balanced joint', 'Aligned repair', 'Certified backward', 'Balanced full']
    values = [proposed['mean_clean'], aligned['mean_clean'], backward['mean_clean'], proposed['mean_full']]
    colors = ['#238b45', '#756bb1', '#2166ac', '#636363']
    axes[0].barh(labels, 100*np.array(values), color=colors)
    axes[0].set_xlim(77, 81)
    axes[0].set_xlabel('Dataset-mean balanced accuracy (%)')
    axes[0].set_title('Certified deployment: all 15 datasets', fontsize=11)
    for i, v in enumerate(values):
        axes[0].text(100*v+.025, i, f'{100*v:.3f}', va='center', fontsize=9)
    panel = [('Balanced joint: unconstrained controls', proposed['common_balanced_classifier_budget_comparisons']),
             ('Balanced joint: constrained controls', controls['within_balanced'])]
    for axis, (title, comparisons) in zip(axes[1:], panel):
        for i, key in enumerate(('fgs2023', 'sfrip2025')):
            r = comparisons[key]
            lo, hi = r['bootstrap95_pp']
            axis.plot([lo, hi], [i, i], color='#238b45', lw=2)
            axis.scatter([r['mean_pp']], [i], color='#238b45', zorder=3)
        axis.axvline(0, color='#777777', lw=1, linestyle='--')
        axis.set_yticks([0, 1], ['FGS budget adaptation', 'SFRIP budget adaptation'])
        axis.set_xlabel('Proposed minus control (percentage points)')
        axis.set_title(title, fontsize=11)
        axis.set_ylim(-.5, 1.5)
    for axis in axes:
        axis.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Exposed development results; intervals are descriptive, not confirmation', fontsize=11)
    fig.savefig(folder/'comparison.png', dpi=180)
    fig.savefig(folder/'comparison.pdf')
    print(folder/'comparison.png')


if __name__ == '__main__':
    main()
