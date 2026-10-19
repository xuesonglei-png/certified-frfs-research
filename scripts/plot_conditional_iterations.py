"""Scientific development tradeoff chart; expose negative intervals."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from review_box_iterations import ROOT


def main():
    payload = json.loads((ROOT/'results/conditional-iteration-review.json').read_text())
    labels = {'nested-balanced-joint': 'Joint rank + repair', 'nested-joint-backward': 'Static backward',
              'nested-conditional-backward': 'Conditional teacher scores',
              'nested-conditional-feasible': 'Conditional + restoration',
              'nested-conditional-label': 'Conditional training labels',
              'nested-crossfit-classification-rounded': 'Crossfit classification + restoration'}
    rows = payload['iterations']
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    fig.subplots_adjust(bottom=.19, right=.69, left=.13, top=.88)
    for i, row in enumerate(rows):
        x = 100*row['removed']; result = row['matched_full']; y = result['mean_pp']
        lo, hi = result['bootstrap95_pp']
        ax.errorbar(x, y, yerr=np.array([[y-lo], [hi-y]]), fmt='o', capsize=4, color=f'C{i}', label=labels[row['source']])
    ax.axhline(0, color='0.35', linewidth=1, linestyle='--')
    ax.set_xlabel('Mean raw attributes removed (%)')
    ax.set_ylabel('Balanced accuracy gap vs matched full (percentage points)')
    ax.set_title('Complete 15-dataset development results')
    ax.grid(alpha=.2)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8)
    fig.text(.5, .025, 'Dataset-paired descriptive bootstrap intervals; exposed development data, not confirmation.',
             ha='center', fontsize=8, color='0.35')
    folder = ROOT/'results/figures'; folder.mkdir(exist_ok=True)
    for suffix in ('png', 'pdf'):
        fig.savefig(folder/('conditional-iteration-tradeoff.'+suffix), dpi=180, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
