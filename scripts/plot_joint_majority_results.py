"""Scientific figures from complete audited development results only."""
import json, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_decision_iteration import ROOT


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('iteration', choices=['41', '42']); args = parser.parse_args()
    folder = ROOT/f'results/predictive-iteration{args.iteration}'
    p = json.loads((folder/'complete-development-analysis.json').read_text()); rows = p['rows']
    primary = 'sparse' if args.iteration == '41' else 'certified_anchored'
    names = [r['dataset'].replace('_', ' ') for r in rows]; y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(12, 6.4), gridspec_kw={'width_ratios': [1.4, 1.1, 1.1]})
    delta = np.array([100*(r['metrics'][primary]['clean']-r['metrics']['fgs']['clean']) for r in rows])
    axes[0].barh(y, delta, color=np.where(delta >= 0, '#267b70', '#bd4b4b'), height=.65)
    axes[0].axvline(0, color='#555555', lw=.8); axes[0].set_yticks(y, names); axes[0].invert_yaxis()
    axes[0].set_xlabel('Clean balanced accuracy difference (pp)'); axes[0].set_title('A. Method minus FGS')
    removed = [100*r['removed_fraction'][primary] for r in rows]
    axes[1].barh(y, removed, color='#4876a4', height=.65); axes[1].set_yticks(y, []); axes[1].invert_yaxis(); axes[1].set_xlim(0, 100)
    axes[1].set_xlabel('Attributes removed (%)'); axes[1].set_title('B. Actual retained-support cost')
    tests = p['primary_tests']; effect = [100*r['mean_gain'] for r in tests]
    ci = np.array([r['bootstrap_mean_gain_interval95'] for r in tests])*100
    axes[2].errorbar(effect, np.arange(3), xerr=np.array([np.array(effect)-ci[:, 0], ci[:, 1]-np.array(effect)]),
        fmt='o', color='#267b70', capsize=4)
    axes[2].axvline(0, color='#555555', lw=.8); axes[2].set_yticks(range(3), [r['baseline'] for r in tests]); axes[2].invert_yaxis()
    axes[2].set_xlabel('Paired dataset mean gain (pp)'); axes[2].set_title('C. Bootstrap 95% intervals')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='x', alpha=.15); ax.set_axisbelow(True)
    fig.suptitle('15 datasets, 3 outer folds: repeated development evaluation', fontsize=12)
    fig.text(.5, .01, 'Dataset means are the analysis units. These are not untouched confirmation results.', ha='center', fontsize=8)
    fig.tight_layout(rect=[0, .035, 1, .95]); output = ROOT/'output'/f'iteration{args.iteration}-paired-development'
    fig.savefig(output.with_suffix('.png'), dpi=220); fig.savefig(output.with_suffix('.pdf')); plt.close(fig)
    print(output.with_suffix('.png'), flush=True)


if __name__ == '__main__': main()
