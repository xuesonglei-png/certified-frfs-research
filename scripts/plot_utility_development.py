"""Scientific development plots with explicit endpoint/evidence labels."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_decision_iteration import ROOT


def main():
    directory=ROOT/'results/predictive-iteration33'
    scores=json.loads((directory/'verified-analysis.json').read_text())['all_datasets']['methods']
    query=json.loads((ROOT/'results/majority-query-audit33/analysis.json').read_text())['all15']['methods']
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.3),layout='constrained')
    for method,color in zip(scores,plt.get_cmap('tab10').colors):
        removal=100*scores[method]['means']['removed'];ba=100*scores[method]['means']['clean']
        coverage=100*query[method]['means']['certified_correct']
        axes[0].scatter(removal,ba,c=[color],s=70,label=method)
        axes[1].scatter(removal,coverage,c=[color],s=70,label=method)
    axes[0].set(xlabel='Dataset-mean attributes removed (%)',ylabel='Dataset-mean balanced accuracy (%)',
        title='Nominal predictions: repeated development')
    axes[1].set(xlabel='Dataset-mean attributes removed (%)',ylabel='Balanced certified-correct fraction (%)',
        title='Finite-query bound at radius 0.01: posthoc')
    axes[0].legend(fontsize=8,ncol=2,loc='lower left');axes[1].legend(fontsize=8,ncol=2,loc='lower left')
    fig.savefig(directory/'nominal-and-certified-tradeoffs.png',dpi=200);plt.close(fig)

if __name__=='__main__':main()
