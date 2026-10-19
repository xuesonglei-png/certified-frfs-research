"""Scientific development diagnostics, never a submission-success figure."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def plot(joint,risk,cost):
    folder = ROOT/'results/figures/method-development'; folder.mkdir(parents=True,exist_ok=True)
    panels = [('Nested path selection',[
        ('vs original path',joint['contrasts']['clean']['original']),
        ('vs full features',joint['contrasts']['clean']['full']),
        ('vs native FGS',joint['contrasts']['clean']['fgs2023']),
        ('vs native SFRIP',joint['contrasts']['clean']['sfrip2025'])]),
        ('Criteria at identical counts',[
        ('Actual refit ranking',risk['contrasts']['clean']['refitted']),
        ('Dependency-only guard',cost['contrasts']['clean']['dependency_only'])])]
    fig,axes = plt.subplots(1,2,figsize=(11,4.3),gridspec_kw={'width_ratios':[1.15,1]})
    for ax,(title,records) in zip(axes,panels):
        values = np.array([r['mean_pp'] for _,r in records]); intervals = np.array([r['bootstrap95_pp'] for _,r in records]); y = np.arange(len(records))
        ax.axvline(0,color='.55',linewidth=1,linestyle='--')
        ax.errorbar(values,y,xerr=np.vstack([values-intervals[:,0],intervals[:,1]-values]),fmt='o',capsize=4,color='#28658e',markersize=6)
        ax.set_yticks(y,labels=[name for name,_ in records]); ax.invert_yaxis(); ax.set_title(title,pad=14)
        ax.set_xlabel('Balanced-accuracy difference (percentage points)')
        ax.grid(axis='x',alpha=.18); ax.spines[['top','right']].set_visible(False)
        for yi,(name,r) in zip(y,records):
            ax.annotate(f"p={r['sign_flip_two_sided_p']:.3f}",(r['bootstrap95_pp'][1],yi),xytext=(7,0),textcoords='offset points',va='center',fontsize=9)
        left,right = ax.get_xlim(); ax.set_xlim(left,right+.3*(right-left))
    fig.suptitle('Audited development results: no general predictive superiority',fontsize=13)
    fig.text(.5,.015,'15 exposed development datasets; dataset-equal means; descriptive bootstrap 95% intervals.\nNative methods use different counts; the right panel compares identical counts. These are not confirmatory tests.',ha='center',fontsize=9,color='.35')
    fig.tight_layout(rect=[0,.1,1,.94]); fig.savefig(folder/'accuracy-criteria.png',dpi=180); fig.savefig(folder/'accuracy-criteria.svg'); plt.close(fig)
    (folder/'accuracy-criteria-data.json').write_text(json.dumps({'panels':panels,
        'caption':'Descriptive exposed-development analysis only; no general superiority or equivalence conclusion.'},indent=2),encoding='utf-8')


if __name__ == '__main__':
    plot(*[json.loads((ROOT/'results'/folder/'complete.json').read_text()) for folder in ('nested-joint-path','refitted-risk-geometry-pilot','dependency-only-guard-cost')])
