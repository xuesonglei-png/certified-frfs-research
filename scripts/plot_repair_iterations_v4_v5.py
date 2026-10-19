"""Scientific diagnostic plots for complete exposed development cohorts."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]

def main():
    source=ROOT/'research/repair-iterations-v4-v5-review.json'
    report=json.loads(source.read_text());assert report['complete']
    rows=report['rows'];names=[r['dataset'].replace('heart_statlog','heart') for r in rows]
    out=ROOT/'results/repair-iterations-diagnostics';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(13,4.4),layout='constrained')
    x=np.arange(len(rows));width=.32
    for offset,v,color in ((-width/2,'v4','#4477AA'),(width/2,'v5','#CC6677')):
        clean=[100*(r[v+'_clean_BA']-r['fgs_native_capped_transfer_clean_BA']) for r in rows]
        certified=[100*(r[v+'_query_certified_BA']-r['full_query_certified_BA']) for r in rows]
        removed=[100*r[v+'_removed'] for r in rows]
        for ax,vals in zip(axes,(clean,certified,removed)):
            ax.bar(x+offset,vals,width,label=v.upper(),color=color)
            ax.set_xticks(x,names,rotation=45,ha='right');ax.axhline(0,color='#666666',lw=.7)
            ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_title('Clean BA versus capped-transfer FGS');axes[0].set_ylabel('Difference (percentage points)')
    axes[1].set_title('Certified correct BA versus full uniform');axes[1].set_ylabel('Difference (percentage points)')
    axes[2].set_title('Removed attributes');axes[2].set_ylabel('Removed (%)');axes[2].legend(frameon=False)
    fig.suptitle('Six exposed development datasets; three nested outer folds each',fontsize=12)
    fig.savefig(out/'accuracy-certification-compression.png',dpi=200)
    fig.savefig(out/'accuracy-certification-compression.pdf')
    plt.close(fig)
    print(out/'accuracy-certification-compression.png')

if __name__=='__main__':main()
