"""Scientific summary figure with development/assessment boundaries visible."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_decision_iteration import ROOT


def main():
    development=json.loads((ROOT/'results/long-capped-sparse/complete.json').read_text())
    fresh=json.loads((ROOT/'results/newdata-compact/complete.json').read_text())
    colors={'proposed':'#16847d','fgs2023':'#63707d','sfrip2025':'#db765d'}
    labels={'proposed':'Certified sparse','fgs2023':'FGS 2023 (independent)','sfrip2025':'SFRIP 2025 (independent)'}
    fig,axes=plt.subplots(1,3,figsize=(16,6))
    for ax,rows,proposed,title in [(axes[0],development['rows'],'utility','15 exposed tasks: ten-step development'),
        (axes[1],fresh['rows'],None,'8 additional tasks: frozen three-step assessment')]:
        x=np.arange(len(rows)); width=.25
        for j,m in enumerate(colors):
            values=[r['utility'] if proposed and m=='proposed' else (r['control'][m] if proposed else r['utility'][m]) for r in rows]
            ax.bar(x+(j-1)*width,np.array(values)*100,width,color=colors[m],label=labels[m])
        ax.set_xticks(x,[r['dataset'] for r in rows],rotation=75,ha='right',fontsize=8)
        ax.set_ylabel('Clean/joint-noise balanced accuracy mean (%)')
        ax.set_ylim(0,105);ax.set_title(title,fontsize=10);ax.grid(axis='y',alpha=.2)
    for m,color in colors.items():
        axes[2].scatter(fresh['removed'][m]*100,fresh['means'][m]*100,s=80,color=color)
        axes[2].annotate(labels[m],(fresh['removed'][m]*100,fresh['means'][m]*100),xytext=(0,10),textcoords='offset points',ha='center',fontsize=8)
    axes[2].set(xlim=(-5,100),ylim=(68,82),xlabel='Mean features removed (%)',ylabel='Mean utility (%)')
    axes[2].set_title('Frozen assessment: accuracy/compression tradeoff',fontsize=10);axes[2].grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.93),ncol=3,frameon=False)
    fig.suptitle('Evidence summary: different versions; superiority remains unconfirmed',fontsize=12,y=.99)
    fig.subplots_adjust(left=.055,right=.985,bottom=.28,top=.82,wspace=.25)
    target=ROOT/'output/current-method-evidence.png';fig.savefig(target,dpi=180);plt.close(fig)
    print(target,flush=True)


if __name__=='__main__':main()
