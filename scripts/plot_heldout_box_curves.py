"""Standalone scientific figure from the complete unfiltered dataset means."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_decision_iteration import ROOT

def main():
    p=json.loads((ROOT/'results/heldout-box-curves/complete.json').read_text())
    folder=ROOT/'results/heldout-box-curves';radii=np.array([0,.005,.01,.02,.05])
    labels={'nested_regularized':'Certified nested reduction','nested_full':'Matched robust full',
            'archived_full':'Logistic full','archived_fgs2023':'FGS2023 + logistic','archived_sfrip2025':'SFRIP2025 + logistic'}
    colors=['#1768ac','#333333','#929292','#c56b19','#258153']
    fig,axes=plt.subplots(1,2,figsize=(10,4.1))
    for (key,label),color in zip(labels.items(),colors):
        curve=np.array([p['means'][key][str(e)] for e in radii])
        axes[0].plot(radii,100*curve,'o-',label=label,color=color,linewidth=1.5,markersize=4)
        if key!='nested_regularized':
            proposed=np.array([p['means']['nested_regularized'][str(e)] for e in radii])
            axes[1].plot(radii,100*(proposed-curve),'o-',label=label,color=color,linewidth=1.5,markersize=4)
    axes[0].set_ylabel('Class-balanced certified test fraction (%)')
    axes[1].set_ylabel('Reduction minus comparator (percentage points)')
    axes[1].axhline(0,color='#888888',linewidth=.8,linestyle='--')
    for ax in axes:
        ax.set_xlabel('Clipped measurement-box radius');ax.grid(alpha=.2);ax.set_xticks([0,.01,.02,.05])
    axes[0].legend(fontsize=8,frameon=False,loc='lower left')
    fig.suptitle('15 exposed development datasets; dataset means, not confirmation',fontsize=11)
    fig.tight_layout();fig.savefig(folder/'heldout-box-curves.png',dpi=220);fig.savefig(folder/'heldout-box-curves.pdf')
    print(folder/'heldout-box-curves.png')

if __name__=='__main__':main()
