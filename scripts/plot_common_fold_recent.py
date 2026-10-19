"""Scientific accuracy/compression tradeoffs; native stopping stays visible."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_decision_iteration import ROOT

def main():
    folder=ROOT/'results/common-fold-recent';p=json.loads((folder/'complete.json').read_text());means=p['means']
    fig,axes=plt.subplots(1,2,figsize=(10,4.3))
    for ax,metric,title in zip(axes,('clean','noise'),('Clean queries','Uniform query perturbation, radius .01')):
        for method,label,color in (('fgs2023','FGS budget adaptation','#c56b19'),('sfrip2025','SFRIP budget adaptation','#258153')):
            points=[means[method+'_'+f] for f in ('0.8','0.65','0.5')]
            ax.plot([100*r['removed'] for r in points],[100*r[metric] for r in points],'o-',color=color,label=label)
            native=means[method+'_native'];ax.scatter(100*native['removed'],100*native[metric],marker='*',s=110,color=color,label=method+' native')
        for method,label,marker,color in (('proposed','Certified policy + robust LP','D','#1768ac'),('matched_full','Matched robust full','s','#333333'),('full','Logistic full','x','#929292')):
            r=means[method];ax.scatter(100*r['removed'],100*r[metric],marker=marker,s=60,color=color,label=label)
        ax.set_title(title,fontsize=10);ax.set_xlabel('Mean attributes removed (%)');ax.set_ylabel('Dataset-mean balanced accuracy (%)');ax.grid(alpha=.2)
    axes[0].legend(fontsize=7,frameon=False,loc='lower left');fig.suptitle('Same inner folds; 15 exposed development datasets',fontsize=11)
    fig.tight_layout();fig.savefig(folder/'accuracy-compression.png',dpi=220);fig.savefig(folder/'accuracy-compression.pdf')
    print(folder/'accuracy-compression.png')

if __name__=='__main__':main()
