"""Failure decomposition and accuracy/compression visualization."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_decision_iteration import ROOT


def main():
    directory=ROOT/'results/predictive-iteration31'
    analysis=json.loads((directory/'expanded-baseline-analysis.json').read_text())['all15']
    methods=analysis['methods'];rows=[]
    for name,p in methods['protected']['datasets'].items():
        full=methods['all']['datasets'][name];fgs=methods['fgs']['datasets'][name]
        records=[]
        for path in sorted(directory.glob(name+'-s101-f*.json')):
            obj=json.loads(path.read_text());r=next(r for r in obj['rows'] if r['method']=='protected');c=r['certificate']
            records.append({'fold':obj['fold'],'fraction_removed':1-len(r['features'])/obj['d'],
                'zero_protected_classes':sum(v==0 for v in c['protected_counts']),
                'protected_counts':c['protected_counts'],'required':c['required'],
                'retained':c['retained_counts'],'metric':r['chosen']['family'],'recipe':r['chosen']['config']})
        rows.append({'dataset':name,'clean':p['clean'],'removed':p['removed'],
            'delta_vs_full':p['clean']-full['clean'],'delta_vs_fgs':p['clean']-fgs['clean'],
            'folds':records})
    rows.sort(key=lambda r:r['delta_vs_fgs'])
    result={'status':'failure_diagnostics_not_selection_rule','rows':rows,
        'full_feature_models':sum(r['fraction_removed']==0 for x in rows for r in x['folds']),
        'models_with_vacuous_class_retention':sum(r['zero_protected_classes']>0 for x in rows for r in x['folds'])}
    (directory/'failure-diagnostics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
    for method,data in methods.items():
        m=data['means'];axes[0].scatter(100*m['removed'],100*m['clean'],s=55)
        axes[0].annotate(method,(100*m['removed'],100*m['clean']),xytext=(4,4),textcoords='offset points')
    axes[0].set(xlabel='Mean attributes removed (%)',ylabel='Dataset-mean balanced accuracy (%)',
        title='15 development datasets: accuracy/compression')
    y=np.arange(len(rows));delta=np.array([r['delta_vs_fgs'] for r in rows])*100
    axes[1].barh(y,delta,color=np.where(delta<0,'#d1685f','#3b8d80'))
    axes[1].set_yticks(y,[r['dataset'] for r in rows]);axes[1].axvline(0,color='black',linewidth=.8)
    axes[1].set(xlabel='Certified minus FGS BA (percentage points)',title='Dataset failures and gains')
    fig.savefig(directory/'accuracy-compression-failures.png',dpi=200);plt.close(fig)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
    for r in rows:print(r['dataset'],'vs_full',round(100*r['delta_vs_full'],3),'vs_FGS',round(100*r['delta_vs_fgs'],3),'removed',round(100*r['removed'],2))

if __name__=='__main__':main()
