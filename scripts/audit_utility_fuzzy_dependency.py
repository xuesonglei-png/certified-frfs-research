"""Original fuzzy-dependency certificates for new utility-selected subsets.

Post-fit training diagnostic only. A majority-retention guarantee is not
silently relabeled as a fuzzy-dependency-loss guarantee.
"""
import json
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.core import certificate,gamma


def main():
    threadpool_limits(1);directory=ROOT/'results/predictive-iteration33';rows=[]
    for file in sorted(directory.glob('*-s101-f*.json')):
        obj=json.loads(file.read_text());X,y=load_dataset(obj['dataset']);tr=np.array(obj['train_indices']);A,_=preprocess_fit(X[tr],X[tr])
        for r in obj['rows']:
            if r['method'] not in ('protected','utility'):continue
            w=np.array(r['weights']);fs=r['features'];result=certificate(A,y[tr],fs,weights=w,eps=.01,method='hybrid')
            rows.append({'dataset':obj['dataset'],'fold':obj['fold'],'method':r['method'],
                'full_dependency':gamma(A,y[tr],weights=w),'selected_dependency':gamma(A,y[tr],fs,weights=w),
                'removed':1-len(fs)/obj['d'],'certificate':result})
        print('dependency checked',obj['dataset'],obj['fold'],flush=True)
    result={'status':'post_fit_original_fuzzy_dependency_diagnostic_not_search_constraint',
        'scope':'finite_training_table_fixed_weights_shared_box_radius01_not_unseen_classification',
        'models':len(rows),'rows':rows}
    (directory/'original-dependency-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'models':len(rows),'means':{method:{key:float(np.mean([
        r['certificate'][key] for r in rows if r['method']==method])) for key in ('nominal_loss','upper_bound','stable_fraction')}
        for method in ('protected','utility')}}))

if __name__=='__main__':main()
