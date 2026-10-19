"""Certification/compression comparison using ONE shared fitted geometry.

Use the proposed training-selected final metric for all same-size selectors.
This prevents comparing coverage in different metrics. No test labels used.
"""
import json
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs import baselines
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.majority_metric import MajorityTerms
from certified_frfs.adaptive_witness import adaptive_bounds
from certified_frfs.flexible_witness import flexible_bounds


def main():
    threadpool_limits(1);rows=[]
    functions={'mi':baselines.select_mi,'relief':baselines.select_relieff,'mrmr':baselines.select_mrmr}
    for source in sorted((ROOT/'results/predictive-iteration31').glob('*-s101-f*.json')):
        obj=json.loads(source.read_text());X,y=load_dataset(obj['dataset']);tr=np.array(obj['train_indices']);y=y[tr]
        A,_=preprocess_fit(X[tr],X[tr]);r=next(r for r in obj['rows'] if r['method']=='protected')
        fs=r['features'];w=np.array(r['weights']);k=len(fs);ws=MajorityTerms(A,y,weights=w)
        full=ws.bounds(range(obj['d']),allowed_enemies=2)>1e-10;groups=[(y==label)&full for label in np.unique(y)]
        selections={'protected':fs,**{method:f(A*w,y,k=k) for method,f in functions.items()}}
        for method,selected in selections.items():
            bounds=np.maximum(ws.bounds(selected,allowed_enemies=2),adaptive_bounds(A,y,selected,w))
            counts=[int(((bounds>1e-10)&g).sum()) for g in groups];total=[int(g.sum()) for g in groups]
            nonempty=[c/t for c,t in zip(counts,total) if t>0]
            rows.append({'dataset':obj['dataset'],'fold':obj['fold'],'method':method,'features':selected,
                'shared_metric_source':'proposed_training_selected_final_metric','protected_counts':total,
                'retained_counts':counts,'retention_on_nonempty_classes':nonempty,
                'minimum_nonvacuous_class_retention':min(nonempty) if nonempty else None,
                'certified_fraction':float(np.mean(bounds>1e-10)),
                'meets_90_percent_each_protected_class':all(c>=np.ceil(.9*t-1e-12) for c,t in zip(counts,total))})
        print(obj['dataset'],obj['fold'],'shared metric audit complete',flush=True)
    summary={}
    for method in ('protected',*functions):
        subset=[r for r in rows if r['method']==method]
        summary[method]={'models':len(subset),'meeting_retention':sum(r['meets_90_percent_each_protected_class'] for r in subset),
            'models_with_no_protected_objects':sum(not r['retention_on_nonempty_classes'] for r in subset)}
    directory=ROOT/'results/shared-metric-certification';directory.mkdir(parents=True,exist_ok=True)
    (directory/'audit.json').write_text(json.dumps({'status':'training_table_diagnostic_not_generalization',
        'certificate':'fixed_or_refreshed_three_friend_witnesses','rows':rows,'summary':summary},indent=2),encoding='utf-8')
    print(json.dumps(summary))

if __name__=='__main__':main()
