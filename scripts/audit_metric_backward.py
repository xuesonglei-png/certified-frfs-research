"""Posthoc fixed-metric search-space diagnostic, not a tuned comparator."""
import json,time
import numpy as np
from run_decision_iteration import ROOT,scaled_split
from run_metric_graph_iteration import scores
from certified_frfs.datasets import load_dataset
from certified_frfs import baselines
from certified_frfs.selection import select_certified
from threadpoolctl import threadpool_limits


def main():
    threadpool_limits(1);directory=ROOT/'results/predictive-iteration18';rows=[]
    for path in sorted(directory.glob('*-s*-f*.json')):
        obj=json.loads(path.read_text());X,y=load_dataset(obj['dataset'])
        tr,te=np.array(obj['train_indices']),np.array(obj['test_indices'])
        A,B=scaled_split(X[tr],X[te]);w=np.array(obj['rows'][0]['weights'])
        for order_name in ['mi','relief']:
            start=time.perf_counter()
            relevance=(baselines.mi_scores(A*w,y[tr],random_state=obj['seed']) if order_name=='mi'
                       else baselines.relieff_scores(A*w,y[tr],random_state=obj['seed']))
            order=np.argsort(relevance,kind='stable').tolist()
            selected=select_certified(A,y[tr],weights=w,order=order,eps=.01,eta=.1)
            fs=selected['features'];metric=scores(A,y[tr],B,y[te],fs,obj['seed']+3,w)
            rows.append({'dataset':obj['dataset'],'fold':obj['fold'],'order':order_name,
                         'features':fs,'removed':1-len(fs)/obj['d'],'metrics':metric,
                         'certificate':selected['certificate'],'seconds':time.perf_counter()-start})
    summary={}
    for name in ['mi','relief']:
        by_dataset={}
        for row in rows:
            if row['order']==name:by_dataset.setdefault(row['dataset'],[]).append(row)
        means={dataset:{'clean':float(np.mean([r['metrics']['0.0']['balanced_accuracy'] for r in group])),
                       'removed':float(np.mean([r['removed'] for r in group]))} for dataset,group in by_dataset.items()}
        summary[name]={'datasets':means,'means':{k:float(np.mean([r[k] for r in means.values()])) for k in ['clean','removed']}}
    (directory/'backward-diagnostic.json').write_text(json.dumps({'rows':rows,'summary':summary,
        'status':'posthoc_development_diagnostic_no_validation_selection'},indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
