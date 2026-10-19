"""Posthoc restoration of baselines under the same dependency tolerance."""
import json
import numpy as np
from run_decision_iteration import ROOT
from run_metric_graph_iteration import scores
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.core import certificate
from threadpoolctl import threadpool_limits


def main():
    threadpool_limits(1);directory=ROOT/'results/predictive-iteration20';rows=[]
    for path in sorted(directory.glob('*-s*-f*.json')):
        obj=json.loads(path.read_text());X,y=load_dataset(obj['dataset'])
        tr,te=np.array(obj['train_indices']),np.array(obj['test_indices'])
        A,B=preprocess_fit(X[tr],X[te])
        for row in obj['rows']:
            if row['method'] not in ['mi','relief','mrmr','sfrip']:continue
            detail=row['detail'];w=np.array(detail['weights']);fs=list(detail['features'])
            order=np.argsort(-w,kind='stable').tolist();restored=[]
            while True:
                cert=certificate(A,y[tr],fs,weights=w,eps=.01)
                if cert['upper_bound']<=.1+1e-10:break
                feature=next(a for a in order if a not in fs)
                fs=sorted(fs+[feature]);restored.append(feature)
            metric=scores(A,y[tr],B,y[te],fs,104,w)
            rows.append({'dataset':obj['dataset'],'fold':obj['fold'],'method':row['method'],
                'features':fs,'restored':restored,'certificate':cert,'metrics':metric,
                'removed':1-len(fs)/obj['d']})
    methods={}
    for method in ['mi','relief','mrmr','sfrip']:
        grouped={}
        for row in rows:
            if row['method']==method:grouped.setdefault(row['dataset'],[]).append(row)
        datasets={name:{'clean':float(np.mean([r['metrics']['0.0']['balanced_accuracy'] for r in group])),
                        'removed':float(np.mean([r['removed'] for r in group])),
                        'restored':float(np.mean([len(r['restored']) for r in group]))} for name,group in grouped.items()}
        methods[method]={'datasets':datasets,'means':{key:float(np.mean([r[key] for r in datasets.values()])) for key in ['clean','removed','restored']}}
    output={'status':'posthoc_equal_constraint_diagnostic_not_official_baseline_algorithm',
            'rows':rows,'methods':methods}
    (directory/'matched-constraint-audit.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps({method:r['means'] for method,r in methods.items()},indent=2))


if __name__=='__main__':main()
