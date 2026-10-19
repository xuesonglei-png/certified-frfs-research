"""Posthoc matched-dimension diagnostic; never used for model selection."""
import argparse
import json
import numpy as np
from run_decision_iteration import ROOT,scaled_split,scores
from certified_frfs.datasets import load_dataset
from certified_frfs import baselines
from threadpoolctl import threadpool_limits


def main(iteration):
    threadpool_limits(1)
    directory=ROOT/f'results/predictive-iteration{iteration}'
    rows=[]
    functions={'mi':baselines.select_mi,'relief':baselines.select_relieff,'mrmr':baselines.select_mrmr}
    for path in sorted(directory.glob('*-s*-f*.json')):
        obj=json.loads(path.read_text())
        proposed=next(r for r in obj['rows'] if r['method']=='graph_robust_majority')
        X,y=load_dataset(obj['dataset'])
        tr,te=np.array(obj['train_indices']),np.array(obj['test_indices'])
        A,B=scaled_split(X[tr],X[te])
        k=len(proposed['features'])
        for method,function in functions.items():
            fs=function(A,y[tr],k=k)
            metric=scores(A,y[tr],B,y[te],fs,obj['seed']+3,k=5)
            rows.append({'dataset':obj['dataset'],'fold':obj['fold'],'method':method,
                         'features':fs,'matched_k':k,'metrics':metric,
                         'proposed_metrics':proposed['metrics']})
    summary={}
    for method in functions:
        datasets={}
        for row in rows:
            if row['method']==method:
                datasets.setdefault(row['dataset'],[]).append(
                    row['proposed_metrics']['0.0']['balanced_accuracy']-
                    row['metrics']['0.0']['balanced_accuracy'])
        delta={name:float(np.mean(values)) for name,values in datasets.items()}
        summary[method]={'dataset_deltas':delta,'mean_delta':float(np.mean(list(delta.values()))),
                         'wins_ties_losses':[sum(v>1e-12 for v in delta.values()),
                         sum(abs(v)<=1e-12 for v in delta.values()),sum(v< -1e-12 for v in delta.values())]}
    (directory/'matched-dimension-audit.json').write_text(json.dumps({'rows':rows,'summary':summary,
        'status':'posthoc_dimension_diagnostic_not_independent_confirmation'},indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--iteration',type=int,choices=[15,16],default=15)
    main(parser.parse_args().iteration)
