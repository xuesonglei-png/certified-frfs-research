"""Verify prototype scores and bounds without selecting a preferred mode."""
import json
import numpy as np
from run_decision_iteration import ROOT,scaled_split
from run_metric_graph_iteration import scores
from certified_frfs.datasets import load_dataset
from certified_frfs.core import certificate
from threadpoolctl import threadpool_limits


def main():
    threadpool_limits(1);directory=ROOT/'results/pool-prototype';summary={};checks=0
    for path in sorted(directory.glob('*-s*-f*.json')):
        obj=json.loads(path.read_text());X,y=load_dataset(obj['dataset'])
        tr,te=np.array(obj['train_indices']),np.array(obj['test_indices'])
        A,B=scaled_split(X[tr],X[te]);w=np.array(obj['weights'])
        for row in obj['rows']:
            fs=row['features'];assert scores(A,y[tr],B,y[te],fs,104,w)==row['metrics']
            actual=certificate(A,y[tr],fs,weights=w,eps=.01)
            assert actual==row['loss_certificate'] and actual['upper_bound']<=.1+1e-10
            summary.setdefault(row['mode'],[]).append({'dataset':obj['dataset'],
                'clean':row['metrics']['0.0']['balanced_accuracy'],
                'removed':1-len(fs)/obj['d']});checks+=1
    means={mode:{'clean':float(np.mean([r['clean'] for r in rows])),
                 'removed':float(np.mean([r['removed'] for r in rows]))} for mode,rows in summary.items()}
    (directory/'summary.json').write_text(json.dumps({'rows':summary,'means':means,
        'verified_scores_and_certificates':checks,'status':'posthoc_prototype_not_confirmatory'},indent=2),encoding='utf-8')
    print(json.dumps({'means':means,'verified':checks},indent=2))


if __name__=='__main__':main()
