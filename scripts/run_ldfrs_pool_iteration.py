"""Iteration22 development: density-informed certified paths; ambiguous baseline.

Freeze candidate rules before outer evaluation; shared holdout/metric protocol.
No official LDFRS reproduction or confirmatory significance claim.
"""
import argparse,hashlib,json
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from run_metric_graph_iteration import scores
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import select_validated_pool,preprocess_fit


def run(job):
    name,fold,iteration=job;threadpool_limits(1)
    out=ROOT/f'results/predictive-iteration{iteration}'/f'{name}-s101-f{fold}.json'
    if out.exists():raise ValueError('preserve existing iteration22 results')
    sources=[Path(__file__),ROOT/'scripts/run_metric_graph_iteration.py',
             *sorted((ROOT/'src/certified_frfs').glob('*.py'))]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    X,y=load_dataset(name)
    tr,te=list(StratifiedKFold(3,shuffle=True,random_state=101).split(X,y))[fold]
    A,B=preprocess_fit(X[tr],X[te]);rows=[]
    methods=['pool_ldfrs','pool','ldfrs','all','mi','relief','mrmr','sfrip'] if iteration==22 else ['pool_ldfrs_exact']
    for method in methods:
        result=select_validated_pool(X[tr],y[tr],method=method,mode='robust',seed=101,joint=True)
        assert np.allclose(A,result.pop('preprocessed_training'))
        metric=scores(A,y[tr],B,y[te],result['features'],104,np.array(result['weights']))
        rows.append({'method':method,'mode':'robust','detail':result,'metrics':metric})
        print(json.dumps({'dataset':name,'fold':fold,'method':method,
            'k':len(result['features']),'clean':metric['0.0']['balanced_accuracy']}),flush=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'dataset':name,'fold':fold,'seed':101,'d':X.shape[1],
        'train_indices':tr.tolist(),'test_indices':te.tolist(),'rows':rows,
        'source_hashes':hashes,'status':'development_not_confirmatory',
        'baseline_status':'independent_LDFRS_conventions_not_author_validated'},indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folds',nargs='+',type=int,default=[0])
    p.add_argument('--iteration',type=int,choices=[22,23],default=22)
    p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(run,(name,fold,a.iteration)) for name in ['iris','wine','sonar','wdbc','ionosphere'] for fold in a.folds]
        for future in as_completed(futures):future.result()
