"""Iteration20: independent validation over certified paths and shared metrics."""
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
    name,fold,joint=job;threadpool_limits(1)
    iteration=21 if joint else 20
    out=ROOT/f'results/predictive-iteration{iteration}'/f'{name}-s101-f{fold}.json'
    sources=[Path(__file__),ROOT/'scripts/run_metric_graph_iteration.py',
             *sorted((ROOT/'src/certified_frfs').glob('*.py'))]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    if out.exists():
        # Adding this resumability branch changes only the runner. Require all
        # imported implementation hashes unchanged before reusing old results.
        previous=json.loads(out.read_text())['source_hashes']
        for key,value in hashes.items():
            if key!=str(Path(__file__).relative_to(ROOT)) and previous.get(key)!=value:
                raise ValueError('implementation changed; preserve results and use new directory')
        return
    X,y=load_dataset(name)
    tr,te=list(StratifiedKFold(3,shuffle=True,random_state=101).split(X,y))[fold]
    A,B=preprocess_fit(X[tr],X[te]);rows=[]
    for method,mode in [('pool','nominal'),('pool','robust'),('pool','compression'),
                        ('all','robust'),('mi','robust'),('relief','robust'),
                        ('mrmr','robust'),('sfrip','robust')]:
        result=select_validated_pool(X[tr],y[tr],method=method,mode=mode,seed=101,joint=joint)
        assert np.allclose(A,result.pop('preprocessed_training'))
        metric=scores(A,y[tr],B,y[te],result['features'],104,np.array(result['weights']))
        rows.append({'method':method,'mode':mode,'detail':result,'metrics':metric})
        print(json.dumps({'dataset':name,'fold':fold,'method':method,'mode':mode,
                          'k':len(result['features']),'family':result['family'],
                          'clean':metric['0.0']['balanced_accuracy']}),flush=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'dataset':name,'fold':fold,'seed':101,'d':X.shape[1],
        'train_indices':tr.tolist(),'test_indices':te.tolist(),'rows':rows,
        'source_hashes':hashes,
        'status':'development_holdout_selector_not_confirmatory'},indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',default=['iris','wine','sonar','wdbc','ionosphere']);p.add_argument('--all-folds',action='store_true');p.add_argument('--workers',type=int,default=3);p.add_argument('--joint',action='store_true');a=p.parse_args()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(run,(n,f,a.joint)) for n in a.datasets for f in (range(3) if a.all_folds else [0])]
        for f in as_completed(futures):f.result()
