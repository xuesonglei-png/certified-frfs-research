"""Development iteration 15: independently implemented interaction graph.

Nested parameter selection; same L1 5NN and perturbation draws as iteration14.
No claim of reproducing author defaults, original certificate, or test guarantee.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from run_decision_iteration import ROOT, configs, subset, scores, scaled_split
from certified_frfs.datasets import load_dataset
from certified_frfs.graph_recent import select_fgs, refine_group_candidates
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits


def run(job):
    threadpool_limits(1)
    name,seed,fold,certified=job
    iteration=16 if certified else 15
    out=ROOT/f'results/predictive-iteration{iteration}'/f'{name}-s{seed}-f{fold}.json'
    sources=[Path(__file__), ROOT/'scripts/run_decision_iteration.py',
             ROOT/'scripts/run_predictive_certified_iteration.py',
             *sorted((ROOT/'src/certified_frfs').glob('*.py'))]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    if out.exists():
        if json.loads(out.read_text())['source_hashes']!=hashes:
            raise ValueError('source changed; preserve previous results')
        return {'dataset':name,'fold':fold,'status':'cached'}
    X,y=load_dataset(name)
    tr,te=list(StratifiedKFold(3,shuffle=True,random_state=seed).split(X,y))[fold]
    count=min(3,int(np.unique(y[tr],return_counts=True)[1].min()))
    if count<2:
        return {'dataset':name,'status':'insufficient_inner_counts'}
    splits=list(StratifiedKFold(count,shuffle=True,random_state=seed+1).split(X[tr],y[tr]))
    inner=[(*scaled_split(X[tr][a],X[tr][b]),y[tr][a],y[tr][b]) for a,b in splits]
    A,B=scaled_split(X[tr],X[te])
    cache={}
    def graph(data,labels,config,key):
        signature=(key,config['beta'],config['xi'])
        if signature not in cache:
            cache[signature]=select_fgs(data,labels,**config)
        return cache[signature]
    rows=[]
    graph_configs=[{'beta':b,'xi':x} for b in [.5,.8,1.] for x in [1.,2.]]
    methods=['fgs_independent','graph_nominal_majority','graph_robust_majority',
             'all','sfrip','mi','relief','mrmr']
    for method in methods:
        start=time.perf_counter()
        candidates=[]
        for config in graph_configs if method in methods[:3] else configs(method):
            values,sizes=[],[]
            for index,(Ia,Ib,ya,yb) in enumerate(inner):
                if method in methods[:3]:
                    base=graph(Ia,ya,config,index)
                    fs=(base['features'] if method=='fgs_independent' else
                        refine_group_candidates(Ia,ya,base,eps=.01 if method=='graph_robust_majority' else 0.,
                                                dependency_eta=.1 if certified else None)['features'])
                else:
                    fs=subset(Ia,ya,method,config,seed)
                metric=scores(Ia,ya,Ib,yb,fs,seed+2,k=5)
                values.append((metric['0.0']['balanced_accuracy']+metric['0.01']['balanced_accuracy'])/2)
                sizes.append(len(fs))
            candidates.append({'config':config,'score':float(np.mean(values)),
                               'inner_scores':values,'mean_k':float(np.mean(sizes))})
        chosen=max(candidates,key=lambda row:(row['score'],-row['mean_k']))
        if method in methods[:3]:
            base=graph(A,y[tr],chosen['config'],'outer_train')
            detail=base if method=='fgs_independent' else refine_group_candidates(
                A,y[tr],base,eps=.01 if method=='graph_robust_majority' else 0.,
                dependency_eta=.1 if certified else None)
            fs=detail['features']
        else:
            fs=subset(A,y[tr],method,chosen['config'],seed)
            detail=None
        metric=scores(A,y[tr],B,y[te],fs,seed+3,k=5)
        rows.append({'method':method,'features':fs,'config':chosen['config'],
                     'inner_candidates':candidates,'metrics':metric,'detail':detail,
                     'seconds':time.perf_counter()-start})
        print(json.dumps({'dataset':name,'fold':fold,'method':method,'k':len(fs),
                          'clean':metric['0.0']['balanced_accuracy']}),flush=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'dataset':name,'seed':seed,'fold':fold,'d':X.shape[1],
        'train_indices':tr.tolist(),'test_indices':te.tolist(),'rows':rows,'source_hashes':hashes,
        'status':'development_not_confirmatory','classifier':'L1_5NN',
        'dependency_eta':.1 if certified else None,
        'certificate_scope':'fixed_weight_shared_box_dependency' if certified else
                            'finite_training_LOO_majority_not_original_dependency_reduct'},indent=2),encoding='utf-8')
    return {'dataset':name,'fold':fold,'status':'complete'}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--datasets',nargs='+',default=['iris','wine','sonar','wdbc','ionosphere'])
    parser.add_argument('--seed',type=int,default=101)
    parser.add_argument('--all-folds',action='store_true')
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--certified',action='store_true')
    args=parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run,(name,args.seed,fold,args.certified)) for name in args.datasets
                 for fold in (range(3) if args.all_folds else [0])]
        for future in as_completed(futures):
            print(json.dumps(future.result()),flush=True)
