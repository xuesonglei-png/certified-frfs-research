"""Iteration17: frozen training-only triplet metric + certified graph groups."""
import argparse, hashlib, json, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT, scaled_split, subset, configs
from certified_frfs.datasets import load_dataset
from certified_frfs.graph_recent import select_fgs, refine_group_candidates
from certified_frfs.triplet_metric import learn_triplet_weights


def scores(A,y,B,labels,fs,seed,w):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import balanced_accuracy_score,f1_score
    model=KNeighborsClassifier(n_neighbors=min(5,len(y)),metric='manhattan',algorithm='brute')
    model.fit(A[:,fs]*w[fs],y)
    rng=np.random.default_rng(seed); result={}
    for eps in [0.,.01,.05]:
        versions=[B] if eps==0 else [np.clip(B+rng.uniform(-eps,eps,B.shape),0,1) for _ in range(3)]
        values=[]
        for Z in versions:
            pred=model.predict(Z[:,fs]*w[fs])
            values.append([balanced_accuracy_score(labels,pred),f1_score(labels,pred,average='macro',zero_division=0)])
        ba,f1=np.mean(values,axis=0)
        result[str(eps)]={'balanced_accuracy':float(ba),'macro_f1':float(f1)}
    return result


def run(job):
    threadpool_limits(1)
    name,seed,fold=job; iteration=19
    out=ROOT/f'results/predictive-iteration{iteration}'/f'{name}-s{seed}-f{fold}.json'
    sources=[Path(__file__),ROOT/'scripts/run_decision_iteration.py',
             *sorted((ROOT/'src/certified_frfs').glob('*.py'))]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    if out.exists():
        if json.loads(out.read_text())['source_hashes']!=hashes: raise ValueError('stale cache')
        return {'dataset':name,'fold':fold,'status':'cached'}
    X,y=load_dataset(name)
    tr,te=list(StratifiedKFold(3,shuffle=True,random_state=seed).split(X,y))[fold]
    ni=min(3,int(np.unique(y[tr],return_counts=True)[1].min()))
    if ni<2:return {'dataset':name,'status':'insufficient_inner_counts'}
    splits=list(StratifiedKFold(ni,shuffle=True,random_state=seed+1).split(X[tr],y[tr]))
    graph_cfg=[{'beta':b,'xi':x} for b in [.5,.8,1.] for x in [1.,2.]]
    methods=['metric_graph_nominal','metric_graph_robust','all_metric','mi','relief','mrmr','sfrip']
    rows=[]
    for method in methods:
        candidates=[]; start=time.perf_counter()
        for cfg in graph_cfg if method.startswith('metric_graph') else ([{}] if method=='all_metric' else configs(method)):
            vals=[]; sizes=[]
            for inner,(ii,jj) in enumerate(splits):
                A,B=scaled_split(X[tr][ii],X[tr][jj]); yy=y[tr][ii]
                w=learn_triplet_weights(A,yy,seed=seed+inner)
                if method.startswith('metric_graph'):
                    base=select_fgs(A*w,yy,**cfg)
                    detail=refine_group_candidates(A,yy,base,eps=.01 if method.endswith('robust') else 0.,
                                                   dependency_eta=.1,weights=w)
                    fs=detail['features']
                elif method=='all_metric': fs=list(range(A.shape[1]))
                else: fs=subset(A*w,yy,method,cfg,seed)
                metric=scores(A,yy,B,y[tr][jj],fs,seed+2,w)
                vals.append((metric['0.0']['balanced_accuracy']+metric['0.01']['balanced_accuracy'])/2);sizes.append(len(fs))
            candidates.append({'config':cfg,'score':float(np.mean(vals)),'inner_scores':vals,'mean_k':float(np.mean(sizes))})
        chosen=max(candidates,key=lambda r:(r['score'],-r['mean_k']))
        A,B=scaled_split(X[tr],X[te]); w=learn_triplet_weights(A,y[tr],seed=seed)
        detail=None
        if method.startswith('metric_graph'):
            base=select_fgs(A*w,y[tr],**chosen['config'])
            detail=refine_group_candidates(A,y[tr],base,eps=.01 if method.endswith('robust') else 0.,
                                           dependency_eta=.1,weights=w);fs=detail['features']
        elif method=='all_metric':fs=list(range(A.shape[1]))
        else:fs=subset(A*w,y[tr],method,chosen['config'],seed)
        metric=scores(A,y[tr],B,y[te],fs,seed+3,w)
        rows.append({'method':method,'features':fs,'weights':w.tolist(),'config':chosen['config'],
                     'inner_candidates':candidates,'metrics':metric,'detail':detail,
                     'seconds':time.perf_counter()-start})
        print(json.dumps({'dataset':name,'fold':fold,'method':method,'k':len(fs),'clean':metric['0.0']['balanced_accuracy']}),flush=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'dataset':name,'seed':seed,'fold':fold,'d':X.shape[1],
        'train_indices':tr.tolist(),'test_indices':te.tolist(),'rows':rows,'source_hashes':hashes,
        'status':'development_not_confirmatory','classifier':'L1_5NN',
        'certificate_scope':'fixed_triplet_metric_shared_box_dependency_eta_01'},indent=2),encoding='utf-8')
    return {'dataset':name,'fold':fold,'status':'complete'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',default=['iris','wine','sonar','wdbc','ionosphere']);p.add_argument('--seed',type=int,default=101);p.add_argument('--all-folds',action='store_true');p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        fs=[pool.submit(run,(n,a.seed,f)) for n in a.datasets for f in (range(3) if a.all_folds else [0])]
        for f in as_completed(fs):print(json.dumps(f.result()))
