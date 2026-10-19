"""Iteration29: directly inner-validate a train-only union/certification recipe.

Uniform metric isolates feature selection from learned-metric selection.
All selectors use identical train-only aggregation and validation splits.
"""
import argparse,hashlib,json
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from run_crossfit_majority_iteration import inner_splits
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit,validation_scores
from certified_frfs.crossfit_majority import recipe_candidates
from certified_frfs.stable_union import certify_union

METHODS=('protected','all','mi','relief','mrmr','sfrip')


def fit_candidates(X,y):
    """Fit each aggregation exclusively on the supplied training observations."""
    d=X.shape[1];w=np.ones(d)/np.sqrt(d);collection={m:{} for m in METHODS}
    split_records=[]
    for index,(ii,jj) in enumerate(inner_splits(y)):
        A,_=preprocess_fit(X[ii],X[jj])
        split_records.append({'train':ii.tolist(),'unused':jj.tolist()})
        for method in METHODS:
            candidates,_=recipe_candidates(A,y[ii],w,method)
            for c in candidates:
                key=json.dumps(c['config'],sort_keys=True)
                r=collection[method].setdefault(key,{'config':c['config'],'fold_subsets':[]})
                r['fold_subsets'].append(c['features'])
    A,_=preprocess_fit(X,X);result={}
    for method,data in collection.items():
        rows=[];cache={}
        for c in data.values():
            fs=sorted(set(a for subset in c['fold_subsets'] for a in subset))
            # An empty constituent is a valid selector outcome, but an empty
            # aggregate cannot define the distance-based classifier.
            if not fs:continue
            cert=None
            if method=='protected':
                k=tuple(fs)
                if k not in cache:cache[k]=certify_union(A,y,fs,w)
                cert=cache[k];fs=cert['features']
            rows.append({**c,'features':fs,'certificate':cert})
        result[method]=rows
    return result,split_records


def run(job):
    name,fold=job;threadpool_limits(1)
    out=ROOT/'results/predictive-iteration29'/f'{name}-s101-f{fold}.json'
    if out.exists():raise ValueError('preserve iteration29')
    X,y=load_dataset(name);d=X.shape[1];w=np.ones(d)/np.sqrt(d)
    tr,te=list(StratifiedKFold(3,shuffle=True,random_state=101).split(X,y))[fold]
    trace={m:{} for m in METHODS};records=[]
    splits=inner_splits(y[tr])
    for index,(ii,jj) in enumerate(splits):
        fitted,subsplits=fit_candidates(X[tr][ii],y[tr][ii])
        A,B=preprocess_fit(X[tr][ii],X[tr][jj])
        records.append({'train':ii.tolist(),'validation':jj.tolist(),'aggregation_splits':subsplits})
        for method in METHODS:
            cache={}
            for c in fitted[method]:
                fs=c['features'];k=tuple(fs)
                if k not in cache:cache[k]=validation_scores(A,y[tr][ii],B,y[tr][jj],fs,w,144+index,joint=True)
                metrics=cache[k];key=json.dumps(c['config'],sort_keys=True)
                trace[method].setdefault(key,{'config':c['config'],'folds':[]})['folds'].append({
                    'inner':index,**c,'validation':metrics,'score':(metrics['clean']+metrics['noise01'])/2})
    final,subsplits=fit_candidates(X[tr],y[tr]);F,Q=preprocess_fit(X[tr],X[te]);rows=[]
    for method in METHODS:
        candidates=[c for c in trace[method].values() if len(c['folds'])==len(splits)]
        for c in candidates:
            c['score']=float(np.mean([r['score'] for r in c['folds']]))
            c['mean_size']=float(np.mean([len(r['features']) for r in c['folds']]))
        # Final availability is training-only; log ineligible configurations.
        available={json.dumps(c['config'],sort_keys=True):c for c in final[method]}
        candidates=[c for c in candidates if json.dumps(c['config'],sort_keys=True) in available]
        if not candidates:raise ValueError('no usable configuration: '+method)
        chosen=max(candidates,key=lambda c:(c['score'],-c['mean_size']))
        c=available[json.dumps(chosen['config'],sort_keys=True)]
        metrics=validation_scores(F,y[tr],Q,y[te],c['features'],w,104,joint=True)
        rows.append({'method':method,**c,'chosen':chosen,'trace':candidates,'weights':w.tolist(),'metrics':metrics})
        print(name,fold,method,metrics['clean'],len(c['features']),flush=True)
    sources=[Path(__file__),ROOT/'src/certified_frfs/stable_union.py',ROOT/'src/certified_frfs/crossfit_majority.py',
        ROOT/'src/certified_frfs/majority_metric.py',ROOT/'src/certified_frfs/baselines.py',ROOT/'src/certified_frfs/recent.py',
        ROOT/'src/certified_frfs/validated_pool.py',ROOT/'scripts/run_crossfit_majority_iteration.py']
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'dataset':name,'fold':fold,'d':d,'train_indices':tr.tolist(),
        'test_indices':te.tolist(),'inner_splits':records,'final_aggregation_splits':subsplits,'rows':rows,
        'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'status':'nested_union_uniform_metric_repeated_development_unequal_recipe_budgets'},indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',default=['iris','wine','sonar','wdbc','ionosphere'])
    p.add_argument('--folds',type=int,nargs='+',default=[0,1,2]);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(run,(name,fold)) for name in a.datasets for fold in a.folds]
        for f in as_completed(futures):f.result()
