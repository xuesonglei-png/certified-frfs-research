"""Iteration30: training-selected uniform/robust soft-neighbor metrics.

Use standard fold-local paths, not post-selection unions. Every comparator
receives both metric options; final certification is recomputed after fitting.
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
from certified_frfs.majority_metric import MajorityTerms
from certified_frfs.softvote_metric import learn_softvote_weights

METHODS=('protected','all','mi','relief','mrmr','sfrip')


def metric(A,y,family):
    return np.ones(A.shape[1])/np.sqrt(A.shape[1]) if family=='uniform' else learn_softvote_weights(A,y)


def run(job,iteration=30,refresh_witness=False,flexible_witness=False):
    name,fold=job;threadpool_limits(1)
    out=ROOT/f'results/predictive-iteration{iteration}'/f'{name}-s101-f{fold}.json'
    if out.exists():raise ValueError(f'preserve iteration{iteration}')
    sources=[Path(__file__),*sorted((ROOT/'src/certified_frfs').glob('*.py')),
             ROOT/'scripts/run_crossfit_majority_iteration.py']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    X,y=load_dataset(name);d=X.shape[1]
    tr,te=list(StratifiedKFold(3,shuffle=True,random_state=101).split(X,y))[fold]
    trace={m:{} for m in METHODS};records=[];splits=inner_splits(y[tr])
    for index,(ii,jj) in enumerate(splits):
        A,B=preprocess_fit(X[tr][ii],X[tr][jj]);fits={}
        for family in ('uniform','softvote'):
            w=metric(A,y[tr][ii],family);fits[family]=w.tolist()
            for method in METHODS:
                candidates,_=recipe_candidates(A,y[tr][ii],w,method,refresh_witness=refresh_witness,flexible_witness=flexible_witness);cache={}
                for c in candidates:
                    fs=c['features']
                    if not fs:continue
                    k=tuple(fs)
                    if k not in cache:cache[k]=validation_scores(A,y[tr][ii],B,y[tr][jj],fs,w,144+index,joint=True)
                    scores=cache[k];key=json.dumps([family,c['config']],sort_keys=True)
                    trace[method].setdefault(key,{'family':family,'config':c['config'],'folds':[]})['folds'].append({
                        'inner':index,'features':fs,'validation':scores,'score':(scores['clean']+scores['noise01'])/2})
        records.append({'train':ii.tolist(),'validation':jj.tolist(),'weights':fits})
    F,Q=preprocess_fit(X[tr],X[te]);final_weights={};rows=[]
    for method in METHODS:
        candidates=[c for c in trace[method].values() if len(c['folds'])==len(splits)]
        for c in candidates:
            c['score']=float(np.mean([r['score'] for r in c['folds']]))
            c['mean_size']=float(np.mean([len(r['features']) for r in c['folds']]))
        chosen=max(candidates,key=lambda c:(c['score'],-c['mean_size'],c['family']=='uniform'))
        family=chosen['family']
        if family not in final_weights:final_weights[family]=metric(F,y[tr],family)
        w=final_weights[family];fitted,_=recipe_candidates(F,y[tr],w,method,refresh_witness=refresh_witness,flexible_witness=flexible_witness)
        fs=next(c['features'] for c in fitted if c['config']==chosen['config'])
        if not fs:raise ValueError('selected final subset empty: '+method)
        scores=validation_scores(F,y[tr],Q,y[te],fs,w,104,joint=True);cert=None
        if method=='protected':
            ws=MajorityTerms(F,y[tr],weights=w);p=ws.bounds(range(d),allowed_enemies=2)>1e-10
            bounds=ws.bounds(fs,allowed_enemies=2)
            if refresh_witness:
                from certified_frfs.adaptive_witness import adaptive_bounds
                bounds=np.maximum(bounds,adaptive_bounds(F,y[tr],fs,w))
            if flexible_witness:
                from certified_frfs.flexible_witness import flexible_bounds
                bounds=np.maximum(bounds,flexible_bounds(F,y[tr],fs,w,pool_size=8))
            groups=[(y[tr]==label)&p for label in np.unique(y[tr])]
            required=[int(np.ceil(.9*g.sum()-1e-12)) for g in groups]
            counts=[int(((bounds>1e-10)&g).sum()) for g in groups]
            assert all(c>=r for c,r in zip(counts,required))
            cert={'protected_counts':[int(g.sum()) for g in groups],'required':required,'retained_counts':counts,
                  'lower_bounds':np.nan_to_num(bounds,posinf=1e100,neginf=-1e100).tolist()}
        rows.append({'method':method,'features':fs,'weights':w.tolist(),'metrics':scores,
                     'chosen':chosen,'trace':candidates,'certificate':cert})
        print(name,fold,method,family,scores['clean'],len(fs),flush=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'dataset':name,'fold':fold,'d':d,'train_indices':tr.tolist(),'test_indices':te.tolist(),
        'inner_splits':records,'rows':rows,'source_hashes':hashes,
        'refresh_witness':refresh_witness,
        'flexible_witness':flexible_witness,
        'status':'robust_softvote_development_unequal_recipe_budgets'},indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',default=['iris','wine','sonar','wdbc','ionosphere'])
    p.add_argument('--folds',type=int,nargs='+',default=[0,1,2]);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(run,(name,fold)) for name in a.datasets for fold in a.folds]
        for f in as_completed(futures):f.result()
