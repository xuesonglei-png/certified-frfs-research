"""Iteration26: fold-local certified selection recipes, no post-selection restoration."""
import argparse,hashlib,json
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from run_majority_metric_iteration import weights
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit,validation_scores
from certified_frfs.majority_metric import MajorityTerms
from certified_frfs.crossfit_majority import recipe_candidates


def inner_splits(y):
    y=np.asarray(y);labels,counts=np.unique(y,return_counts=True)
    singleton=np.isin(y,labels[counts==1]);ids=np.flatnonzero(~singleton)
    splits=min(3,int(np.unique(y[ids],return_counts=True)[1].min()))
    return [(np.r_[ids[ii],np.flatnonzero(singleton)],ids[jj])
            for ii,jj in StratifiedKFold(splits,shuffle=True,random_state=142).split(ids,y[ids])]


def key(family,config):return json.dumps({'family':family,'config':config},sort_keys=True)


def run(job):
    name,fold,iteration=job;threadpool_limits(1)
    out=ROOT/f'results/predictive-iteration{iteration}'/f'{name}-s101-f{fold}.json'
    if out.exists():raise ValueError(f'preserve iteration{iteration}')
    sources=[Path(__file__),ROOT/'scripts/run_majority_metric_iteration.py',*sorted((ROOT/'src/certified_frfs').glob('*.py'))]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    X,y=load_dataset(name);d=X.shape[1]
    tr,te=list(StratifiedKFold(3,shuffle=True,random_state=101).split(X,y))[fold]
    splits=inner_splits(y[tr]);methods=['protected','all','mi','relief','mrmr','sfrip']
    trace={method:{} for method in methods};split_records=[];empty_outputs=[]
    for index,(ii,jj) in enumerate(splits):
        A,B=preprocess_fit(X[tr][ii],X[tr][jj]);split_records.append({'train':ii.tolist(),'validation':jj.tolist()})
        for family in ('uniform','triplet','majority'):
            w=weights(A,y[tr][ii],family)
            for method in methods:
                candidates,_=recipe_candidates(A,y[tr][ii],w,method,include_margin=iteration==27);cache={}
                for candidate in candidates:
                    fs=candidate['features'];fskey=tuple(fs)
                    if not fs:
                        empty_outputs.append({'method':method,'family':family,'config':candidate['config'],'inner':index})
                        continue
                    if fskey not in cache:cache[fskey]=validation_scores(A,y[tr][ii],B,y[tr][jj],fs,w,144+index,joint=True)
                    metrics=cache[fskey];k=key(family,candidate['config'])
                    record=trace[method].setdefault(k,{'family':family,'config':candidate['config'],'folds':[]})
                    record['folds'].append({'inner':index,'features':fs,'validation':metrics,
                        'score':(metrics['clean']+metrics['noise01'])/2})
    F,Q=preprocess_fit(X[tr],X[te]);final_weights={};rows=[]
    for method in methods:
        candidates=[candidate for candidate in trace[method].values() if len(candidate['folds'])==len(splits)]
        if not candidates:raise ValueError('all configurations have undefined empty-subset fold scores: '+method)
        for candidate in candidates:
            assert len(candidate['folds'])==len(splits)
            candidate['score']=float(np.mean([r['score'] for r in candidate['folds']]))
            candidate['mean_size']=float(np.mean([len(r['features']) for r in candidate['folds']]))
        chosen=max(candidates,key=lambda r:(r['score'],-r['mean_size'],r['family']=='uniform'))
        family=chosen['family']
        if family not in final_weights:final_weights[family]=weights(F,y[tr],family)
        w=final_weights[family];final_candidates,paths=recipe_candidates(F,y[tr],w,method,include_margin=iteration==27)
        fs=next(r['features'] for r in final_candidates if r['config']==chosen['config'])
        if not fs:raise ValueError('selected final recipe returns an empty subset: '+method)
        metric=validation_scores(F,y[tr],Q,y[te],fs,w,104,joint=True)
        workspace=MajorityTerms(F,y[tr],weights=w);full=workspace.bounds(range(d),allowed_enemies=2)
        final=workspace.bounds(fs,allowed_enemies=2);protected=full>1e-10
        groups=[(y[tr]==label)&protected for label in np.unique(y)]
        counts=[int(np.sum((final>1e-10)&g)) for g in groups];required=[int(np.ceil(.9*g.sum()-1e-12)) for g in groups]
        if method=='protected':assert all(c>=r for c,r in zip(counts,required))
        rows.append({'method':method,'chosen':chosen,'trace':candidates,'features':fs,'weights':w.tolist(),
            'restoration':[],'metrics':metric,'paths':paths,
            'certificate':{'allowed_enemies':2,'protected_counts':[int(g.sum()) for g in groups],
                'retained_counts':counts,'required':required,
                'lower_bounds':np.nan_to_num(final,posinf=1e100,neginf=-1e100).tolist(),
                'scope':'finite_training_majority_retention_not_test_risk'}})
        print(json.dumps({'dataset':name,'fold':fold,'method':method,'family':family,'k':len(fs),**metric}),flush=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'dataset':name,'fold':fold,'d':d,'train_indices':tr.tolist(),'test_indices':te.tolist(),
        'inner_splits':split_records,'rows':rows,'source_hashes':hashes,'empty_inner_outputs':empty_outputs,
        'empty_output_policy':'exclude_configuration_if_any_inner_fold_returns_empty; no_attribute_fill',
        'status':'development_recipe_crossfit_unequal_budgets'},indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',default=['iris','wine','sonar','wdbc','ionosphere'])
    p.add_argument('--iteration',type=int,choices=[26,27],default=26)
    p.add_argument('--folds',type=int,nargs='+',default=[0,1,2]);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(run,(name,fold,a.iteration)) for name in a.datasets for fold in a.folds]
        for future in as_completed(futures):future.result()
