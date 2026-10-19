"""Iteration33: utility-directed constrained and matched unconstrained paths.

Reuse immutable iteration31 train-only metric fits and verified unchanged
baseline scores. New selectors each have12validation configurations. No
outer label or baseline test score enters fitting/configuration selection.
"""
import argparse,copy,hashlib,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from run_softvote_iteration import metric
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit,validation_scores
from certified_frfs.utility_search import utility_candidates
from certified_frfs.majority_metric import MajorityTerms
from certified_frfs.adaptive_witness import adaptive_bounds


def run(job,iteration=33,dependency_eta=None,shared_loss=False,crossfit_ranking=False,beam_width=1):
    name,fold=job;threadpool_limits(1);directory=ROOT/f'results/predictive-iteration{iteration}'
    out=directory/f'{name}-s101-f{fold}.json'
    if out.exists():raise ValueError(f'preserve iteration{iteration}')
    source=ROOT/'results/predictive-iteration31'/out.name;old=json.loads(source.read_text())
    X,y=load_dataset(name);tr,te=np.array(old['train_indices']),np.array(old['test_indices']);d=old['d']
    traces={m:{} for m in ('protected','utility')};records=[]
    for index,record in enumerate(old['inner_splits']):
        ii,jj=np.array(record['train']),np.array(record['validation']);A,B=preprocess_fit(X[tr][ii],X[tr][jj])
        fits=[]
        for family,values in record['weights'].items():
            w=np.array(values)
            ranking=None
            if crossfit_ranking:
                from certified_frfs.crossfit_utility import CrossfitUtilityWorkspace
                ranking=CrossfitUtilityWorkspace(X[tr][ii],y[tr][ii],family=family)
            for method in traces:
                start=time.perf_counter();candidates,path=utility_candidates(A,y[tr][ii],w,certified=method=='protected',dependency_eta=dependency_eta if method=='protected' else None,shared_loss=shared_loss if method=='protected' else False,ranking_workspace=ranking,beam_width=beam_width)
                fits.append({'method':method,'family':family,'fit_seconds':time.perf_counter()-start,'path':path})
                cache={}
                for c in candidates:
                    fs=c['features'];key=tuple(fs)
                    if key not in cache:cache[key]=validation_scores(A,y[tr][ii],B,y[tr][jj],fs,w,144+index,joint=True)
                    scores=cache[key];k=json.dumps([family,c['config']],sort_keys=True)
                    traces[method].setdefault(k,{'family':family,'config':c['config'],'folds':[]})['folds'].append({
                        'inner':index,'features':fs,'validation':scores,'score':(scores['clean']+scores['noise01'])/2})
        records.append({**record,'utility_fits':fits})
        print(name,fold,'inner',index,'search complete',flush=True)
    F,Q=preprocess_fit(X[tr],X[te]);weights={};rows=[]
    for method in traces:
        candidates=list(traces[method].values())
        assert len(candidates)==12
        for c in candidates:
            assert len(c['folds'])==len(records)
            c['score']=float(np.mean([r['score'] for r in c['folds']]))
            c['mean_size']=float(np.mean([len(r['features']) for r in c['folds']]))
        chosen=max(candidates,key=lambda c:(c['score'],-c['mean_size'],c['family']=='uniform'))
        family=chosen['family']
        if family not in weights:
            reused=next((r['weights'] for r in old['rows'] if r['chosen']['family']==family),None)
            weights[family]=np.array(reused) if reused is not None else metric(F,y[tr],family)
        ranking=None
        if crossfit_ranking:
            from certified_frfs.crossfit_utility import CrossfitUtilityWorkspace
            ranking=CrossfitUtilityWorkspace(X[tr],y[tr],family=family)
        w=weights[family];start=time.perf_counter();fitted,path=utility_candidates(F,y[tr],w,certified=method=='protected',dependency_eta=dependency_eta if method=='protected' else None,shared_loss=shared_loss if method=='protected' else False,ranking_workspace=ranking,beam_width=beam_width)
        fit_seconds=time.perf_counter()-start
        fs=next(c['features'] for c in fitted if c['config']==chosen['config'])
        scores=validation_scores(F,y[tr],Q,y[te],fs,w,104,joint=True);cert=None
        if method=='protected':
            ws=MajorityTerms(F,y[tr],weights=w);p=ws.bounds(range(d),allowed_enemies=2)>1e-10
            bounds=np.maximum(ws.bounds(fs,allowed_enemies=2),adaptive_bounds(F,y[tr],fs,w))
            groups=[(y[tr]==label)&p for label in np.unique(y[tr])]
            required=[int(np.ceil(.9*g.sum()-1e-12)) for g in groups];counts=[int(((bounds>1e-10)&g).sum()) for g in groups]
            assert all(c>=r for c,r in zip(counts,required))
            cert={'protected_counts':[int(g.sum()) for g in groups],'required':required,'retained_counts':counts,
                'lower_bounds':np.nan_to_num(bounds,posinf=1e100,neginf=-1e100).tolist()}
            if dependency_eta is not None:
                from certified_frfs.core import certificate
                cert['dependency_certificate']=certificate(F,y[tr],fs,weights=w,eps=.01,method='hybrid')
                if shared_loss and cert['dependency_certificate']['upper_bound']>dependency_eta:
                    from certified_frfs.shared_loss import shared_loss_bound
                    cert['dependency_certificate']=shared_loss_bound(F,y[tr],fs,w)
                assert cert['dependency_certificate']['upper_bound']<=dependency_eta
        rows.append({'method':method,'features':fs,'weights':w.tolist(),'metrics':scores,
            'chosen':chosen,'trace':candidates,'path':path,'fit_seconds':fit_seconds,'certificate':cert})
        print(name,fold,method,family,scores['clean'],len(fs),flush=True)
    rows.extend(copy.deepcopy(r) for r in old['rows'] if r['method']!='protected')
    fgs_source=ROOT/'results/fgs-nested-comparison'/out.name;fgs=json.loads(fgs_source.read_text());fgs['method']='fgs';rows.append(fgs)
    sources=[Path(__file__),ROOT/'src/certified_frfs/utility_search.py',ROOT/'src/certified_frfs/majority_metric.py',
        ROOT/'src/certified_frfs/adaptive_witness.py',ROOT/'src/certified_frfs/core.py',
        ROOT/'src/certified_frfs/validated_pool.py',ROOT/'scripts/run_softvote_iteration.py',
        ROOT/'src/certified_frfs/shared_loss.py',ROOT/'src/certified_frfs/margin.py']
    if crossfit_ranking:sources.append(ROOT/'src/certified_frfs/crossfit_utility.py')
    if beam_width>1:sources.append(ROOT/'src/certified_frfs/beam_utility.py')
    directory.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'dataset':name,'fold':fold,'d':d,'train_indices':tr.tolist(),'test_indices':te.tolist(),
        'inner_splits':records,'rows':rows,'previous_protected':next(r for r in old['rows'] if r['method']=='protected'),
        'baseline_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'fgs_source_sha256':hashlib.sha256(fgs_source.read_bytes()).hexdigest(),
        'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'dependency_eta':dependency_eta,
        'shared_loss':shared_loss,
        'crossfit_ranking':crossfit_ranking,
        'beam_width':beam_width,
        'status':'utility_search_repeated_development_shared_cached_train_metric_fits',
        'budgets':'protected/utility/MI/Relief/mRMR/SFRIP:12; FGS:18; all:2; internal fit costs differ'},indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',default=['iris','wine','sonar','wdbc','ionosphere'])
    p.add_argument('--folds',type=int,nargs='+',default=[0]);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(run,(name,fold)) for name in a.datasets for fold in a.folds]
        for f in as_completed(futures):f.result()
