"""All15/all44 frozen support classwise-anchor MILP pilot with matched full."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib,json,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.classwise_box_linear_refit import fit_classwise_box_linear
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.linear_retention import box_class_margins
from review_box_iterations import paired

FOLDER=ROOT/'results/classwise-box-pilot'

def evaluate(queries,truth,classes,fit):
    predictions=[classes[(Q@np.array(fit['coef']).T+np.array(fit['intercept'])).argmax(axis=1)] for Q in queries]
    scores=[float(np.mean([np.mean(p[truth==c]==c) for c in np.unique(truth)])) for p in predictions]
    return {'clean':scores[0],'query_noise01':float(np.mean(scores[1:]))}

def check(A,y,classes,W,b,fit):
    old=box_class_margins(A,y,W,b,classes);new=box_class_margins(A,y,np.array(fit['coef']),np.array(fit['intercept']),classes)
    protected=old>1e-10;kept=(new>=.5*old-1e-8)&(new>1e-10)&protected
    required=[int(np.ceil(.9*np.sum((y==c)&protected)-1e-12)) for c in classes]
    counts=[int(np.sum((y==c)&kept)) for c in classes]
    assert all(a>=r for a,r in zip(counts,required))
    return {'required':required,'retained_counts':counts,'protected_count':int(protected.sum()),'deployed_margins':new.tolist()}

def run(name):
    threadpool_limits(1);out=FOLDER/f'{name}.json'
    if out.exists():raise FileExistsError(out)
    paths=[ROOT/'results'/f/f'{name}.json' for f in ('nested-teacher-regularized','common-fold-recent')]
    source,baseline=[json.loads(p.read_text()) for p in paths];X,y=load_dataset(name);rows=[]
    for i,s in enumerate(source['splits']):
        ri,vi=np.array(s['reference_indices']),np.array(s['validation_indices']);A,B=preprocess_fit(X[ri],X[vi]);labels=y[ri]
        teacher=s['outer_stages'][str(s['full_config']['C'])];classes=np.array(teacher['teacher_classes']);W,b=np.array(teacher['teacher_coef']),np.array(teacher['teacher_intercept'])
        candidates={};rejected=[];mass=np.sqrt(X.shape[1]);fullweights=np.ones(X.shape[1])/mass
        for policy in teacher['policies'].values():
            failure=policy['failure'];c=policy['candidate']
            if policy['fallback'] and failure and 'requested_candidate' in failure:c=failure['requested_candidate']
            fs=c['features'];key=tuple(sorted(fs))
            if key in candidates or any(tuple(sorted(r['features']))==key for r in rejected):continue
            cert=weight_certificate(A,labels,fs,fullweights,np.array(c['weights'])[fs])
            if cert['feasible']:candidates[key]={**c,'certificate':cert}
            else:rejected.append({'features':fs,'certificate':cert})
        trials=[]
        for key,c in sorted(candidates.items(),key=lambda z:(len(z[0]),z[0])):
            start=time.perf_counter();fit=fit_classwise_box_linear(A,labels,W,b,classes,c['features']);duration=time.perf_counter()-start
            certificate=check(A,labels,classes,W,b,fit) if fit['feasible'] else None
            trials.append({'candidate':c,'fit':fit,'deployed_certificate':certificate,'seconds':duration})
            print(name,i,'support',len(key),'MILP feasible',fit['feasible'],'seconds',round(duration,2),flush=True)
        full=next(r for r in trials if len(r['candidate']['features'])==X.shape[1]);teacherfit={'feasible':True,'coef':W.tolist(),'intercept':b.tolist(),'scope':'explicit_full_teacher_fallback'}
        feasible=[r for r in trials if r['fit']['feasible']]
        fallback=not feasible
        selected=min(feasible,key=lambda r:(len(r['candidate']['features']),r['fit']['objective'])) if feasible else {
            'candidate':full['candidate'],'fit':teacherfit,'deployed_certificate':check(A,labels,classes,W,b,teacherfit),'seconds':0}
        fullfit=full['fit'] if full['fit']['feasible'] else teacherfit
        rng=np.random.default_rng(144+i);queries=[B]+[np.clip(B+rng.uniform(-.01,.01,B.shape),0,1) for _ in range(3)]
        bs=baseline['splits'][i];assert ri.tolist()==bs['reference_indices'] and vi.tolist()==bs['validation_indices']
        rows.append({'reference_indices':ri.tolist(),'validation_indices':vi.tolist(),'teacher':teacher,'rejected':rejected,'trials':trials,
            'selected':selected,'fallback':fallback,'full_fit':fullfit,'full_fallback':not full['fit']['feasible'],
            'metrics':evaluate(queries,y[vi],classes,selected['fit']),'full_metrics':evaluate(queries,y[vi],classes,fullfit),
            'controls':{'all_anchor_full':s['full_metrics'],'fgs_native':bs['results']['fgs2023']['native']['metrics'],
                        'sfrip_native':bs['results']['sfrip2025']['native']['metrics']}})
    paths += [Path(__file__),ROOT/'research/classwise-box-pilot-protocol.md',ROOT/'src/certified_frfs/classwise_box_linear_refit.py',ROOT/'src/certified_frfs/feasible_transport.py',ROOT/'src/certified_frfs/linear_retention.py',ROOT/'src/certified_frfs/validated_pool.py',ROOT/'data/processed'/f'{name}.npz']
    out.write_text(json.dumps({'dataset':name,'d':X.shape[1],'splits':rows,'fingerprints':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}},indent=2),encoding='utf-8')

def summarize(names):
    rows=[]
    for name in names:
        p=json.loads((FOLDER/f'{name}.json').read_text());s=p['splits']
        rows.append({'dataset':name,'clean':float(np.mean([r['metrics']['clean'] for r in s])),
            'full':float(np.mean([r['full_metrics']['clean'] for r in s])),
            'noise':float(np.mean([r['metrics']['query_noise01'] for r in s])),
            'full_noise':float(np.mean([r['full_metrics']['query_noise01'] for r in s])),
            'removed':float(np.mean([1-len(r['selected']['candidate']['features'])/p['d'] for r in s])),
            'fallbacks':sum(r['fallback'] for r in s),'full_fallbacks':sum(r['full_fallback'] for r in s),
            'successful_fits':sum(r['fit']['feasible'] for t in s for r in t['trials']),
            'controls':{m:float(np.mean([r['controls'][m]['clean'] for r in s])) for m in ('all_anchor_full','fgs_native','sfrip_native')}})
    result={'rows':rows,'mean_clean':float(np.mean([r['clean'] for r in rows])),'mean_full':float(np.mean([r['full'] for r in rows])),
        'mean_removed':float(np.mean([r['removed'] for r in rows])),'vs_matched_full':paired([r['clean']-r['full'] for r in rows]),
        'controls':{m:paired([r['clean']-r['controls'][m] for r in rows]) for m in rows[0]['controls']},
        'scope':'all15_all44_classwise90_reference_anchor_pilot_exposed_development_not_nested_or_confirmation'}
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print({k:v for k,v in result.items() if k!='rows'},flush=True)

if __name__=='__main__':
    FOLDER.mkdir(exist_ok=True);names=[r['dataset'] for r in json.loads((ROOT/'results/hierarchical-teacher-selection/complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,n) for n in names]):f.result()
    summarize(names)
