from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib,json
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.exact_box_linear_refit import fit_exact_box_linear as fit_robust_linear
from certified_frfs.linear_retention import box_class_margins

FOLDER=ROOT/'results/exact-box-matched-controls'

def metrics(queries,truth,classes,fit):
    W,b=np.array(fit['coef']),np.array(fit['intercept'])
    predictions=[classes[(Q@W.T+b).argmax(axis=1)] for Q in queries]
    return {'clean':float(balanced_accuracy_score(truth,predictions[0])),
            'query_noise01':float(np.mean([balanced_accuracy_score(truth,p) for p in predictions[1:]])),
            'class_recalls':{str(c):float(np.mean(predictions[0][truth==c]==c)) for c in np.unique(truth)}}

def run(name):
    threadpool_limits(1);out=FOLDER/f'{name}.json'
    if out.exists():raise FileExistsError(out)
    X,y=load_dataset(name)
    paths=[ROOT/'results'/folder/f'{name}.json' for folder in ('linear-retention','exact-box-linear-refit','nested-classifier-pipelines')]
    teacher,prior,nested=[json.loads(p.read_text()) for p in paths];rows=[]
    for index,s in enumerate(prior['splits']):
        t=teacher['splits'][index];n=nested['splits'][index]
        ri,vi=np.array(s['reference_indices']),np.array(s['validation_indices'])
        assert ri.tolist()==t['reference_indices']==n['reference_indices'] and vi.tolist()==t['validation_indices']==n['validation_indices']
        A,B=preprocess_fit(X[ri],X[vi]);labels=y[ri];classes=np.array(t['teacher_classes']);W,b=np.array(t['teacher_coef']),np.array(t['teacher_intercept'])
        rng=np.random.default_rng(144+index);queries=[B]+[np.clip(B+rng.uniform(-.01,.01,B.shape),0,1) for _ in range(3)]
        full=next(r['fit'] for r in s['trials'] if len(r['candidate']['features'])==X.shape[1])
        fallback=not full['feasible']
        if fallback:full={'feasible':True,'coef':W.tolist(),'intercept':b.tolist(),'scope':'full_teacher_fallback'}
        results={'proposed':{'features':s['selected']['candidate']['features'],'fit':s['selected']['fit'],'metrics':metrics(queries,y[vi],classes,s['selected']['fit']),
                             'fallback':s['fallback']},
                 'robust_full':{'features':list(range(X.shape[1])),'fit':full,'metrics':metrics(queries,y[vi],classes,full),'fallback':fallback}}
        for m in ('fgs2023','sfrip2025'):
            fs=n['results']['logistic'][m]['features']
            fitted=fit_robust_linear(A,labels,W,b,classes,fs) if fs else {'feasible':False,'status':'empty_support'}
            results[m]={'features':fs,'fit':fitted,'metrics':metrics(queries,y[vi],classes,fitted) if fitted['feasible'] else None,'fallback':False}
        for m,r in results.items():
            if r['fit']['feasible']:
                margin=box_class_margins(A,labels,np.array(r['fit']['coef']),np.array(r['fit']['intercept']),classes)
                original=box_class_margins(A,labels,W,b,classes);protected=original>1e-10
                assert np.all(margin[protected]>=.5*original[protected]-1e-8) and np.all(margin[protected]>1e-10)
        np.testing.assert_allclose(results['proposed']['metrics']['clean'],s['metrics']['clean'],atol=1e-12)
        rows.append({'reference_indices':ri.tolist(),'validation_indices':vi.tolist(),'results':results})
        print(name,index,'matched controls scored',flush=True)
    paths += [Path(__file__),ROOT/'research/exact-box-linear-refit-protocol.md',ROOT/'src/certified_frfs/exact_box_linear_refit.py',ROOT/'src/certified_frfs/linear_retention.py',ROOT/'src/certified_frfs/validated_pool.py',ROOT/'data/processed'/f'{name}.npz']
    out.write_text(json.dumps({'dataset':name,'d':X.shape[1],'splits':rows,'fingerprints':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}},indent=2),encoding='utf-8')

def summarize(names):
    rows=[]
    for name in names:
        p=json.loads((FOLDER/f'{name}.json').read_text());s=p['splits']
        clean={m:float(np.mean([r['results'][m]['metrics']['clean'] for r in s])) for m in ('proposed','robust_full')}
        conditional={}
        for m in ('fgs2023','sfrip2025'):
            valid=[r for r in s if r['results'][m]['metrics'] is not None]
            conditional[m]={'feasible_splits':len(valid),'total_splits':len(s),
                'gain_if_feasible':float(np.mean([r['results']['proposed']['metrics']['clean']-r['results'][m]['metrics']['clean'] for r in valid])) if valid else None}
        rows.append({'dataset':name,'clean':clean,'gain_vs_robust_full':clean['proposed']-clean['robust_full'],
            'removed':float(np.mean([1-len(r['results']['proposed']['features'])/p['d'] for r in s])),
            'full_fallbacks':sum(r['results']['robust_full']['fallback'] for r in s),'conditional':conditional})
    gain=float(np.mean([r['gain_vs_robust_full'] for r in rows]));wins=sum(r['gain_vs_robust_full']>1e-12 for r in rows);removed=float(np.mean([r['removed'] for r in rows]))
    result={'rows':rows,'means':{m:float(np.mean([r['clean'][m] for r in rows])) for m in ('proposed','robust_full')},
        'gain_vs_robust_full':gain,'wins':wins,'mean_removed':removed,'full_fallbacks':sum(r['full_fallbacks'] for r in rows),
        'baseline_feasible_splits':{m:sum(r['conditional'][m]['feasible_splits'] for r in rows) for m in ('fgs2023','sfrip2025')},
        'attribution_gate_passed':gain>=.005 and wins>=10 and removed>=.1,
        'scope':'all15_all44_matched_robust_training_attribution_exposed_development_conditional_baselines_not_confirmation'}
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)

if __name__=='__main__':
    names=[r['dataset'] for r in json.loads((ROOT/'results/robust-linear-refit/complete.json').read_text())['rows']];FOLDER.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,n) for n in names]):f.result()
    summarize(names)
