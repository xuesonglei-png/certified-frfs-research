from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib,json
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.robust_linear_refit import fit_robust_linear

FOLDER=ROOT/'results/robust-linear-refit'

def run(name):
    threadpool_limits(1);out=FOLDER/f'{name}.json'
    if out.exists():raise FileExistsError(out)
    X,y=load_dataset(name);p=json.loads((ROOT/'results/linear-retention'/f'{name}.json').read_text());records=[]
    for index,s in enumerate(p['splits']):
        ri,vi=np.array(s['reference_indices']),np.array(s['validation_indices']);A,B=preprocess_fit(X[ri],X[vi]);labels=y[ri]
        W,b,classes=np.array(s['teacher_coef']),np.array(s['teacher_intercept']),np.array(s['teacher_classes'])
        distinct={}
        for r in s['candidates']:
            c=r['candidate'];distinct.setdefault(tuple(sorted(c['features'])),c)
        keys=sorted(distinct,key=lambda fs:(len(fs),fs));keys=list(dict.fromkeys(keys[:4]+[tuple(range(X.shape[1]))]))
        trials=[]
        for key in keys:
            fit=fit_robust_linear(A,labels,W,b,classes,list(key))
            trials.append({'candidate':distinct[key],'fit':fit})
        feasible=[r for r in trials if r['fit']['feasible']]
        fallback=not bool(feasible)
        if fallback:
            selected={'candidate':distinct[tuple(range(X.shape[1]))],'fit':{'coef':W.tolist(),'intercept':b.tolist(),'feasible':True,'scope':'full_teacher_fallback'}}
        else:selected=min(feasible,key=lambda r:(len(r['candidate']['features']),r['fit']['objective']))
        coef=np.array(selected['fit']['coef']);intercept=np.array(selected['fit']['intercept']);rng=np.random.default_rng(144+index)
        versions=[B]+[np.clip(B+rng.uniform(-.01,.01,B.shape),0,1) for _ in range(3)]
        values=[float(balanced_accuracy_score(y[vi],classes[(Q@coef.T+intercept).argmax(axis=1)])) for Q in versions]
        records.append({'reference_indices':ri.tolist(),'validation_indices':vi.tolist(),'trials':trials,'selected':selected,'fallback':fallback,
                        'metrics':{'clean':values[0],'query_noise01':float(np.mean(values[1:]))}})
        print(name,index,'size',len(selected['candidate']['features']),'clean',values[0],'fallback',fallback,flush=True)
    paths=[Path(__file__),ROOT/'research/robust-linear-refit-protocol.md',ROOT/'src/certified_frfs/robust_linear_refit.py',ROOT/'src/certified_frfs/linear_retention.py',ROOT/'results/linear-retention'/f'{name}.json',ROOT/'data/processed'/f'{name}.npz']
    out.write_text(json.dumps({'dataset':name,'d':X.shape[1],'splits':records,'fingerprints':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}},indent=2),encoding='utf-8')

def summarize(names):
    baseline=json.loads((ROOT/'results/nested-classifier-pipelines/complete.json').read_text());control={r['dataset']:r for r in baseline['rows']};rows=[]
    for name in names:
        p=json.loads((FOLDER/f'{name}.json').read_text());s=p['splits']
        rows.append({'dataset':name,'clean':float(np.mean([r['metrics']['clean'] for r in s])),'removed':float(np.mean([1-len(r['selected']['candidate']['features'])/p['d'] for r in s])),
                     'fallbacks':sum(r['fallback'] for r in s),'controls':{m:control[name]['means']['logistic'][m]['clean'] for m in ('full','fgs2023','sfrip2025','proposed_uniform')}})
    comparisons={m:{'gain':float(np.mean([r['clean']-r['controls'][m] for r in rows])),'wins':sum(r['clean']>r['controls'][m]+1e-12 for r in rows)} for m in rows[0]['controls']}
    removed=float(np.mean([r['removed'] for r in rows]));result={'rows':rows,'mean_clean':float(np.mean([r['clean'] for r in rows])),'mean_removed':removed,'comparisons':comparisons,
        'gate_passed':removed>=.1 and comparisons['full']['gain']>=-.005 and all(comparisons[m]['gain']>=.005 and comparisons[m]['wins']>=10 for m in ('fgs2023','sfrip2025')),
        'scope':'all15_all44_deployed_robust_linear_refit_exposed_development_not_confirmation'}
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)

if __name__=='__main__':
    names=[r['dataset'] for r in json.loads((ROOT/'results/linear-retention/complete.json').read_text())['rows']];FOLDER.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,n) for n in names]):f.result()
    summarize(names)
