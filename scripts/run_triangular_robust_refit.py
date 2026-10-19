"""Frozen fuzzy group support LP pilot with identical robust full control."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib,json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.triangular_robust_refit import fit_triangular_robust
from run_fuzzy_teacher_grouped_certified import score
from review_box_iterations import paired

FOLDER=ROOT/'results/triangular-robust-refit'

def run(name):
    threadpool_limits(1);out=FOLDER/f'{name}.json'
    if out.exists():raise FileExistsError(out)
    sourcepath=ROOT/'results/fuzzy-teacher-grouped-certified'/f'{name}.json';p=json.loads(sourcepath.read_text());X,y=load_dataset(name);rows=[]
    for i,s in enumerate(p['splits']):
        ri,vi=np.array(s['reference_indices']),np.array(s['validation_indices']);A,B=preprocess_fit(X[ri],X[vi]);t=s['teacher'];W,b,classes=np.array(t['coef']),np.array(t['intercept']),np.array(t['classes'])
        candidates=[r['candidate'] for r in s['trials'] if r['candidate']['fuzzy_certificate']['feasible']]
        mass=np.sqrt(p['d']);candidates.append({'features':list(range(p['d'])),'weights':(np.ones(p['d'])/mass).tolist(),'origin':'full'})
        trials=[];seen=set()
        for c in candidates:
            key=tuple(sorted(c['features']))
            if key in seen:continue
            seen.add(key);fit=fit_triangular_robust(A,y[ri],W,b,classes,c['features'])
            trials.append({'candidate':c,'fit':fit})
        feasible=[r for r in trials if r['fit']['feasible']];full=next(r for r in trials if len(r['candidate']['features'])==p['d'])
        selected=min(feasible,key=lambda r:(len(r['candidate']['features']),r['fit']['objective'])) if feasible else {'candidate':full['candidate'],'fit':{**t,'feasible':True,'scope':'full_teacher_fallback'}}
        fullfit=full['fit'] if full['fit']['feasible'] else t
        rng=np.random.default_rng(144+i);queries=[B]+[np.clip(B+rng.uniform(-.01,.01,B.shape),0,1) for _ in range(3)]
        metrics=lambda f:{'clean':score(B,y[vi],f),'noise':float(np.mean([score(Q,y[vi],f) for Q in queries[1:]]))}
        rows.append({'reference_indices':ri.tolist(),'validation_indices':vi.tolist(),'teacher':t,'trials':trials,'selected':selected,'fallback':not feasible,
            'full_fit':fullfit,'full_fallback':not full['fit']['feasible'],'metrics':metrics(selected['fit']),'full_metrics':metrics(fullfit)})
        print(name,i,'support',len(selected['candidate']['features']),'BA',rows[-1]['metrics']['clean'],flush=True)
    files=[Path(__file__),sourcepath,ROOT/'research/triangular-robust-refit-protocol.md',ROOT/'src/certified_frfs/triangular_robust_refit.py',ROOT/'src/certified_frfs/triangular_classifier.py',ROOT/'scripts/run_fuzzy_teacher_grouped_certified.py',ROOT/'src/certified_frfs/validated_pool.py',ROOT/'data/processed'/f'{name}.npz']
    out.write_text(json.dumps({'dataset':name,'d':p['d'],'splits':rows,'fingerprints':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}},indent=2),encoding='utf-8')

def summarize(names):
    rows=[]
    for name in names:
        p=json.loads((FOLDER/f'{name}.json').read_text());s=p['splits'];rows.append({'dataset':name,'clean':float(np.mean([r['metrics']['clean'] for r in s])),
            'full':float(np.mean([r['full_metrics']['clean'] for r in s])),'removed':float(np.mean([1-len(r['selected']['candidate']['features'])/p['d'] for r in s])),
            'fallbacks':sum(r['fallback'] for r in s),'full_fallbacks':sum(r['full_fallback'] for r in s)})
    result={'rows':rows,'mean_clean':float(np.mean([r['clean'] for r in rows])),'mean_full':float(np.mean([r['full'] for r in rows])),
        'mean_removed':float(np.mean([r['removed'] for r in rows])),'vs_matched_full':paired([r['clean']-r['full'] for r in rows]),
        'scope':'all15_all44_exact_coupled_triangular_LP_frozen_supports_development_not_nested_confirmation'}
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print({k:v for k,v in result.items() if k!='rows'},flush=True)

if __name__=='__main__':
    FOLDER.mkdir(exist_ok=True);names=[r['dataset'] for r in json.loads((ROOT/'results/fuzzy-teacher-grouped-certified/complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,n) for n in names]):f.result()
    summarize(names)
