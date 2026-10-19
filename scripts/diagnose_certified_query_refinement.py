from concurrent.futures import ProcessPoolExecutor,as_completed
import json,hashlib,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit,validation_scores
from certified_frfs.certified_query_refinement import refine_certified_query

FOLDER=ROOT/'results/certified-query-refinement'


def fingerprints(name):
    paths=[Path(__file__),ROOT/'research/certified-query-refinement-protocol.md',
        ROOT/'results/long-capped-sparse'/f'{name}.json',ROOT/'data/processed'/f'{name}.npz']
    paths.extend(sorted((ROOT/'src/certified_frfs').glob('*.py')))
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run(name):
    threadpool_limits(1);hashes=fingerprints(name);out=FOLDER/f'{name}.json'
    if out.exists():assert json.loads(out.read_text())['fingerprints']==hashes;return
    p=json.loads((ROOT/'results/long-capped-sparse'/f'{name}.json').read_text());X,y=load_dataset(name);rows=[]
    for index,s in enumerate(p['splits']):
        ri,vi=np.array(s['reference_indices']),np.array(s['validation_indices']);start=time.perf_counter()
        fit=refine_certified_query(X[ri],y[ri],s['fit']['compact_selected']);seconds=time.perf_counter()-start
        A,B=preprocess_fit(X[ri],X[vi]);metrics=validation_scores(A,y[ri],B,y[vi],fit['features'],np.array(fit['weights']),144+index,joint=True)
        rows.append({'reference_indices':ri.tolist(),'validation_indices':vi.tolist(),'fit':fit,'metrics':metrics,
            'initial_metrics':s['metrics'],'seconds':seconds})
        print(name,index,metrics,'updates',len(fit['trace'])-1,'size',len(fit['features']),flush=True)
    assert fingerprints(name)==hashes
    out.write_text(json.dumps({'dataset':name,'d':p['d'],'splits':rows,'fingerprints':hashes},indent=2),encoding='utf-8')


def summarize(names):
    baseline=json.loads((ROOT/'results/long-capped-sparse/complete.json').read_text());rows=[]
    for name in names:
        p=json.loads((FOLDER/f'{name}.json').read_text());assert p['fingerprints']==fingerprints(name)
        base=next(r for r in baseline['rows'] if r['dataset']==name)
        rows.append({'dataset':name,'utility':float(np.mean([(s['metrics']['clean']+s['metrics']['noise01'])/2 for s in p['splits']])),
            'removed':float(np.mean([1-len(s['fit']['features'])/p['d'] for s in p['splits']])),
            'updates':sum(len(s['fit']['trace'])-1 for s in p['splits']),
            'control':{'initial':base['utility'],**base['control']}})
    means=float(np.mean([r['utility'] for r in rows]));gain=float(np.mean([r['utility']-r['control']['initial'] for r in rows]))
    wins=sum(r['utility']>r['control']['initial']+1e-12 for r in rows)
    result={'rows':rows,'mean_utility':means,'mean_removed':float(np.mean([r['removed'] for r in rows])),
        'comparisons':{m:{'gain':float(np.mean([r['utility']-r['control'][m] for r in rows])),
            'wins':sum(r['utility']>r['control'][m]+1e-12 for r in rows)} for m in ('initial','fgs2023','sfrip2025')},
        'gate_passed':gain>=.005 and wins>=10,'scope':'all15_all44_certified_query_refinement_exposed_development_not_confirmation'}
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)


if __name__=='__main__':
    names=[r['dataset'] for r in json.loads((ROOT/'results/long-capped-sparse/complete.json').read_text())['rows']]
    FOLDER.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run,n) for n in names]):future.result()
    summarize(names)
