"""Fixed ten-direction development comparison, all original datasets."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.feasible_query_descent import fit_feasible_query
from certified_frfs.trajectory_certificate import certify_capped_trajectory
from certified_frfs.capped_sparse import sparse_trajectory_extension
from certified_frfs.validated_pool import preprocess_fit, validation_scores

FOLDER=ROOT/'results/long-capped-sparse'


def fingerprints(name):
    paths=[Path(__file__),ROOT/'research/long-capped-sparse-protocol.md',
        ROOT/'results/capped-sparse-extension'/f'{name}.json',ROOT/'data/processed'/f'{name}.npz']
    paths.extend(sorted((ROOT/'src/certified_frfs').glob('*.py')))
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run(name):
    threadpool_limits(1);hashes=fingerprints(name);out=FOLDER/f'{name}.json'
    if out.exists():assert json.loads(out.read_text())['fingerprints']==hashes;return
    prior=json.loads((ROOT/'results/capped-sparse-extension'/f'{name}.json').read_text());X,y=load_dataset(name);rows=[]
    for index,s in enumerate(prior['splits']):
        ri,vi=np.array(s['reference_indices']),np.array(s['validation_indices']);start=time.perf_counter()
        trajectory=fit_feasible_query(X[ri],y[ri],certified=False,weight_cap_ratio=2,steps=10)
        certified=certify_capped_trajectory(X[ri],y[ri],trajectory)
        fit=sparse_trajectory_extension(X[ri],y[ri],certified);seconds=time.perf_counter()-start
        A,B=preprocess_fit(X[ri],X[vi]);c=fit['compact_selected']
        metrics=validation_scores(A,y[ri],B,y[vi],c['features'],np.array(c['weights']),144+index,joint=True)
        rows.append({'reference_indices':ri.tolist(),'validation_indices':vi.tolist(),'trajectory':trajectory,
            'certified_trajectory':certified,'fit':fit,'metrics':metrics,'seconds':seconds})
        print(name,index,metrics,'size',len(c['features']),'steps',len(trajectory['trace'])-1,flush=True)
    assert fingerprints(name)==hashes
    out.write_text(json.dumps({'dataset':name,'d':X.shape[1],'fingerprints':hashes,'splits':rows},indent=2),encoding='utf-8')


def summarize(names):
    old=json.loads((ROOT/'results/capped-sparse-extension/complete.json').read_text());rows=[]
    for name in names:
        p=json.loads((FOLDER/f'{name}.json').read_text());assert p['fingerprints']==fingerprints(name)
        base=next(r for r in old['rows'] if r['dataset']==name)
        rows.append({'dataset':name,'utility':float(np.mean([(s['metrics']['clean']+s['metrics']['noise01'])/2 for s in p['splits']])),
            'removed':float(np.mean([1-len(s['fit']['compact_selected']['features'])/p['d'] for s in p['splits']])),
            'control':base['utility']})
    result={'rows':rows,'mean_utility':float(np.mean([r['utility'] for r in rows])),
        'mean_removed':float(np.mean([r['removed'] for r in rows])),
        'comparisons':{m:{'gain':float(np.mean([r['utility']-r['control'][m] for r in rows])),
            'wins':sum(r['utility']>r['control'][m]+1e-12 for r in rows)} for m in ('compact','fgs2023','sfrip2025')},
        'scope':'ten_direction_all15_all44_exposed_development_not_confirmation'}
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)


if __name__=='__main__':
    names=[r['dataset'] for r in json.loads((ROOT/'results/capped-sparse-extension/complete.json').read_text())['rows']]
    FOLDER.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,n) for n in names]):f.result()
    summarize(names)
