from concurrent.futures import ProcessPoolExecutor,as_completed
import json,hashlib,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from diagnose_witness_mixture import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.majority_bound_direct import majority_bound_search

FOLDER=ROOT/'results/majority-bound-search'


def run(path):
    threadpool_limits(1);p=json.loads(path.read_text());task=p['task'];stem=path.stem;out=FOLDER/path.name
    if out.exists():raise FileExistsError(out)
    prior_path=ROOT/'results/witness-mixture-diagnostic'/path.name
    prior=json.loads(prior_path.read_text());scenario_path=ROOT/'results/scenario-repair-diagnostic'/path.name
    scenario=json.loads(scenario_path.read_text())
    if task['group']=='original15':X,y=load_dataset(task['dataset']);data_path=ROOT/'data/processed'/f"{task['dataset']}.npz"
    else:
        data_path=ROOT/'data/newdata-compact'/f"{task['dataset']}.npz"
        with np.load(data_path) as data:X,y=data['X'],data['y']
    ri=np.array(prior['reference_indices']);A,_=preprocess_fit(X[ri],X[ri]);full=np.ones(X.shape[1])/np.sqrt(X.shape[1]);fs=prior['features']
    paths=[Path(__file__),ROOT/'research/majority-bound-search-protocol.md',prior_path,scenario_path,data_path]
    paths.extend(sorted((ROOT/'src/certified_frfs').glob('*.py')))
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    start=time.perf_counter();fit=majority_bound_search(A,y[ri],fs,full,np.array(scenario['fit']['weights'])[fs])
    assert all(hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==value for relative,value in hashes.items())
    result={'task':task,'fit':fit,'seconds':time.perf_counter()-start,
        'initial_bound':scenario['fit']['trace'][-1]['certificate']['dependency_certificate']['upper_bound'],
        'unprotected_search_bound':p['fit']['bound'],'fingerprints':hashes}
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(stem,'bound',result['initial_bound'],'->',fit['bound'],'cert',fit['certificate']['feasible'],flush=True)


if __name__=='__main__':
    FOLDER.mkdir(exist_ok=True);jobs=[p for p in sorted((ROOT/'results/direct-bound-search').glob('*.json')) if not p.name.startswith('complete')]
    assert len(jobs)==12
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,p) for p in jobs]):f.result()
    rows=[json.loads(p.read_text()) for p in FOLDER.glob('*.json') if not p.name.startswith('complete')]
    result={'candidates':len(rows),'admitted':sum(r['fit']['certificate']['feasible'] for r in rows),
        'gate_passed':sum(r['fit']['certificate']['feasible'] for r in rows)>=3,
        'mean_decrease_from_scenario_metric':float(np.mean([r['initial_bound']-r['fit']['bound'] for r in rows])),
        'scope':'majority_preserving_direct_bound_development_not_prediction'}
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)
