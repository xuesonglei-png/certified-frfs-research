from concurrent.futures import ProcessPoolExecutor,as_completed
import json,hashlib,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from diagnose_witness_mixture import ROOT,tasks
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.bound_direct import direct_bound_search

FOLDER=ROOT/'results/direct-bound-search'


def run(task):
    threadpool_limits(1);stem=f"{task['group']}-{task['dataset']}-{task['split']}-{task['candidate']}";out=FOLDER/f'{stem}.json'
    if out.exists():return
    scenario=json.loads((ROOT/'results/scenario-repair-diagnostic'/f'{stem}.json').read_text())
    mixture=json.loads((ROOT/'results/witness-mixture-diagnostic'/f'{stem}.json').read_text())
    if task['group']=='original15':X,y=load_dataset(task['dataset'])
    else:
        with np.load(ROOT/'data/newdata-compact'/f"{task['dataset']}.npz") as data:X,y=data['X'],data['y']
    ri=np.array(mixture['reference_indices']);A,_=preprocess_fit(X[ri],X[ri]);full=np.ones(X.shape[1])/np.sqrt(X.shape[1]);fs=mixture['features']
    initial=np.array(scenario['fit']['weights'])[fs];start=time.perf_counter();fit=direct_bound_search(A,y[ri],fs,full,initial);elapsed=time.perf_counter()-start
    result={'task':task,'fit':fit,'old_bound':mixture['result']['upper_bound'],'seconds':elapsed,
        'fingerprints':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'src/certified_frfs/bound_direct.py',ROOT/'research/direct-bound-search-protocol.md']}}
    out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(stem,result['old_bound'],'->',fit['bound'],'cert',fit['certificate']['feasible'],flush=True)


if __name__=='__main__':
    FOLDER.mkdir(exist_ok=True);jobs=[]
    for task in tasks():
        p=json.loads((ROOT/'results/scenario-repair-diagnostic'/f"{task['group']}-{task['dataset']}-{task['split']}-{task['candidate']}.json").read_text())
        if p['fit']['trace'][-1].get('certificate',{}).get('majority_feasible') and not p['fit']['certified']:jobs.append(task)
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,t) for t in jobs]):f.result()
    rows=[json.loads(p.read_text()) for p in FOLDER.glob('*.json') if not p.name.startswith('complete')]
    result={'candidates':len(rows),'admitted':sum(r['fit']['certificate']['feasible'] for r in rows),'gate_passed':sum(r['fit']['certificate']['feasible'] for r in rows)>=3,
        'mean_decrease':float(np.mean([r['old_bound']-r['fit']['bound'] for r in rows])),'scope':'fixed_support_direct_bound_development_not_prediction'}
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)
