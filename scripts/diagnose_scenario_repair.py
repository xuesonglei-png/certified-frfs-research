from concurrent.futures import ProcessPoolExecutor,as_completed
import json,hashlib,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from diagnose_witness_mixture import ROOT,tasks
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.scenario_repair import repair_with_scenarios

FOLDER=ROOT/'results/scenario-repair-diagnostic'


def run(task):
    threadpool_limits(1);stem=f"{task['group']}-{task['dataset']}-{task['split']}-{task['candidate']}";out=FOLDER/f'{stem}.json'
    if out.exists():raise FileExistsError(out)
    attack_path=ROOT/'results/shared-metric-attack'/f'{stem}.json';attack=json.loads(attack_path.read_text())
    mixture_path=ROOT/'results/witness-mixture-diagnostic'/f'{stem}.json';prior=json.loads(mixture_path.read_text())
    if task['group']=='original15':X,y=load_dataset(task['dataset'])
    else:
        with np.load(ROOT/'data/newdata-compact'/f"{task['dataset']}.npz") as data:X,y=data['X'],data['y']
    ri=np.array(prior['reference_indices']);A,_=preprocess_fit(X[ri],X[ri]);full=np.ones(X.shape[1])/np.sqrt(X.shape[1])
    paths=[Path(__file__),ROOT/'research/scenario-repair-protocol.md',attack_path,mixture_path]
    paths.extend(sorted((ROOT/'src/certified_frfs').glob('*.py')))
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    start=time.perf_counter();fit=repair_with_scenarios(A,y[ri],prior['features'],full,prior['weights'],attack['fit']['perturbed_table'])
    assert all(hashlib.sha256((ROOT/r).read_bytes()).hexdigest()==v for r,v in hashes.items())
    result={'task':task,'fit':fit,'seconds':time.perf_counter()-start,'fingerprints':hashes}
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(stem,'certified',fit['certified'],'LP statuses',[r['lp']['status'] for r in fit['trace']],flush=True)


if __name__=='__main__':
    FOLDER.mkdir(exist_ok=True);jobs=tasks()
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,t) for t in jobs]):f.result()
    rows=[json.loads((FOLDER/f"{t['group']}-{t['dataset']}-{t['split']}-{t['candidate']}.json").read_text()) for t in jobs]
    result={'candidates':len(rows),'new_certified':sum(r['fit']['certified'] for r in rows),
        'gate_passed':sum(r['fit']['certified'] for r in rows)>=3,'scope':'fixed_support_scenario_repair_development_not_prediction'}
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)
