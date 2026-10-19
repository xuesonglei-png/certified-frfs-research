from concurrent.futures import ProcessPoolExecutor,as_completed
import json,hashlib,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from diagnose_witness_mixture import ROOT,FOLDER as MIXTURE,tasks
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.shared_metric_attack import attack_shared_metric
from certified_frfs.core import gamma

FOLDER=ROOT/'results/shared-metric-attack'


def run(task):
    threadpool_limits(1);stem=f"{task['group']}-{task['dataset']}-{task['split']}-{task['candidate']}"
    out=FOLDER/f'{stem}.json'
    if out.exists():raise FileExistsError(out)
    prior_path=MIXTURE/f'{stem}.json';prior=json.loads(prior_path.read_text())
    if task['group']=='original15':X,y=load_dataset(task['dataset'])
    else:
        with np.load(ROOT/'data/newdata-compact'/f"{task['dataset']}.npz") as data:X,y=data['X'],data['y']
    ri=np.array(prior['reference_indices']);A,_=preprocess_fit(X[ri],X[ri]);wf=np.ones(X.shape[1])/np.sqrt(X.shape[1]);wr=np.array(prior['weights'])
    start=time.perf_counter();fit=attack_shared_metric(A,y[ri],wf,wr);seconds=time.perf_counter()-start
    Z=np.array(fit['perturbed_table']);actual=gamma(Z,y[ri],weights=wf)-gamma(Z,y[ri],weights=wr)
    assert abs(actual-fit['observed_loss_lower'])<1e-10 and actual<=prior['result']['upper_bound']+1e-10
    result={'task':task,'fit':fit,'clean_loss':gamma(A,y[ri],weights=wf)-gamma(A,y[ri],weights=wr),
        'proven_budget_violation':actual>.1,'upper':prior['result']['upper_bound'],'seconds':seconds,
        'prior_sha256':hashlib.sha256(prior_path.read_bytes()).hexdigest(),
        'fingerprints':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__),ROOT/'src/certified_frfs/shared_metric_attack.py',ROOT/'research/shared-metric-attack-protocol.md')}}
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(stem,'clean',result['clean_loss'],'attacked',actual,'upper',result['upper'],'violation',result['proven_budget_violation'],flush=True)


if __name__=='__main__':
    FOLDER.mkdir(exist_ok=True);jobs=tasks()
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,t) for t in jobs]):f.result()
    rows=[json.loads((FOLDER/f"{t['group']}-{t['dataset']}-{t['split']}-{t['candidate']}.json").read_text()) for t in jobs]
    result={'candidates':len(rows),'proven_budget_violations':sum(r['proven_budget_violation'] for r in rows),
        'clean_budget_violations':sum(r['clean_loss']>.1 for r in rows),'unresolved':sum(not r['proven_budget_violation'] for r in rows),
        'mean_upper_minus_attack':float(np.mean([r['upper']-r['fit']['observed_loss_lower'] for r in rows])),
        'scope':'shared_attack_lower_witness_not_certification'}
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)
