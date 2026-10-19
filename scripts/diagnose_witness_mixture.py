"""Fixed rejected-candidate screen; no validation-driven certificate tuning."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib,json,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.witness_mixture import witness_mixture_loss_bound

FOLDER=ROOT/'results/witness-mixture-diagnostic'


def fingerprints():
    paths=[Path(__file__),ROOT/'research/witness-mixture-protocol.md']
    paths.extend(sorted((ROOT/'src/certified_frfs').glob('*.py')))
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run(task):
    threadpool_limits(1);name=task['dataset'];group=task['group'];index=task['split'];ci=task['candidate']
    out=FOLDER/f'{group}-{name}-{index}-{ci}.json';hashes=fingerprints()
    if out.exists():assert json.loads(out.read_text())['fingerprints']==hashes;return
    path=ROOT/task['source'];prior=json.loads(path.read_text());s=prior['splits'][index];c=s['fit']['candidates'][ci]
    if group=='original15':X,y=load_dataset(name);ri=np.array(s['reference_indices']);data_path=ROOT/'data/processed'/f'{name}.npz'
    else:
        data_path=ROOT/'data/newdata-compact'/f'{name}.npz'
        with np.load(data_path) as data:X,y=data['X'],data['y']
        ri=np.array(s['train_indices'])
    assert c['certificate']['majority_feasible'] and c['certificate']['dependency_certificate']['upper_bound']>.1
    A,_=preprocess_fit(X[ri],X[ri]);full=np.ones(X.shape[1])/np.sqrt(X.shape[1]);v=np.array(c['weights']);fs=c['features']
    start=time.perf_counter();bound=witness_mixture_loss_bound(A,y[ri],fs,full,v[fs]);seconds=time.perf_counter()-start
    result={'task':task,'reference_indices':ri.tolist(),'features':fs,'weights':v.tolist(),'result':bound,'seconds':seconds,
        'old_upper':c['certificate']['dependency_certificate']['upper_bound'],'admitted':bound['upper_bound']<=.1,
        'fingerprints':hashes,'prior_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'data_sha256':hashlib.sha256(data_path.read_bytes()).hexdigest()}
    assert fingerprints()==hashes
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(group,name,index,ci,'upper',result['old_upper'],'->',bound['upper_bound'],'admitted',result['admitted'],'seconds',round(seconds,2),flush=True)


def tasks():
    output=[]
    sources=[('original15',p) for p in sorted((ROOT/'results/capped-sparse-extension').glob('*.json')) if not p.name.startswith('complete')]
    sources.append(('newdata',ROOT/'results/newdata-compact/spectf.json'))
    for group,path in sources:
        p=json.loads(path.read_text())
        for index,s in enumerate(p['splits']):
            for ci,c in enumerate(s['fit']['candidates']):
                cert=c['certificate']
                if c['origin']=='capped_sparse' and cert['majority_feasible'] and cert['dependency_certificate']['upper_bound']>.1:
                    output.append({'group':group,'dataset':p['dataset'],'split':index,'candidate':ci,'source':str(path.relative_to(ROOT))})
    assert sum(t['group']=='original15' for t in output)==18 and sum(t['group']=='newdata' for t in output)==9
    return output


if __name__=='__main__':
    FOLDER.mkdir(exist_ok=True);jobs=tasks()
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run,t) for t in jobs]):future.result()
    rows=[json.loads((FOLDER/f"{t['group']}-{t['dataset']}-{t['split']}-{t['candidate']}.json").read_text()) for t in jobs]
    result={'tasks':len(rows),'admitted':sum(r['admitted'] for r in rows),'gate_passed':sum(r['admitted'] for r in rows)>=3,
        'mean_tightening':float(np.mean([r['old_upper']-r['result']['upper_bound'] for r in rows])),
        'mean_seconds':float(np.mean([r['seconds'] for r in rows])),'scope':'fixed_rejected_training_candidates_not_prediction_or_confirmation'}
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)
