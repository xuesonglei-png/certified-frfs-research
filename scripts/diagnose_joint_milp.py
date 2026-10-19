"""All15 first-inner training-only joint metric/support screening."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from diagnose_guarded_support import FOLDER as PRIOR, NAMES
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.joint_milp import joint_certified_metric


def main():
    threadpool_limits(1); folder=ROOT/'results/joint-milp-diagnostic'; folder.mkdir(exist_ok=True)
    output=folder/'complete.json'
    if output.exists(): raise FileExistsError(output)
    paths=[Path(__file__), ROOT/'src/certified_frfs/joint_milp.py', ROOT/'research/joint-milp-protocol.md']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    rows=[]
    for name in NAMES:
        path=PRIOR/f'{name}.json'; p=json.loads(path.read_text()); s=p['splits'][0]
        X,y=load_dataset(name); ri=np.array(s['reference_indices']); A,_=preprocess_fit(X[ri],X[ri]); started=time.perf_counter()
        fitted=joint_certified_metric(A,y[ri],time_limit=10.)
        rows.append({'dataset':name,'d':A.shape[1],'reference_indices':ri.tolist(),'fit':fitted,
            'seconds':time.perf_counter()-started,'prior_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'data_sha256':hashlib.sha256((ROOT/'data/processed'/f'{name}.npz').read_bytes()).hexdigest()})
        print(name,'features',len(fitted['features']),'/',A.shape[1], 'fallback',fitted['full_fallback'],
            'status',fitted['fit']['solver_status'],'seconds',round(rows[-1]['seconds'],2),flush=True)
        (folder/'partial.json').write_text(json.dumps({'complete':False,'rows':rows,'source_hashes':hashes},indent=2),encoding='utf-8')
    removed=float(np.mean([1-len(r['fit']['features'])/r['d'] for r in rows]))
    nonfull=sum(len(r['fit']['features'])<r['d'] for r in rows)
    gates={'five_nonfull_datasets':nonfull>=5,'mean_removed_at_least_fifteen_percent':removed>=.15}
    assert hashes=={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    result={'complete':True,'rows':rows,'mean_removed':removed,'nonfull_datasets':nonfull,
        'gates':gates,'gate_passed':all(gates.values()),'source_hashes':hashes,
        'scope':'training_only_joint_support_metric_feasibility_not_prediction_advantage'}
    output.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)

if __name__=='__main__':main()
