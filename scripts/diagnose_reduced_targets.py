"""Fixed reduced-margin targets on the same twelve training-selected supports."""
import hashlib
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from diagnose_guarded_support import FOLDER
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.majority_polytope import fit_majority_polytope


def main():
    threadpool_limits(1); prior=FOLDER/'polytope-admission.json'; previous=json.loads(prior.read_text())
    output=FOLDER/'reduced-target-admission.json'
    if output.exists(): raise FileExistsError(output)
    sources=[Path(__file__), ROOT/'src/certified_frfs/majority_polytope.py', ROOT/'research/reduced-target-protocol.md']
    source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    rows=[]
    for old in previous['rows']:
        name=old['dataset']; index=old['inner']; p=json.loads((FOLDER/f'{name}.json').read_text())
        X,y=load_dataset(name); ri=np.array(p['splits'][index]['reference_indices']); A,_=preprocess_fit(X[ri],X[ri]); w=np.ones(A.shape[1])/np.sqrt(A.shape[1])
        v,fit=fit_majority_polytope(A,y[ri],old['features'],w,target_rule='reduced')
        rows.append({'dataset':name,'inner':index,'features':old['features'],'fit':fit,
            'weights':None if v is None else v.tolist(),'old_status':old['status'],'old_certificate':old['certificate'],
            'prior_sha256':hashlib.sha256((FOLDER/f'{name}.json').read_bytes()).hexdigest()})
        print(name,index,fit['status'],bool(fit['certificate'] and fit['certificate']['feasible']),flush=True)
    admissions=sum(bool(r['fit']['certificate'] and r['fit']['certificate']['feasible']) for r in rows)
    result={'rows':rows,'joint_admissions':admissions,'supports':len(rows),'gate_passed':admissions>=6,
        'source_hashes':source_hashes,'prior_sha256':hashlib.sha256(prior.read_bytes()).hexdigest(),
        'scope':'reduced_margin_training_targets_no_external_predictions'}
    assert source_hashes=={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    output.write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))

if __name__=='__main__':main()
