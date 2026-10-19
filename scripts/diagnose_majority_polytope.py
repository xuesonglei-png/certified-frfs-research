"""Fit the fixed-witness LP on the frozen twelve rejected guard supports."""
import hashlib
import json
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from diagnose_guarded_support import FOLDER, NAMES
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.majority_polytope import fit_majority_polytope


def main():
    threadpool_limits(1); out = FOLDER/'polytope-admission.json'
    if out.exists(): raise FileExistsError(out)
    rows=[]
    for name in NAMES:
        p=json.loads((FOLDER/f'{name}.json').read_text()); X,y=load_dataset(name)
        for index,s in enumerate(p['splits']):
            c=s['fit']['unconstrained']
            if len(c['features'])==p['d'] or c['certificate']['feasible']: continue
            ri=np.array(s['reference_indices']); A,_=preprocess_fit(X[ri],X[ri]); fs=c['features']; full=np.ones(p['d'])/np.sqrt(p['d'])
            v,info=fit_majority_polytope(A,y[ri],fs,full)
            row={'dataset':name,'inner':index,'features':fs,'status':info['status'],'solver_success':info['solver_success'],
                 'certificate':info.get('certificate'),'weights':None if v is None else v.tolist(),
                 'minimum_constraint':info.get('minimum_constraint'),'weight_shift_l1':info.get('weight_shift_l1'),
                 'prior_sha256':hashlib.sha256((FOLDER/f'{name}.json').read_bytes()).hexdigest()}
            rows.append(row); print(name,index,row['status'],row['certificate'] and row['certificate']['feasible'],flush=True)
    result={'rows':rows,'checked_supports':len(rows),'joint_admissions':sum(bool(r['certificate'] and r['certificate']['feasible']) for r in rows),
            'solver_feasible':sum(r['status']=='polytope_verified' for r in rows),
            'scope':'frozen_training_polytope_admission_only_no_external_scores','source_sha256':hashlib.sha256(__file__.encode()).hexdigest()}
    out.write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)

if __name__=='__main__': main()
