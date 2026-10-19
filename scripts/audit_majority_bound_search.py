import json,hashlib,zipfile
import numpy as np
from threadpoolctl import threadpool_limits
from diagnose_majority_bound_search import ROOT,FOLDER
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.majority_bound_direct import majority_bound_search


def main():
    threadpool_limits(1);counts={'candidates':0,'replayed_rounds':0,'majority_checks':0,'admitted':0}
    for path in sorted(FOLDER.glob('*.json')):
        if path.name.startswith('complete'):continue
        p=json.loads(path.read_text());task=p['task']
        for rel,value in p['fingerprints'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==value,rel
        prior=json.loads((ROOT/'results/witness-mixture-diagnostic'/path.name).read_text())
        old=json.loads((ROOT/'results/scenario-repair-diagnostic'/path.name).read_text())
        if task['group']=='original15':X,y=load_dataset(task['dataset'])
        else:
            with np.load(ROOT/'data/newdata-compact'/f"{task['dataset']}.npz") as data:X,y=data['X'],data['y']
        ri=np.array(prior['reference_indices']);A,_=preprocess_fit(X[ri],X[ri]);full=np.ones(X.shape[1])/np.sqrt(X.shape[1]);fs=prior['features']
        fit=majority_bound_search(A,y[ri],fs,full,np.array(old['fit']['weights'])[fs]);assert fit==p['fit']
        assert fit['certificate']['majority_feasible'] and all(t['after']<=t['before']+1e-12 for t in fit['trace'])
        counts['candidates']+=1;counts['replayed_rounds']+=len(fit['trace']);counts['admitted']+=int(fit['certificate']['feasible'])
        counts['majority_checks']+=sum(len(t['majority_checks']) for t in fit['trace'])
        print(path.stem,'protected search replay audited',flush=True)
    (FOLDER/'complete-audit.json').write_text(json.dumps(counts,indent=2),encoding='utf-8')
    with zipfile.ZipFile(ROOT/'output/majority-bound-search-evidence.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for path in FOLDER.glob('*.json'):z.write(path,str(path.relative_to(ROOT)))
        for path in (ROOT/'src/certified_frfs').glob('*.py'):z.write(path,str(path.relative_to(ROOT)))
        for rel in ('scripts/diagnose_majority_bound_search.py','scripts/audit_majority_bound_search.py','research/majority-bound-search-protocol.md','tests/test_majority_bound_direct.py'):z.write(ROOT/rel,rel)
    print(json.dumps(counts,indent=2),flush=True)


if __name__=='__main__':main()
