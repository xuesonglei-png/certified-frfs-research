"""Replay the seeded search and independently check final certificates."""
import json,hashlib,zipfile
import numpy as np
from threadpoolctl import threadpool_limits
from diagnose_direct_bound_search import ROOT,FOLDER
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.bound_direct import direct_bound_search
from certified_frfs.feasible_transport import weight_certificate


def main():
    threadpool_limits(1);counts={'candidates':0,'proposal_evaluations':0,'new_certified':0}
    for path in sorted(FOLDER.glob('*.json')):
        if path.name.startswith('complete'):continue
        p=json.loads(path.read_text());task=p['task'];stem=path.stem
        for relative,value in p['fingerprints'].items():assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==value
        prior=json.loads((ROOT/'results/witness-mixture-diagnostic'/f'{stem}.json').read_text())
        scenario=json.loads((ROOT/'results/scenario-repair-diagnostic'/f'{stem}.json').read_text())
        if task['group']=='original15':X,y=load_dataset(task['dataset'])
        else:
            with np.load(ROOT/'data/newdata-compact'/f"{task['dataset']}.npz") as data:X,y=data['X'],data['y']
        ri=np.array(prior['reference_indices']);A,_=preprocess_fit(X[ri],X[ri]);full=np.ones(X.shape[1])/np.sqrt(X.shape[1]);fs=prior['features']
        rebuilt=direct_bound_search(A,y[ri],fs,full,np.array(scenario['fit']['weights'])[fs])
        assert rebuilt==p['fit'];w=np.array(rebuilt['weights'])
        assert w.min()>=0 and abs(w.sum()-full.sum())<1e-8 and w.max()<=2*full.sum()/X.shape[1]+1e-9
        assert weight_certificate(A,y[ri],fs,full,w)==rebuilt['certificate']
        counts['candidates']+=1;counts['proposal_evaluations']+=sum(t['proposals'] for t in rebuilt['trace'])
        counts['new_certified']+=int(rebuilt['certificate']['feasible']);print(stem,'search replay audited',flush=True)
    (FOLDER/'complete-audit.json').write_text(json.dumps(counts,indent=2),encoding='utf-8')
    with zipfile.ZipFile(ROOT/'output/direct-bound-search-evidence.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in FOLDER.glob('*.json'):archive.write(path,str(path.relative_to(ROOT)))
        for path in (ROOT/'src/certified_frfs').glob('*.py'):archive.write(path,str(path.relative_to(ROOT)))
        for rel in ('scripts/diagnose_direct_bound_search.py','scripts/audit_direct_bound_search.py','research/direct-bound-search-protocol.md','tests/test_bound_direct.py'):
            archive.write(ROOT/rel,rel)
    print(json.dumps(counts,indent=2),flush=True)


if __name__=='__main__':main()
