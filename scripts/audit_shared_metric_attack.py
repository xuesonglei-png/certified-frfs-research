import json,hashlib,zipfile
import numpy as np
from threadpoolctl import threadpool_limits
from diagnose_shared_metric_attack import ROOT,FOLDER
from diagnose_witness_mixture import FOLDER as MIXTURE,tasks
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.core import gamma


def main():
    threadpool_limits(1);counts={'candidates':0,'concrete_budget_violations':0,'unresolved':0}
    for task in tasks():
        stem=f"{task['group']}-{task['dataset']}-{task['split']}-{task['candidate']}"
        p=json.loads((FOLDER/f'{stem}.json').read_text());assert p['task']==task
        for relative,value in p['fingerprints'].items():assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==value
        prior_path=MIXTURE/f'{stem}.json';assert hashlib.sha256(prior_path.read_bytes()).hexdigest()==p['prior_sha256']
        prior=json.loads(prior_path.read_text())
        if task['group']=='original15':X,y=load_dataset(task['dataset'])
        else:
            with np.load(ROOT/'data/newdata-compact'/f"{task['dataset']}.npz") as data:X,y=data['X'],data['y']
        ri=np.array(prior['reference_indices']);A,_=preprocess_fit(X[ri],X[ri]);Z=np.array(p['fit']['perturbed_table'])
        assert Z.shape==A.shape and np.isfinite(Z).all() and np.all(Z>=np.maximum(0,A-.01)) and np.all(Z<=np.minimum(1,A+.01))
        wf=np.ones(X.shape[1])/np.sqrt(X.shape[1]);wr=np.array(prior['weights'])
        value=gamma(Z,y[ri],weights=wf)-gamma(Z,y[ri],weights=wr)
        clean=gamma(A,y[ri],weights=wf)-gamma(A,y[ri],weights=wr)
        assert abs(value-p['fit']['observed_loss_lower'])<1e-10 and abs(clean-p['clean_loss'])<1e-10
        assert value<=p['upper']+1e-10 and p['proven_budget_violation']==(value>.1)
        assert p['fit']['starts']==8 and p['fit']['steps']==20 and p['fit']['seed']==19201 and p['fit']['eps']==.01
        assert len(p['fit']['runs'])==8 and all(np.all(np.diff(r['trace'])>=0) for r in p['fit']['runs'])
        counts['candidates']+=1;counts['concrete_budget_violations']+=int(value>.1);counts['unresolved']+=int(value<=.1)
    result={**counts,'scope':'independent_actual_gamma_box_and_witness_checks_not_safety_certification'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with zipfile.ZipFile(ROOT/'output/shared-metric-attack-evidence.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in FOLDER.glob('*.json'):archive.write(path,str(path.relative_to(ROOT)))
        for path in (ROOT/'src/certified_frfs').glob('*.py'):archive.write(path,str(path.relative_to(ROOT)))
        for rel in ('scripts/diagnose_shared_metric_attack.py','scripts/audit_shared_metric_attack.py','research/shared-metric-attack-protocol.md','tests/test_shared_metric_attack.py'):
            archive.write(ROOT/rel,rel)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
