"""Independent scenario-loss, box, metric and final-certificate verification."""
import json,hashlib,zipfile
import numpy as np
from threadpoolctl import threadpool_limits
from diagnose_scenario_repair import ROOT,FOLDER
from diagnose_witness_mixture import tasks
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.core import gamma
from certified_frfs.feasible_transport import weight_certificate


def main():
    threadpool_limits(1);counts={'candidates':0,'successful_LPs':0,'scenario_loss_checks':0,'fresh_certificates':0,'attack_witnesses':0}
    for task in tasks():
        stem=f"{task['group']}-{task['dataset']}-{task['split']}-{task['candidate']}"
        p=json.loads((FOLDER/f'{stem}.json').read_text());assert p['task']==task
        for relative,value in p['fingerprints'].items():assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==value,relative
        prior=json.loads((ROOT/'results/witness-mixture-diagnostic'/f'{stem}.json').read_text())
        attack=json.loads((ROOT/'results/shared-metric-attack'/f'{stem}.json').read_text())
        if task['group']=='original15':X,y=load_dataset(task['dataset'])
        else:
            with np.load(ROOT/'data/newdata-compact'/f"{task['dataset']}.npz") as data:X,y=data['X'],data['y']
        ri=np.array(prior['reference_indices']);A,_=preprocess_fit(X[ri],X[ri]);labels=y[ri]
        full=np.ones(X.shape[1])/np.sqrt(X.shape[1]);mass=full.sum();support=set(prior['features'])
        scenarios=[A,np.array(attack['fit']['perturbed_table'])];current=np.array(prior['weights']);fit=p['fit']
        for iteration,record in enumerate(fit['trace']):
            assert record['iteration']==iteration and iteration<3 and record['lp']['eta']==.09
            lp=record['lp']
            if not lp['success']:
                assert iteration==len(fit['trace'])-1
                break
            current=np.array(lp['weights']);fs=np.flatnonzero(current>0).tolist()
            assert set(fs)<=support and current.min()>=0 and current.max()<=2*mass/X.shape[1]+1e-9 and abs(current.sum()-mass)<1e-8
            losses=[gamma(Z,labels,weights=full)-gamma(Z,labels,weights=current) for Z in scenarios]
            assert max(losses)<=.09+1e-9 and np.allclose(losses,lp['scenario_losses'],atol=1e-10,rtol=0)
            counts['scenario_loss_checks']+=len(losses);counts['successful_LPs']+=1
            certificate=weight_certificate(A,labels,fs,full,current[fs]);assert certificate==record['certificate']
            counts['fresh_certificates']+=1
            Z=np.array(record['attack']['perturbed_table'])
            assert Z.shape==A.shape and np.all(Z>=np.maximum(0,A-.01)) and np.all(Z<=np.minimum(1,A+.01))
            actual=gamma(Z,labels,weights=full)-gamma(Z,labels,weights=current)
            assert abs(actual-record['attack']['observed_loss_lower'])<1e-10
            counts['attack_witnesses']+=1
            if certificate['feasible']:
                assert iteration==len(fit['trace'])-1
                break
            scenarios.append(Z)
        assert np.allclose(current,fit['weights'],atol=0,rtol=0)
        assert fit['features']==np.flatnonzero(current>0).tolist()
        assert len(scenarios)==len(fit['scenarios']) and np.allclose(scenarios,fit['scenarios'],atol=0,rtol=0)
        assert fit['certified']==bool(fit['trace'][-1].get('certificate',{}).get('feasible',False))
        counts['candidates']+=1
    result={**counts,'scope':'independent_scenario_losses_attack_tables_caps_and_certificates_not_LP_optimality_or_global_safety'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with zipfile.ZipFile(ROOT/'output/scenario-repair-evidence.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in FOLDER.glob('*.json'):archive.write(path,str(path.relative_to(ROOT)))
        for path in (ROOT/'src/certified_frfs').glob('*.py'):archive.write(path,str(path.relative_to(ROOT)))
        for rel in ('scripts/diagnose_scenario_repair.py','scripts/audit_scenario_repair.py','tests/test_scenario_repair.py','research/scenario-repair-protocol.md'):
            archive.write(ROOT/rel,rel)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
