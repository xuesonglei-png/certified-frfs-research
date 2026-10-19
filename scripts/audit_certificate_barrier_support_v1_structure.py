"""Independent replay of the finite search graph and prospective gate.

Exact scalar guard/LOO replays occur in the frozen runner at every checked
point. This second audit adds search-trajectory and budget verification; it
does not rerun or silently change the frozen scientific procedure.
"""
import hashlib,json
from pathlib import Path
import numpy as np
from run_repair_relation_selector_v4 import ROOT,NAMES

FOLDER=ROOT/'results/certificate-barrier-support-v1'


def exact_violation(c):
    values=[]
    for certkey,classkey in (('certificate','classwise'),('family_certificate','family_classwise')):
        cert=c[certkey]
        values.append(max(0.,cert['dependency_certificate']['upper_bound']/.1-1))
        values.extend(max(0.,(x+1e-10*(1+abs(x)))/.1-1) for x in c[classkey]['bounds'])
        values.extend(max(0.,(r-k)/max(1,r)) for r,k in zip(cert['required'],cert['retained_counts'],strict=True))
    return [max(values,default=0.),sum(values)]


def audit_one(name):
    r=json.loads((FOLDER/(name+'.json')).read_text());assert r['complete'] and not r['new_outer_scoring']
    assert (r['depths'],r['width'],r['per_depth'])==(2,4,8)
    seed=tuple(r['seed_support']);d=r['d'];records={tuple(x['features']):x for x in r['checks']}
    assert len(records)==r['exact_checks']<=17
    assert len(seed)==int(np.ceil(d/2)) and records[seed]['depth']==0
    visited={seed};beam=[seed];discovered={seed}
    for step in r['proposals']:
        depth=step['depth'];assert step['parents']==[list(fs) for fs in beam]
        pending={}
        for fs in beam:
            for outgoing in fs:
                for incoming in sorted(set(range(d))-set(fs)):
                    child=tuple(sorted((set(fs)-{outgoing})|{incoming}))
                    if child not in visited and child not in pending:pending[child]=fs
        screened={tuple(x['features']):x for x in step['screened']}
        assert set(screened)==set(pending)
        for fs,x in screened.items():
            assert x['parent']==list(pending[fs]);assert len(fs)==len(seed)
            p=x['proxy'];parts=[max(0.,v/.1-1) for v in p['nominal_class_loss']]+p['nominal_protected_deficits']
            assert p['proxy_key']==[max(parts,default=0.),sum(parts),-p['training_LOO_BA']]
        ranked=sorted(pending,key=lambda fs:(*screened[fs]['proxy']['proxy_key'],fs))
        assert [list(fs) for fs in ranked]==[x['features'] for x in step['screened']]
        shortlist=ranked[:8];assert step['exact_shortlist']==[list(fs) for fs in shortlist]
        for fs in shortlist:
            x=records[fs];assert x['depth']==depth and x['parent']==list(pending[fs])
            assert x['proxy']==screened[fs]['proxy'];visited.add(fs)
        discovered.update(pending)
        beam=sorted(visited,key=lambda fs:(*records[fs]['exact_violation'],
                           -records[fs]['proxy']['training_LOO_BA'],fs))[:4]
    assert visited==set(records) and len(discovered)==r['proxy_checks']
    for fs,x in records.items():
        assert x['exact_violation']==exact_violation(x['candidate'])
        assert x['candidate']['features']==list(fs)
    feasible=[fs for fs in visited if records[fs]['candidate']['joint_feasible']]
    chosen=max(feasible,key=lambda fs:(records[fs]['proxy']['training_LOO_BA'],tuple(-a for a in fs))) if feasible else None
    assert r['selected']==(None if chosen is None else records[chosen])
    if chosen is not None:
        assert r['additional_removal']==(r['old_certified_count']-len(chosen))/d
        assert r['selected_minus_old_training_LOO_BA']==records[chosen]['proxy']['training_LOO_BA']-r['old_training_LOO_BA']
    else:assert r['additional_removal']==0 and r['selected_minus_old_training_LOO_BA'] is None
    assert r['strict_count_recovery']==(r['additional_removal']>0)
    assert r['independent_check_counts']['independent_nominal_LOO_replays']==len(records)
    assert r['independent_check_counts']['independent_dual_guard_candidates']==len(records)+1
    return r


def main():
    frozen=json.loads((FOLDER/'source-freeze.json').read_text())['hashes']
    for p,want in frozen.items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==want,p
    rows=[audit_one(name) for name in NAMES]
    recovered=[r for r in rows if r['strict_count_recovery']]
    mean=float(np.mean([r['additional_removal'] for r in rows]))
    expected=len(recovered)>=3 and mean>=.05 and all(r['selected_minus_old_training_LOO_BA']>=-.005-1e-12 for r in recovered)
    summary=json.loads((FOLDER/'summary.json').read_text())
    assert summary['resource_gate']==expected and summary['mean_additional_removal']==mean
    out={'passed':True,'datasets':6,'exact_checked_nodes':sum(r['exact_checks'] for r in rows),
         'resource_gate':expected,'no_new_prediction_or_guard_fitting':True,
         'scope':'independent_search_structure_and_gate_plus_existing_runner_scalar_guard_LOO_replay',
         'auditor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (FOLDER/'structure-audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))


if __name__=='__main__':main()
