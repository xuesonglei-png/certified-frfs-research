"""All declared low-dimensional supports; checkpointed training-only diagnostic."""
import hashlib,itertools,json,math,time
from collections import Counter
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.relation_support_selector_v1 import _candidate
from certified_frfs.majority_metric import MajorityTerms
from half_support_enumeration_v1 import NominalWitnesses
from audit_selective_support_selector_v3 import check_candidate
from run_certificate_barrier_support_v1 import independent_loo
from run_repair_relation_selector_v4 import ROOT,OUT,NAMES

FOLDER=ROOT/'results/half-support-enumeration-v1'


def freeze():
    files=list((ROOT/'src/certified_frfs').glob('*.py'))
    files += [ROOT/'scripts'/s for s in ('half_support_enumeration_v1.py',
       'run_half_support_enumeration_v1.py','audit_half_support_enumeration_v1.py',
       'certificate_barrier_support_v1.py','run_certificate_barrier_support_v1.py',
       'audit_selective_support_selector_v3.py','audit_relation_support_selector_v1.py',
       'run_repair_relation_selector_v4.py')]
    files += [ROOT/'tests/test_half_support_enumeration_v1.py',
              ROOT/'research/half-support-enumeration-v1-protocol.md']
    for n in NAMES:files += [OUT/(n+'.json'),ROOT/'data/processed'/(n+'.npz'),
                             ROOT/'results/certificate-barrier-support-v1'/(n+'.json')]
    hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    FOLDER.mkdir(exist_ok=True);p=FOLDER/'source-freeze.json'
    if p.exists():assert json.loads(p.read_text())['hashes']==hashes
    else:p.write_text(json.dumps({'hashes':hashes,'created_unix':time.time(),
                                 'tests_passed_before_freeze':5,'new_outer_scoring':False},indent=2))
    return hashes


def save(path,result):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result,separators=(',',':')));tmp.replace(path)


def training(name):
    parent=json.loads((OUT/(name+'.json')).read_text());row=parent['rows'][0];b=row['fit']['inner'][0]
    X,y=load_dataset(name);outer=np.asarray(row['train_indices']);inner=np.asarray(b['fit_indices'])
    A,_=preprocess_fit(X[outer[inner]],X[outer[inner]])
    return A,y[outer[inner]],outer.tolist(),inner.tolist()


def run(name):
    hashes=freeze();path=FOLDER/(name+'.json');A,y,outer,inner=training(name);d=A.shape[1];m=math.ceil(d/2)
    result=json.loads(path.read_text()) if path.exists() else {'dataset':name,'d':d,'m':m,
       'declared_combinations':math.comb(d,m),'available':d<=13,'complete':False,
       'enclosing_training_indices':outer,'inner_training_indices':inner,
       'source_hashes':hashes,'records':[],'elapsed_seconds':0.,'new_outer_scoring':False}
    assert result['source_hashes']==hashes
    if result['complete']:return result
    if d>13:
        result.update({'complete':True,'unavailable_reason':'dimension_above_prospective_13_limit',
                       'finite_recipe_existence':'unavailable_no_partial_sampling'})
        save(path,result);print(name,'explicit unavailable',math.comb(d,m),flush=True);return result
    start=time.perf_counter();ref=np.ones(d)/np.sqrt(d)
    protected=MajorityTerms(A,y,eps=.01,weights=ref).bounds(range(d),allowed_enemies=2)>1e-10
    witness=NominalWitnesses(A,y,ref,protected);records=result['records'];counts=Counter(result.get('independent_checks',{}))
    previous=result['elapsed_seconds'];pruned_types={q for r in records if r['status']=='nominal_witness_rejected' for q in r['nominal']['pruning_reasons']}
    for index,fs in enumerate(itertools.combinations(range(d),m)):
        if index<len(records):assert records[index]['features']==list(fs);continue
        nominal=witness(fs);record={'index':index,'features':list(fs),'nominal':nominal}
        reasons=nominal['pruning_reasons']
        if reasons:
            record['status']='nominal_witness_rejected'
            # First example of each distinct prune type gets the unmodified
            # production certifier as an implementation sanity check.
            if any(q not in pruned_types for q in reasons):
                c=_candidate(A,y,ref,list(fs),len(fs));assert not c['joint_feasible']
                record['representative_original_guard']=c;pruned_types.update(reasons)
        else:
            c=_candidate(A,y,ref,list(fs),len(fs));record['candidate']=c
            np.testing.assert_allclose(c['weights'],nominal['weights'],rtol=0,atol=1e-14)
            record['status']='certified' if c['joint_feasible'] else 'original_guard_rejected'
            if c['joint_feasible']:
                check_candidate(A,y,ref,list(fs),c,counts)
                ba=independent_loo(A,y,list(fs),c['weights'])
                assert abs(ba-nominal['training_LOO_BA'])<1e-12
                counts['accepted_independent_scalar_LOO_replays']+=1
        records.append(record)
        if len(records)%25==0:
            result.update({'elapsed_seconds':previous+time.perf_counter()-start,'independent_checks':dict(counts)})
            save(path,result);print(name,len(records),'/',math.comb(d,m),dict(Counter(r['status'] for r in records)),flush=True)
    local=json.loads((ROOT/'results/certificate-barrier-support-v1'/(name+'.json')).read_text())
    local_checked={tuple(x['features']) for x in local['checks']}
    certified=[r for r in records if r['status']=='certified']
    statuscounts=dict(Counter(r['status'] for r in records));reasoncounts=dict(Counter(q for r in records for q in r['nominal']['pruning_reasons']))
    result.update({'complete':True,'status_counts':statuscounts,'prune_reason_counts':reasoncounts,
        'elapsed_seconds':previous+time.perf_counter()-start,'independent_checks':dict(counts),
        'finite_recipe_existence':bool(certified),'certified_supports':[r['features'] for r in certified],
        'certified_supports_missed_by_local_exact_checks':[r['features'] for r in certified if tuple(r['features']) not in local_checked],
        'local_exact_check_count':len(local_checked),'local_found_count':sum(x['candidate']['joint_feasible'] for x in local['checks']),
        'nominal_training_LOO_BA_range_certified':None if not certified else
            [min(r['nominal']['training_LOO_BA'] for r in certified),max(r['nominal']['training_LOO_BA'] for r in certified)]})
    assert freeze()==hashes;save(path,result);print(name,'COMPLETE',statuscounts,flush=True);return result


def main():
    threadpool_limits(1);rows=[run(n) for n in NAMES]
    summary={'complete':True,'datasets':6,'available_datasets':sum(r['available'] for r in rows),
       'enumerated_combinations':sum(len(r['records']) for r in rows),'new_outer_scoring':False,
       'scope':'finite_uniform_capped_transfer_recipe_training_certifier_existence_not_robust_impossibility_or_test_risk',
       'rows':[{k:r[k] for k in ('dataset','d','m','declared_combinations','available','finite_recipe_existence')}
         |({k:r[k] for k in ('status_counts','prune_reason_counts','local_exact_check_count','local_found_count','elapsed_seconds')}
          |{'certified_supports_missed_by_local':len(r['certified_supports_missed_by_local_exact_checks'])} if r['available'] else {}) for r in rows]}
    save(FOLDER/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
