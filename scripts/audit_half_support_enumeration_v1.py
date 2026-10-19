"""Coverage, nominal pruning and certified-endpoint scalar audit, no selection."""
import hashlib,itertools,json,math
from collections import Counter
import numpy as np
from half_support_enumeration_v1 import reasons_of
from certificate_barrier_support_v1 import metric_weights
from run_half_support_enumeration_v1 import FOLDER,training
from run_repair_relation_selector_v4 import ROOT,NAMES
from audit_relation_support_selector_v1 import independent_protected
from audit_selective_support_selector_v3 import check_candidate
from run_certificate_barrier_support_v1 import independent_loo


def main():
    hashes=json.loads((FOLDER/'source-freeze.json').read_text())['hashes']
    for p,want in hashes.items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==want,p
    audited=Counter();rows=[]
    for name in NAMES:
        r=json.loads((FOLDER/(name+'.json')).read_text());assert r['complete'] and not r['new_outer_scoring']
        A,y,outer,inner=training(name);d=A.shape[1];m=math.ceil(d/2);ref=np.ones(d)/np.sqrt(d)
        assert r['enclosing_training_indices']==outer and r['inner_training_indices']==inner
        assert (r['d'],r['m'],r['declared_combinations'])==(d,m,math.comb(d,m))
        if d>13:
            assert not r['available'] and not r['records'] and r['finite_recipe_existence']=='unavailable_no_partial_sampling'
            rows.append({'dataset':name,'explicit_unavailable':True});continue
        assert r['available'] and [x['features'] for x in r['records']]==[list(fs) for fs in itertools.combinations(range(d),m)]
        classes=np.unique(y);protected=independent_protected(A,y,ref);cross=y[:,None]!=y[None,:]
        Dfull=np.zeros((len(y),len(y)))
        for a,v in enumerate(ref):Dfull+=v*abs(A[:,a,None]-A[None,:,a])
        fullgamma=np.minimum(1.,np.min(np.where(cross,Dfull,np.inf),axis=1));accepted=[]
        for index,x in enumerate(r['records']):
            assert x['index']==index;p=x['nominal'];fs=x['features'];w=metric_weights(A,fs,ref)
            np.testing.assert_allclose(p['weights'],w,rtol=0,atol=1e-14)
            D=np.zeros_like(Dfull)
            for a,v in zip(fs,w,strict=True):D+=v*abs(A[:,a,None]-A[None,:,a])
            gamma=np.minimum(1.,np.min(np.where(cross,D,np.inf),axis=1));signed=fullgamma-gamma;positive=np.maximum(0.,signed)
            np.testing.assert_allclose(p['positive_class_loss'],[positive[y==c].mean() for c in classes],rtol=0,atol=2e-14)
            np.testing.assert_allclose(p['signed_class_loss'],[signed[y==c].mean() for c in classes],rtol=0,atol=2e-14)
            assert abs(p['positive_global_loss']-positive.mean())<2e-14
            np.fill_diagonal(D,np.inf);possible=[]
            for i in range(len(y)):
                radius=sorted(D[i])[4];tol=1e-10*(1+abs(radius))
                strict=[j for j in range(len(y)) if D[i,j]<radius-tol]
                boundary=[j for j in range(len(y)) if radius-tol<=D[i,j]<=radius+tol]
                friends=sum(y[j]==y[i] for j in strict)+min(sum(y[j]==y[i] for j in boundary),5-len(strict))
                assert friends==p['max_sameclass_in_nominal_top5'][i];possible.append(friends>=3)
            possible=np.asarray(possible)
            req=[int(np.ceil(.9*sum((y==c)&protected)-1e-12)) for c in classes]
            counts=[int(sum(possible&(y==c)&protected)) for c in classes]
            assert req==p['required_counts'] and counts==p['possible_majority_counts']
            assert p['pruning_reasons']==reasons_of(p)
            if p['pruning_reasons']:
                assert x['status']=='nominal_witness_rejected' and 'candidate' not in x
                if 'representative_original_guard' in x:assert not x['representative_original_guard']['joint_feasible']
                audited['nominal_pruning_records']+=1
            else:
                c=x['candidate'];assert x['status']==('certified' if c['joint_feasible'] else 'original_guard_rejected')
                if c['joint_feasible']:
                    check_candidate(A,y,ref,fs,c,audited);ba=independent_loo(A,y,fs,c['weights'])
                    assert abs(ba-p['training_LOO_BA'])<1e-12;accepted.append(fs);audited['accepted_scalar_LOO_replays']+=1
                else:audited['original_guard_rejected_retained']+=1
            audited['nominal_witness_records']+=1
        assert r['status_counts']==dict(Counter(x['status'] for x in r['records']))
        assert r['prune_reason_counts']==dict(Counter(q for x in r['records'] for q in x['nominal']['pruning_reasons']))
        assert r['certified_supports']==accepted and r['finite_recipe_existence']==bool(accepted)
        local=json.loads((ROOT/'results/certificate-barrier-support-v1'/(name+'.json')).read_text())
        checked={tuple(x['features']) for x in local['checks']}
        assert r['certified_supports_missed_by_local_exact_checks']==[fs for fs in accepted if tuple(fs) not in checked]
        rows.append({'dataset':name,'combinations':len(r['records']),'certified':len(accepted),
                     'missed_by_local_exact_checks':len(r['certified_supports_missed_by_local_exact_checks'])})
        print(name,'coverage/nominal/endpoints audit pass',rows[-1],flush=True)
    report={'passed':True,'counts':dict(audited),'rows':rows,'new_outer_scoring':False,
        'scope':'complete_nominal_prune_coverage_and_accepted_scalar_dual_guard_replay_not_interval_arithmetic',
        'auditor_sha256':hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest()}
    (FOLDER/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
