"""Independent scalar profiles/transfers/votes and original protected masks.

Production dependency bound is explicitly reused; no full independent bound
derivation or global optimality proof is implied by this replay.
"""
import json
import math
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_atomic_postmetric_deletion import ROOT,FOLDER,PARENT,fingerprints,summary_counts,digest
from continue_soft_topk_knn_pilot import verify
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.feasible_transport import weight_certificate
from audit_flexible_majority_repair_admission import independent_protected
from audit_classwise_repair_admission import independent_class_certificate
from audit_simplex_source_budget_diagnostic import independent_loo
from audit_forest_compact_transfer import equivalent


def receiver_scores(A,features,donor,cache):
    candidates=[]
    for a in features:
        if a==donor: continue
        key=tuple(sorted((a,donor)))
        if key not in cache:
            cache[key]=math.fsum(abs(abs(A[i,a]-A[j,a])-abs(A[i,donor]-A[j,donor]))
                for i in range(len(A)) for j in range(i+1,len(A)))/(len(A)*(len(A)-1)/2)
        candidates.append({'feature':a,'mean_pair_distance_distortion':cache[key]})
    return sorted(candidates,key=lambda p:(round(p['mean_pair_distance_distortion'],12),p['feature']))[:2]


def audit(name):
    path=FOLDER/(name+'.json'); saved=json.loads(path.read_text())
    assert saved['complete'] and not saved['query_labels_used'] and saved['fingerprints']==fingerprints(name)
    source=json.loads((PARENT/(name+'.json')).read_text()); assert source['reference_indices']==saved['reference_indices']
    X,y=load_dataset(name); ri=np.array(source['reference_indices']); A,_=preprocess_fit(X[ri],X[ri][:1]); labels=y[ri]
    full=np.ones(A.shape[1])/np.sqrt(A.shape[1]); protected=independent_protected(A,labels,full); cache={}
    assert len(saved['records'])==len(source['fit']['checks'])
    for record,old in zip(saved['records'],source['fit']['checks'],strict=True):
        fit=record['fit']; fs=old['features_before']; parent=np.array(old['weights_before']); donor=old['removed_feature']
        assert fit['parent_features']==fs and fit['donor']==donor and fit['receiver_pool']==2
        assert record['source_candidate_id']==old['id'] and record['old_eligible']==old['eligible']
        equivalent(record['old_rescaled_BA'],old['hard_training_LOO_BA'])
        equivalent(fit['parent_weights'],parent.tolist()); equivalent(fit['parent_hard_training_LOO_BA'],old['BA_before'])
        equivalent(independent_loo(A,labels,fs,parent),old['BA_before'])
        assert fit['guard_reference']=='original_full'; equivalent(fit['mass'],np.sqrt(A.shape[1]))
        receivers=receiver_scores(A,fs,donor,cache); assert len(fit['checks'])==len(receivers)
        cfs=[a for a in fs if a!=donor]
        for receiver,check in zip(receivers,fit['checks'],strict=True):
            assert check['feature']==receiver['feature'] and check['features']==cfs and donor not in cfs
            equivalent(check['mean_pair_distance_distortion'],receiver['mean_pair_distance_distortion'])
            cw=np.array([parent[fs.index(a)]+(parent[fs.index(donor)] if a==receiver['feature'] else 0) for a in cfs])
            equivalent(check['weights'],cw.tolist()); equivalent(check['donor_mass'],parent[fs.index(donor)])
            assert cw.min()>=0 and abs(cw.sum()-np.sqrt(A.shape[1]))<1e-8
            cert=weight_certificate(A,labels,cfs,full,cw,protected=protected)
            cls=independent_class_certificate(labels,cert); actual=independent_loo(A,labels,cfs,cw)
            equivalent(check['certificate'],cert); equivalent(check['class_certificate'],cls)
            equivalent(check['hard_training_LOO_BA'],actual)
            accepted=bool(cert['feasible'] and cls['feasible_all_classes']); nonworse=bool(actual>=old['BA_before']-1e-12)
            assert check['accepted']==accepted and check['nonworsening_parent_BA']==nonworse
            assert check['eligible']==(accepted and nonworse)
        eligible=[i for i,c in enumerate(fit['checks']) if c['eligible']]
        selected=min(eligible,key=lambda i:(-round(fit['checks'][i]['hard_training_LOO_BA'],12),fit['checks'][i]['feature'])) if eligible else None
        assert fit['selected_id']==selected
    print(name,'independent atomic donor deletion replay complete',len(saved['records']),flush=True)
    return saved


def main():
    threadpool_limits(1); pa=verify(PARENT); names=sorted(pa['artifact_hashes']); assert len(names)==15
    reports=[audit(n) for n in names]; counts=summary_counts(reports)
    summary=FOLDER/'complete.json'; assert json.loads(summary.read_text())['counts']==counts and counts['candidates']==117
    result={'passed':True,'counts':counts,'artifact_hashes':{n:digest(FOLDER/(n+'.json')) for n in names},
        'summary_sha256':digest(summary),'auditor_sha256':digest(Path(__file__)),
        'goal_achieved':False,'limits':'production_shared_dependency_checker_reused_no_predictive_or_minimum_reduct_claim'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(result,flush=True)


if __name__=='__main__': main()
