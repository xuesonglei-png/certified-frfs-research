"""Independent soft objective/vote/protected-mask and choice reconstruction.

The shared-box dependency bound is rechecked through the production bound;
this is not a second implementation of that bound or a global optimality proof.
"""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_refitted_postmetric_deletion import ROOT,FOLDER,PARENT,fingerprints,digest
from continue_soft_topk_knn_pilot import verify
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.feasible_transport import weight_certificate
from audit_flexible_majority_repair_admission import independent_protected
from audit_classwise_repair_admission import independent_class_certificate
from audit_simplex_source_budget_diagnostic import independent_loo
from independent_soft_topk_objective import independent_objective
from audit_forest_compact_transfer import equivalent


def audit(name):
    path=FOLDER/(name+'.json'); saved=json.loads(path.read_text())
    assert saved['complete'] and not saved['query_labels_used']
    assert saved['fingerprints']==fingerprints(name)
    parent=json.loads((PARENT/(name+'.json')).read_text())
    assert saved['reference_indices']==parent['reference_indices']
    X,y=load_dataset(name); ri=np.array(saved['reference_indices']); labels=y[ri]
    A,_=preprocess_fit(X[ri],X[ri][:1]); mass=np.sqrt(A.shape[1]); full=np.ones(A.shape[1])/mass
    protected=independent_protected(A,labels,full)
    assert len(saved['records'])==len(parent['fit']['checks'])
    for row,old in zip(saved['records'],parent['fit']['checks'],strict=True):
        fit=row['fit']; fs=fit['features']; anchor=np.array(fit['rescaled_weights']); target=np.array(fit['target_weights'])
        assert row['source_candidate_id']==old['id'] and row['original_eligible']==old['eligible']
        assert fs==old['features']; equivalent(anchor.tolist(),old['weights'])
        equivalent(fit['parent_hard_training_LOO_BA'],old['BA_before'])
        solver=fit['refit']; assert solver['max_iter']==1000 and not solver['constrained']
        assert fit['guard_reference']=='original_full' and fit['grid']==[0.,.25,.5,1.]
        for w in (anchor,target): assert w.min()>=0 and abs(w.sum()-mass)<1e-8
        if solver['status']=='local_simplex_soft5_target':
            assert solver['optimizer_success']
            equivalent(independent_objective(A[:,fs],labels,anchor/mass),solver['training_objective_initial'])
            equivalent(independent_objective(A[:,fs],labels,target/mass),solver['training_objective_final'])
        else:
            equivalent(target.tolist(),anchor.tolist())
        assert len(fit['checks'])==4
        for t,check in zip(fit['grid'],fit['checks'],strict=True):
            w=(1-t)*anchor+t*target
            assert check['interpolation']==t; equivalent(check['weights'],w.tolist())
            cert=weight_certificate(A,labels,fs,full,w,protected=protected)
            cls=independent_class_certificate(labels,cert)
            equivalent(check['certificate'],cert); equivalent(check['class_certificate'],cls)
            actual=independent_loo(A,labels,fs,w); equivalent(check['hard_training_LOO_BA'],actual)
            accepted=bool(cert['feasible'] and cls['feasible_all_classes'])
            nonworse=bool(actual>=old['BA_before']-1e-12)
            assert check['accepted']==accepted and check['nonworsening_parent_BA']==nonworse
            assert check['eligible']==(accepted and nonworse)
        eligible=[i for i,c in enumerate(fit['checks']) if c['eligible']]
        selected=min(eligible,key=lambda i:(-round(fit['checks'][i]['hard_training_LOO_BA'],12),fit['checks'][i]['interpolation'])) if eligible else None
        assert fit['selected_id']==selected and fit['checks'][0]['eligible']==old['eligible']
    print(name,'independent refit diagnostic replay complete',len(saved['records']),flush=True)
    return saved


def main():
    threadpool_limits(1); pa=verify(PARENT); names=sorted(pa['artifact_hashes']); assert len(names)==15
    rows=[audit(name) for name in names]
    total=sum(len(r['records']) for r in rows); assert total==117
    expected={'datasets':15,'candidates':total,'checks':4*total,
        'old_eligible':sum(p['original_eligible'] for r in rows for p in r['records']),
        'new_eligible':sum(p['fit']['selected_id'] is not None for r in rows for p in r['records']),
        'newly_eligible':sum(not p['original_eligible'] and p['fit']['selected_id'] is not None for r in rows for p in r['records']),
        'optimizer_failures':sum(not p['fit']['refit']['optimizer_success'] for r in rows for p in r['records'])}
    summary=FOLDER/'complete.json'; assert json.loads(summary.read_text())['counts']==expected
    result={'passed':True,'counts':expected,'artifact_hashes':{n:digest(FOLDER/(n+'.json')) for n in names},
        'summary_sha256':digest(summary),'auditor_sha256':digest(Path(__file__)),
        'goal_achieved':False,'limits':'production_dependency_bound_reused_no_independent_optimizer_globality_or_prediction_claim'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(result,flush=True)


if __name__=='__main__': main()
