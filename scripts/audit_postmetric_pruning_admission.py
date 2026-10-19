"""Independent deletion pools, scalar votes, guard sources and choices."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_postmetric_pruning_admission import ROOT,FOLDER,PARENT,digest
from continue_soft_topk_knn_pilot import verify
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.feasible_transport import weight_certificate
from audit_flexible_majority_repair_admission import independent_protected
from audit_classwise_repair_admission import independent_class_certificate
from audit_simplex_source_budget_diagnostic import independent_loo
from audit_forest_compact_transfer import equivalent


def audit(name):
    path=FOLDER/(name+'.json'); saved=json.loads(path.read_text())
    assert saved['complete'] and not saved['query_labels_or_accuracy_used']
    for rel,sha in saved['fingerprints'].items(): assert digest(ROOT/rel)==sha
    pa=verify(PARENT); assert pa['artifact_hashes'][name]==digest(PARENT/(name+'.json'))
    parent=json.loads((PARENT/(name+'.json')).read_text()); metric=parent['outer']['results']['proposed']['metric']
    assert saved['reference_indices']==parent['reference_indices'] and saved['initial_features']==metric['features']
    equivalent(saved['initial_weights'],metric['weights'])
    X,y=load_dataset(name); ri=np.array(saved['reference_indices']); labels=y[ri]
    A,_=preprocess_fit(X[ri],X[ri][:1]); mass=np.sqrt(A.shape[1]); full=np.ones(A.shape[1])/mass
    protected=independent_protected(A,labels,full); fit=saved['fit']
    assert fit['fraction']==.5 and fit['pool']==4 and fit['max_checks']==48 and fit['guard_reference']=='original_full'
    fs=saved['initial_features'].copy(); w=np.array(saved['initial_weights']); ba=independent_loo(A,labels,fs,w)
    equivalent(ba,fit['initial_hard_LOO_BA']); assert fit['target_count']==max(1,int(np.ceil(.5*len(fs))))
    cursor=0; chosen=[]; checks=fit['checks']; endpoint=fit['original_initial_guard']
    cert=weight_certificate(A,labels,fs,full,w,protected=protected); classes=independent_class_certificate(labels,cert)
    equivalent(cert,endpoint['certificate']); equivalent(classes,endpoint['class_certificate'])
    assert endpoint['accepted'] and cert['feasible'] and classes['feasible_all_classes']
    while cursor<len(checks):
        importance=[w[i]*(max(A[:,a])-min(A[:,a])) for i,a in enumerate(fs)]
        order=sorted(range(len(fs)),key=lambda i:(importance[i],w[i],fs[i]))[:4]
        batch=[]
        for drop in order:
            if cursor>=48: break
            kept=[i for i in range(len(fs)) if i!=drop]; candidate_fs=[fs[i] for i in kept]
            candidate_w=w[kept].copy()
            if candidate_w.sum()<=0: continue
            candidate_w*=mass/candidate_w.sum()
            point=checks[cursor]; assert point['id']==cursor and point['features_before']==fs
            equivalent(point['weights_before'],w.tolist()); equivalent(point['BA_before'],ba)
            assert point['removed_feature']==fs[drop] and point['features']==candidate_fs
            equivalent(point['weights'],candidate_w.tolist())
            fresh=weight_certificate(A,labels,candidate_fs,full,candidate_w,protected=protected)
            cls=independent_class_certificate(labels,fresh); actual=independent_loo(A,labels,candidate_fs,candidate_w)
            equivalent(fresh,point['certificate']); equivalent(cls,point['class_certificate'])
            equivalent(actual,point['hard_training_LOO_BA'])
            accepted=bool(fresh['feasible'] and cls['feasible_all_classes']); nonworse=bool(actual>=ba-1e-12)
            assert point['accepted']==accepted and point['nonworsening_training_BA']==nonworse
            assert point['eligible']==(accepted and nonworse)
            batch.append(point); cursor+=1
        eligible=[p for p in batch if p['eligible']]
        if not eligible:
            assert cursor==len(checks); break
        best=min(eligible,key=lambda p:(-round(p['hard_training_LOO_BA'],12),p['removed_feature']))
        chosen.append(best['id']); fs=best['features'].copy(); w=np.array(best['weights']); ba=best['hard_training_LOO_BA']
        endpoint={k:best[k] for k in ('certificate','class_certificate','accepted')}
    assert fit['chosen_ids']==chosen and fit['features']==saved['features']==fs
    equivalent(fit['weights'],w.tolist()); equivalent(saved['weights'],w.tolist())
    equivalent(fit['final_hard_LOO_BA'],ba); equivalent(endpoint,fit['endpoint_guard'])
    assert ba>=fit['initial_hard_LOO_BA']-1e-12 and endpoint['accepted'] and len(checks)<=48
    expected='target_reached' if len(fs)<=fit['target_count'] else ('guard_budget_exhausted' if len(checks)>=48 else 'no_acceptable_pooled_deletion')
    assert fit['status']==expected
    print(name,'COMPLETE',len(checks),'independent candidate replays',flush=True)
    return {'dataset':name,'sha256':digest(path),'checks':len(checks),
        'row':{'dataset':name,'d':saved['d'],'initial_count':len(saved['initial_features']),'final_count':len(fs),
        'status':fit['status'],'checks':len(checks),'initial_hard_LOO_BA':fit['initial_hard_LOO_BA'],'final_hard_LOO_BA':ba}}


def main():
    threadpool_limits(1); summary=json.loads((FOLDER/'complete.json').read_text())
    reports=[audit(row['dataset']) for row in summary['rows']]; equivalent([r['row'] for r in reports],summary['rows'])
    rows=summary['rows']; assert len(rows)==summary['datasets']==15
    assert summary['newly_pruned_datasets']==sum(r['final_count']<r['initial_count'] for r in rows)
    equivalent(summary['mean_initial_removed'],float(np.mean([1-r['initial_count']/r['d'] for r in rows])))
    equivalent(summary['mean_final_removed'],float(np.mean([1-r['final_count']/r['d'] for r in rows])))
    equivalent(summary['dataset_balanced_hard_training_gain_pp'],float(np.mean([100*(r['final_hard_LOO_BA']-r['initial_hard_LOO_BA']) for r in rows])))
    result={'passed':True,'counts':{'datasets':15,'scalar_candidate_votes':sum(r['checks'] for r in reports)},
        'artifact_hashes':{r['dataset']:r['sha256'] for r in reports},'summary_sha256':digest(FOLDER/'complete.json'),
        'auditor_sha256':digest(Path(__file__)),'goal_achieved':False,
        'scope':'finite_training_pruning_votes_guards_choices_and_budgets_not_query_prediction'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(result,flush=True)


if __name__=='__main__': main()
