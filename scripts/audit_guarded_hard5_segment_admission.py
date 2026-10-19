"""Independent full-reference candidate votes, guards, choice and sampling."""
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_guarded_hard5_segment_admission import ROOT,FOLDER,PARENT,digest
from continue_soft_topk_knn_pilot import verify
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.feasible_transport import weight_certificate
from audit_flexible_majority_repair_admission import independent_protected
from audit_classwise_repair_admission import independent_class_certificate
from audit_simplex_source_budget_diagnostic import independent_loo
from audit_forest_compact_transfer import equivalent


def independent_sample(y):
    if len(y)<=96: return list(range(len(y)))
    rng=np.random.default_rng(20261071); retained=[]; leftovers=[]
    for c in sorted(set(y.tolist())):
        group=rng.permutation([i for i in range(len(y)) if y[i]==c]).tolist()
        retained+=group[:2]; leftovers+=group[2:]
    assert len(retained)<=96
    retained+=rng.permutation(leftovers).tolist()[:96-len(retained)]
    return sorted(retained)


def audit(name):
    threadpool_limits(1); path=FOLDER/(name+'.json'); saved=json.loads(path.read_text())
    assert saved['complete'] and not saved['query_labels_or_accuracy_used']
    for rel,sha in saved['fingerprints'].items(): assert digest(ROOT/rel)==sha
    parent=json.loads((PARENT/(name+'.json')).read_text()); pa=verify(PARENT)
    assert pa['artifact_hashes'][name]==digest(PARENT/(name+'.json'))
    assert saved['reference_indices']==parent['reference_indices']
    assert len(saved['records'])==len(parent['records'])
    X,y=load_dataset(name); ri=np.array(saved['reference_indices']); labels=y[ri]
    A,_=preprocess_fit(X[ri],X[ri][:1]); mass=np.sqrt(A.shape[1]); full=np.ones(A.shape[1])/mass
    protected=independent_protected(A,labels,full); sample=independent_sample(labels)
    checks=0; accepted=0
    for record,old in zip(saved['records'],parent['records'],strict=True):
        fs=record['features']; assert fs==old['features']; equivalent(record['anchor'],old['anchor'])
        equivalent(record['input_target'],old['target'])
        anchor=np.array(record['anchor']); target=np.array(record['input_target']); fit=record['fit']
        assert fit['guard_reference']=='original_full' and fit['proposal_indices']==sample
        assert fit['proposal_limit']==96 and fit['proposal_seed']==20261071
        assert len(fit['checks'])<=16 and fit['proposal_info']['max_candidates']==16
        assert [p['t'] for p in fit['checks'][:7]]==[0.,.05,.1,.25,.5,.75,1.]
        seen=set()
        for i,point in enumerate(fit['checks']):
            t=point['t']; assert 0<=t<=1 and t not in seen; seen.add(t)
            if i>=7:
                assert point['origin']=='hard_correctness_interval' and point['left']<t<point['right']
                equivalent(t,point['left']+(point['right']-point['left'])/2)
            else: assert point['origin']=='fixed_grid'
            w=(1-t)*anchor+t*target; equivalent(w.tolist(),point['weights'])
            assert w.min()>=0 and abs(w.sum()-mass)<=1e-8
            cert=weight_certificate(A,labels,fs,full,w,protected=protected)
            classes=independent_class_certificate(labels,cert)
            equivalent(cert,point['certificate']); equivalent(classes,point['class_certificate'])
            assert point['accepted']==bool(cert['feasible'] and classes['feasible_all_classes'])
            equivalent(independent_loo(A,labels,fs,w),point['hard_full_training_LOO_BA'])
            checks+=1; accepted+=point['accepted']
        admitted=[i for i,p in enumerate(fit['checks']) if p['accepted']]
        assert fit['checks'][0]['accepted'] and fit['checks'][6]['accepted']
        chosen=min(admitted,key=lambda i:(-round(fit['checks'][i]['hard_full_training_LOO_BA'],12),abs(1-fit['checks'][i]['t']),i))
        assert fit['selected']==chosen and fit['input_target_accepted']
        best=fit['checks'][chosen]; fixed=max(p['hard_full_training_LOO_BA'] for p in fit['checks'][:7] if p['accepted'])
        equivalent(best['weights'],record['target']); equivalent(best['weights'],fit['weights'])
        equivalent(best['hard_full_training_LOO_BA'],fit['final_hard_LOO_BA'])
        equivalent(fixed,fit['fixed_best_hard_LOO_BA'])
        input_ba=fit['checks'][6]['hard_full_training_LOO_BA']; equivalent(input_ba,fit['input_target_hard_LOO_BA'])
        equivalent(input_ba,old['fit']['final_hard_LOO_BA'])
        equivalent(record['gain_over_input_target_pp'],100*(best['hard_full_training_LOO_BA']-input_ba))
        equivalent(record['gain_over_fixed_grid_pp'],100*(best['hard_full_training_LOO_BA']-fixed))
        assert best['hard_full_training_LOO_BA']>=fixed-1e-12
    print(name,'COMPLETE',len(saved['records']),'supports',checks,'scalar candidate replays',flush=True)
    return {'dataset':name,'supports':len(saved['records']),'checks':checks,'accepted':accepted,'sha256':digest(path)}


if __name__=='__main__':
    summary=json.loads((FOLDER/'complete.json').read_text()); reports=[]
    with ProcessPoolExecutor(max_workers=1) as executor:
        reports=[f.result() for f in as_completed([executor.submit(audit,row['dataset']) for row in summary['rows']])]
    assert len(reports)==15 and sum(r['supports'] for r in reports)==86
    for row in summary['rows']:
        saved=json.loads((FOLDER/(row['dataset']+'.json')).read_text()); records=saved['records']
        equivalent(row['mean_gain_over_parent_pp'],float(np.mean([r['gain_over_input_target_pp'] for r in records])))
        equivalent(row['mean_gain_over_fixed_pp'],float(np.mean([r['gain_over_fixed_grid_pp'] for r in records])))
        assert row['supports']==len(records)
        assert row['improved_over_parent']==sum(r['gain_over_input_target_pp']>1e-9 for r in records)
        assert row['improved_over_fixed']==sum(r['gain_over_fixed_grid_pp']>1e-9 for r in records)
        assert row['candidate_checks']==sum(len(r['fit']['checks']) for r in records)
        assert row['guard_rejections']==sum(not p['accepted'] for r in records for p in r['fit']['checks'])
    equivalent(summary['dataset_balanced_gain_over_parent_pp'],float(np.mean([r['mean_gain_over_parent_pp'] for r in summary['rows']])))
    equivalent(summary['dataset_balanced_gain_over_fixed_pp'],float(np.mean([r['mean_gain_over_fixed_pp'] for r in summary['rows']])))
    assert summary['datasets']==15 and summary['supports']==86
    result={'passed':True,'counts':{'datasets':15,'supports':86,'scalar_votes':sum(r['checks'] for r in reports),
        'guard_replays':sum(r['checks'] for r in reports),'accepted_candidates':sum(r['accepted'] for r in reports)},
        'artifact_hashes':{r['dataset']:r['sha256'] for r in reports},'summary_sha256':digest(FOLDER/'complete.json'),
        'auditor_sha256':digest(Path(__file__)),'goal_achieved':False,
        'scope':'finite_candidates_full_reference_votes_guards_sampling_choice_not_exhaustive_continuous_search_or_prediction'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(result,flush=True)
