"""Frozen V4 source/targets/actual endpoints/all old attacks, no solves.

This checks model embeddings and backend configuration, not saved dual
multipliers (unavailable). Complete screen and every original-positive failed
class must be present; no favorable-admission-only audit is allowed.
"""
import argparse
from collections import Counter
import hashlib,json,time,traceback
from pathlib import Path
import sys
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from audit_classwise_shared_attack_all_rows_v1 import scalar_memberships
from audit_shared_graph_endpoints_independent_v1 import enemy_cuts_from_saved
from independent_positive_graph_embedding_v4 import TrueEmbedding

FOLDER=ROOT/'results/shared-graph-dependency-v4'
NAMES=('banknote','glass','heart_statlog','ionosphere','seeds','wdbc')


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def validate_backend(record):
    assert record['backend']=='highspy' and record['version']=='1.15.1'
    settings=record['settings'];assert settings=={
        'output_flag':False,'threads':1,'random_seed':0,'presolve':'off','mip_rel_gap':0.,'mip_abs_gap':0.,
        'time_limit':8.,'mip_feasibility_tolerance':1e-8,'primal_feasibility_tolerance':1e-8,'dual_feasibility_tolerance':1e-8}
    assert record['matrix_roundtrip_max_difference']<=1e-12 and record['roundtrip_allowance']==1e-12
    assert record['original_matrix_unchanged_except_roundoff']
    assert record['model_import_status'] in ('HighsStatus.kOk','HighsStatus.kWarning')


def audit_dataset(name,hashes,expected_sha=None):
    path=FOLDER/(name+'.json');saved=json.loads(path.read_text());assert saved['complete']
    if expected_sha is not None:assert digest(path)==expected_sha
    assert saved['source_hashes']==hashes and not saved['new_classifier_predictions']
    parent=json.loads((ROOT/'results/repair-relation-selector-v4'/(name+'.json')).read_text())
    attacks=json.loads((ROOT/'results/classwise-shared-attack-v1'/(name+'.json')).read_text())
    panel=parent['rows'][0];ids=np.asarray(panel['train_indices'])[np.asarray(panel['fit']['inner'][0]['fit_indices'])]
    X,y=load_dataset(name);A,_=preprocess_fit(X[ids],X[ids]);y=y[ids];classes,counts=np.unique(y,return_counts=True)
    assert len(saved['rows'])==len(attacks['reports']);tally=Counter();status=Counter();details=[];cache={};maxres=0.;new=[]
    for ci,(row,source) in enumerate(zip(saved['rows'],attacks['reports'],strict=True)):
        assert row==json.loads((FOLDER/name/(str(ci)+'.json')).read_text())
        assert row['source_hashes']==hashes and row['candidate_index']==ci and row['complete']
        assert not row['new_classifier_predictions'] and not row['validation_or_outer_observations_used']
        c=source['candidate'];fs=c['features'];wr=np.asarray(c['weights']);wide=np.zeros(A.shape[1]);wide[fs]=wr
        for key in ('family','stratum'):assert row[key]==source[key]
        assert row['features']==fs and row['weights']==c['weights'] and row['known_violation']==source['class_violation_witness']
        assert row['legacy_joint_feasible']==c['joint_feasible']
        assert len(row['guards'])==2
        for side,ck,cl,g in zip(('original','family'),('certificate','family_certificate'),
                               ('classwise','family_classwise'),row['guards'],strict=True):
            wf=np.asarray(c[ck]['dependency_certificate'].get('full_weights',c['weights']))
            np.testing.assert_array_equal(wf,g['full_weights']);assert g['side']==side
            identity=tuple(wf),tuple(fs),tuple(wr)
            if identity in cache:
                assert g['reused_identical_metric'] and {k:v for k,v in g.items() if k not in ('side','reused_identical_metric')}==cache[identity]
                tally['reused_guards']+=1
            else:
                assert not g['reused_identical_metric'];cache[identity]={k:v for k,v in g.items() if k not in ('side','reused_identical_metric')}
                signed=g['signed_report'];objects=signed['signed_object_report']['signed_object_upper_bounds']
                cuts=enemy_cuts_from_saved(A,wide,signed)
                np.testing.assert_array_equal(g['signed_class_bounds'],signed['class_certificate']['upper_bounds'])
                for target,positive,records,oldbounds,newbounds in (
                    ('signed',False,g['classes'],g['signed_class_bounds'],g['new_class_bounds']),
                    ('positive',True,g['positive_classes'],c[cl]['bounds'],g['positive_class_bounds'])):
                    assert len(records)==len(classes)==len(oldbounds)==len(newbounds)
                    for label,oldbound,newbound,record in zip(classes,oldbounds,newbounds,records,strict=True):
                        assert record['target_class']==label and record['old_upper_bound']==oldbound
                        assert record['skipped_existing_feasible']==(oldbound<=.1)
                        tally[target+'_classes']+=1
                        if record['skipped_existing_feasible']:
                            assert record['new_upper_bound']==oldbound==newbound;continue
                        fit=record['solve'];tally[target+'_solver_calls']+=1;status[str(fit['solver_status'])]+=1
                        # Unaccepted statuses cannot provide a usable certification upper.
                        assert fit['solver_status'] in (0,1),(name,ci,target,fit['solver_status'])
                        assert fit['solver_message'] in ('Optimal','Time limit reached')
                        validate_backend(fit['backend_record'])
                        expected_target='mean_positive_object_loss' if positive else 'positive_class_mean_loss'
                        assert fit['target']==expected_target and fit['eps']==.01 and not fit['exact_abs']
                        upper=fit['solver_upper_bound'];allowance=1e-7*(1+abs(upper))
                        assert np.isfinite(upper) and fit['numerical_allowance']==allowance
                        if not fit['solver_bound_available']:assert upper==1.
                        np.testing.assert_allclose(fit['upper_bound'],float(np.clip(upper+allowance,0.,1.)),atol=1e-12,rtol=0)
                        assert record['new_upper_bound']==newbound==min(oldbound,fit['upper_bound'])
                        validator=TrueEmbedding(A,y,wf,wide,label,positive=positive,object_upper=objects,enemy_upper=cuts)
                        for key,value in validator.meta.items():assert fit[key]==value,(name,ci,target,key)
                        nominal,res=validator.table(A);maxres=max(maxres,res)
                        endpoint,res=validator.table(fit['perturbed_table']);maxres=max(maxres,res)
                        np.testing.assert_allclose(nominal,fit['nominal_target_loss'],atol=1e-10,rtol=0)
                        np.testing.assert_allclose(endpoint,fit['observed_target_loss_lower'],atol=1e-10,rtol=0)
                        assert endpoint+1e-12>=nominal and endpoint<=upper+allowance+1e-10
                        tally[target+'_endpoint_embeddings']+=2
                        for previous in source['attacks']:
                            if previous['fit']['target_class']==label and np.array_equal(previous['full_weights'],wf):
                                actual,res=validator.table(previous['fit']['perturbed_table']);maxres=max(maxres,res)
                                if not positive:np.testing.assert_allclose(actual,previous['fit']['observed_signed_class_loss_lower'],atol=1e-10,rtol=0)
                                assert actual<=newbound+1e-7;tally[target+'_old_attack_embeddings']+=1
                        details.append({'dataset':name,'candidate':ci,'side':side,'class':label.item(),'target':target,
                            'old_upper':oldbound,'new_upper':newbound,'reported_solver_upper':upper,'actual_loss':endpoint,
                            'status':fit['solver_status'],'variables':fit['variables'],'constraints':fit['constraints']})
            assert g['signed_feasible']==all(b<=.1 for b in g['signed_class_bounds'])
            assert g['new_feasible']==all(b<=.1 for b in g['new_class_bounds'])
            assert g['positive_feasible']==all(b<=.1 for b in g['positive_class_bounds'])
            # Per-class positive guards imply global positive mean, with original class counts.
            if g['positive_feasible']:assert float(counts@np.asarray(g['positive_class_bounds'])/len(y))<=.1+1e-12
        majority=c['certificate']['majority_feasible'] and c['family_certificate']['majority_feasible']
        signedpass=all(g['signed_feasible'] for g in row['guards']);newpass=all(g['new_feasible'] for g in row['guards'])
        positivepass=all(g['positive_feasible'] for g in row['guards'])
        assert row['original_majority_feasible']==majority
        assert row['signed_dependency_feasible']==signedpass and row['new_dependency_feasible']==newpass
        assert row['signed_joint_feasible']==(signedpass and majority) and row['new_joint_feasible']==(newpass and majority)
        assert row['positive_dependency_feasible']==positivepass and row['positive_joint_feasible']==(positivepass and majority)
        assert not row['known_violation'] or not newpass
        assert not row['known_violation'] or not positivepass
        # Check all old attacks, including identical/reference guards previously reused.
        for previous in source['attacks']:
            wf=np.asarray(previous['full_weights']);label=previous['fit']['target_class'];ci_class=int(np.flatnonzero(classes==label)[0])
            Z=np.asarray(previous['fit']['perturbed_table'])
            assert Z.shape==A.shape and np.all(Z>=np.maximum(0.,A-.01)-1e-12) and np.all(Z<=np.minimum(1.,A+.01)+1e-12)
            delta=scalar_memberships(Z,y,range(A.shape[1]),wf)-scalar_memberships(Z,y,range(A.shape[1]),wide)
            sv=float(delta[y==label].mean());pv=float(np.maximum(0.,delta[y==label]).mean())
            np.testing.assert_allclose(sv,previous['fit']['observed_signed_class_loss_lower'],atol=1e-10,rtol=0)
            for g in row['guards']:
                if np.array_equal(g['full_weights'],wf):
                    assert sv<=g['new_class_bounds'][ci_class]+1e-7 and pv<=g['positive_class_bounds'][ci_class]+1e-7
            tally['all_saved_attack_tables']+=1
        tally['candidates']+=1;tally['known_violations']+=row['known_violation']
        tallies={'new_dependency_admissions':newpass and not signedpass,
                 'new_joint_admissions':newpass and majority and not row['signed_joint_feasible'],
                 'positive_dependency_admissions':positivepass and not row['legacy_joint_feasible'],
                 'positive_joint_admissions':positivepass and majority and not row['legacy_joint_feasible']}
        for key,value in tallies.items():tally[key]+=value
        if any(tallies.values()):new.append({'dataset':name,'candidate':ci,'family':row['family'],'features':fs,**tallies})
    return {'dataset':name,'passed':True,'checks':dict(tally),'status_counts':dict(status),
            'maximum_embedding_residual':maxres,'new_admissions':new,'classes':details,'artifact_sha256':digest(path)}


def main(only=None):
    threadpool_limits(1);hashes=json.loads((FOLDER/'source-freeze.json').read_text())['hashes']
    for rel,sha in hashes.items():assert digest(ROOT/rel)==sha,rel
    summary=json.loads((FOLDER/'summary.json').read_text()) if (FOLDER/'summary.json').exists() else None
    if only is None:assert summary and summary['complete']
    start=time.perf_counter();names=(only,) if only else NAMES
    reports=[audit_dataset(n,hashes,None if summary is None else summary['artifact_hashes'][n]) for n in names]
    counts=Counter();statuses=Counter()
    for report in reports:counts.update(report['checks']);statuses.update(report['status_counts'])
    if only is None:
        for key in ('candidates','known_violations','new_dependency_admissions','new_joint_admissions','positive_dependency_admissions','positive_joint_admissions'):
            assert counts[key]==summary[key],key
        assert counts['signed_solver_calls']+counts['positive_solver_calls']==summary['solver_calls']
        assert dict(statuses)==summary['solver_status_counts'] and counts['all_saved_attack_tables']==30
        for report in reports:
            for key,value in summary['dataset_counts'][report['dataset']].items():assert report['checks'][key]==value
        for target,prefix in (('signed',''),('positive','positive_')):
            dep='new_dependency_admissions' if target=='signed' else 'positive_dependency_admissions'
            joint='new_joint_admissions' if target=='signed' else 'positive_joint_admissions'
            eligible=[r for report in reports for r in report['new_admissions'] if r[dep]]
            gate=len(eligible)>=3 and len({r['dataset'] for r in eligible})>=2 and counts[joint]>=1
            assert gate==summary[prefix+'resource_gate']
    result={'passed':True,'complete_screen':only is None,'checks':dict(counts),'status_counts':dict(statuses),
        'dataset_rows':reports,'seconds':time.perf_counter()-start,'auditor_sha256':digest(__file__),
        'helper_sha256':digest(ROOT/'scripts/independent_positive_graph_embedding_v4.py'),
        'new_fit_or_attack_or_solver_runs':False,'classification_superiority_established':False,
        'scope':'both_distinct_targets_all_fixed_candidates_and_all_failed_classes_actual_model_embeddings',
        'limits':['Frozen builder/preprocessing reused; membership, positive-part, active and nearest-selector assignments independently replayed.',
                  'Backend options and saved roundtrip claims checked; no independent solver dual multipliers or formal proof available.',
                  'Legacy majority and signed/star numeric certificates remain frozen inputs.',
                  'Only positive-object admissions preserve the original stronger positive-object target.',
                  'Training reference certification gives no generalization or accuracy advantage.']}
    suffix='-'+only if only else '';out=FOLDER/('independent-both-target-endpoint-audit-v1'+suffix+'.json');assert not out.exists()
    out.write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in result.items() if k!='dataset_rows'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--only',choices=NAMES);args=p.parse_args()
    try:main(args.only)
    except Exception:
        (FOLDER/('independent-both-target-audit-failure-'+str(time.time_ns())+'.json')).write_text(json.dumps({'traceback':traceback.format_exc()},indent=2))
        raise
