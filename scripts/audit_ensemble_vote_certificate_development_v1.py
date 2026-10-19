"""Independent dataset and complete factorial audit before any promotion."""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_ensemble_vote_certificate_development_v1 import ROOT,FOLDER,FREEZE,ARMS,METHODS,frozen,inputs,row,digest
from audit_ensemble_vote_certificate_metric_v1 import check_ensemble_vote_certificate_fit
from independent_frs_pair_transition_cell import reconstruct_cell
from independent_geometric_evaluation import replay_final
from audit_forest_compact_transfer import equivalent


def audit(name):
    threadpool_limits(1);freeze=frozen();counts=Counter(datasets=1)
    source,previous,A,y,B,truth,hashes=inputs(name,freeze)
    path=FOLDER/(name+'.json');saved=json.loads(path.read_text())
    assert saved['complete'] and saved['source_hashes']==hashes and saved['freeze_sha256']==digest(FREEZE)
    assert not saved['heldout_labels_used_for_endpoint_selection'] and not saved['new_outer_evaluation_performed']
    assert not saved['classification_superiority_established'] and not saved['goal_achieved']
    equivalent(saved['validation_truth'],truth.tolist())
    assert saved['source_available']==source['available'] and saved['d']==source['d']
    for k in ('reference_indices','training_indices','validation_indices'):assert saved[k]==previous[k]
    cell_evidence=None
    for arm in ARMS:
        partial=FOLDER/(name+'.'+arm+'.fit.json');assert saved['fit_hashes'][arm]==digest(partial)
        r=json.loads(partial.read_text());assert r['source_hashes']==hashes and r['freeze_sha256']==digest(FREEZE)
        fit=r['fit'];cell,support,objective=arm.removeprefix('ensemble_').split('_');control=f'ensemble_{cell}_control_{objective}'
        expected_reuse=control if source['available'] and support=='face' and not source['zeros'] else None
        assert saved['fit_reuse'][arm]==r['reuse']==expected_reuse
        if expected_reuse:
            equivalent(fit,json.loads((FOLDER/(name+'.'+control+'.fit.json')).read_text())['fit'])
            counts['identical_empty_face_reuses']+=1
        elif source['available']:
            assert fit['refresh'] and fit['refresh_frs']==(cell=='joint') and fit['optimization_eps']==0.
            assert fit['objective_family']==('own_mass' if objective=='ownmass' else 'plurality')
            assert fit['partition_seeds']==[20261071,20261072,20261073]
            assert fit['max_steps']==12 and fit['lp_time_limit']==20.
            equivalent(fit['anchor'],source['original_anchor']);equivalent(fit['full_reference'],source['full_reference'])
            equivalent(fit['seed'],source['seed'] if support=='face' else source['original_anchor'])
            assert fit['fixed_zeros']==(source['zeros'] if support=='face' else [])
            check_ensemble_vote_certificate_fit(A,y,fit,counts)
        else:
            assert not fit['available'] and fit['status']=='source_certified_geometry_unavailable'
            equivalent(fit['weights'],source['seed']);counts['source_unavailable_arms']+=1
        active=[j for j,w in enumerate(fit['weights']) if w!=0.]
        metric={'features':[source['original_support'][j] for j in active],'weights':[fit['weights'][j] for j in active]}
        equivalent(metric,saved['metrics'][arm])
        expected_training={k:fit.get(k) for k in ('available','status','initial_objective','final_objective','accepted_steps','fit_seconds','temperature','objective_family')}
        expected_training['nonempty_refreshes']=sum(s.get('witness_refresh_changed',False) for s in fit['steps'])
        equivalent(expected_training,saved['training'][arm]);del r,fit
    verified_metrics={}
    for method in METHODS:
        metric=saved['metrics'][method];key=json.dumps(metric,sort_keys=True)
        if method not in ARMS:
            src='joint_pair_transition' if method=='joint_pair' else method
            equivalent(metric,previous['metrics'][src]);equivalent(saved['results'][method],previous['results'][src])
            counts['inherited_audited_predictions']+=1
        elif key in verified_metrics:
            equivalent(saved['results'][method],saved['results'][verified_metrics[key]])
            counts['identical_metric_prediction_reuses']+=1
        else:
            local=Counter();replay_final(A,y,B,truth,{'metric':metric,**saved['results'][method]},local)
            counts['independent_inner_validation_predictions']+=local.pop('independent_outer_predictions');counts.update(local)
        verified_metrics[key]=method
    report={'passed':True,'dataset':name,'row':row(saved),'counts':dict(counts),'sha256':digest(path),
        'fit_hashes':saved['fit_hashes'],'freeze_sha256':digest(FREEZE),'auditor_sha256':digest(Path(__file__)),
        'goal_achieved':False,'classification_superiority_established':False}
    (FOLDER/(name+'.audit.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(name,'independent ensemble-vote certificate audit',dict(counts),flush=True)
    return report


def complete():
    freeze=frozen();names=sorted(freeze['parents']['frs-joint-zero-face']['artifact_hashes']);reports=[];counts=Counter()
    summary=json.loads((FOLDER/'complete.json').read_text())
    for name in names:
        report=json.loads((FOLDER/(name+'.audit.json')).read_text())
        assert report['passed'] and report['sha256']==digest(FOLDER/(name+'.json'))
        assert report['auditor_sha256']==digest(Path(__file__)) and report['freeze_sha256']==digest(FREEZE)
        for arm,sha in report['fit_hashes'].items():assert digest(FOLDER/(name+'.'+arm+'.fit.json'))==sha
        reports.append(report);counts.update(report['counts'])
    rows=[r['row'] for r in reports];equivalent(rows,summary['rows']);assert len(rows)==summary['datasets']==15
    for m in METHODS:
        equivalent({k:float(np.mean([r['metrics'][m][k] for r in rows])) for k in rows[0]['metrics'][m]},summary['means'][m])
    for arm,controls in summary['contrasts'].items():
        for control,expected in controls.items():
            d=[100*(r['metrics'][arm]['clean_BA']-r['metrics'][control]['clean_BA']) for r in rows]
            equivalent({'mean_BA_delta_pp':float(np.mean(d)),'wins':sum(v>1e-10 for v in d),
                'losses':sum(v< -1e-10 for v in d),'ties':sum(abs(v)<=1e-10 for v in d)},expected)
    equivalent({m:float(np.mean([r['actual_feature_counts'][m] for r in rows])) for m in METHODS},summary['mean_actual_feature_counts'])
    hashes={r['dataset']:r['sha256'] for r in reports};equivalent(hashes,summary['artifact_hashes'])
    out={'passed':True,'counts':dict(counts),'artifact_hashes':hashes,'summary_sha256':digest(FOLDER/'complete.json'),
        'auditor_sha256':digest(Path(__file__)),'freeze_sha256':digest(FREEZE),
        'goal_achieved':False,'classification_superiority_established':False}
    (FOLDER/'complete-audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8');return out


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset');args=parser.parse_args()
    if args.dataset:audit(args.dataset)
    else:print(json.dumps(complete(),indent=2),flush=True)
