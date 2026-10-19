"""Independent fixed bounds, facial seed, original reference and prediction replay."""
import argparse
import copy
import json
from collections import Counter
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_fixed_face_frs_development import ROOT,PARENTS,FOLDER,ARMS,METHODS,frozen,prerequisites,digest
from audit_fixed_face_frs_fit import check_fixed_face_fit
from audit_forest_compact_transfer import equivalent
from independent_geometric_evaluation import replay_final
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit


def audit(name):
    threadpool_limits(1);parents=prerequisites();hashes=frozen();counts=Counter(datasets=1)
    path=FOLDER/(name+'.json');saved=json.loads(path.read_text())
    assert saved['complete'] and saved['fingerprints']==hashes and saved['fold']==0
    assert not saved['query_labels_used_for_new_endpoint_selection'] and not saved['new_outer_evaluation_performed']
    assert not saved['classification_superiority_established'] and not saved['goal_achieved']
    for n,p in PARENTS.items():
        assert saved['source_hashes'][n]==parents[n]['artifact_hashes'][name]==digest(p/(name+'.json'))
    joint=json.loads((PARENTS['frs-joint-zero-face']/(name+'.json')).read_text())
    old=json.loads((PARENTS['capped-frs-training']/(name+'.json')).read_text())
    previous=json.loads((PARENTS['frs-joint-zero-face-prediction']/(name+'.json')).read_text())
    assert saved['available']==joint['available']==old['inner_training'][0]['strict_cell_available']
    for k in ('reference_indices','training_indices','validation_indices'):
        assert saved[k]==previous[k]
    assert saved['d']==joint['d'] and tuple(saved['results'])==tuple(saved['metrics'])==METHODS
    fs=joint['original_support'];anchor=joint['original_anchor']
    zeros=joint['eliminations']['pair_transition']['exact_zero_indices'] if joint['available'] else []
    seed=joint['eliminations']['pair_transition']['target'] if joint['available'] else anchor
    assert saved['fixed_zero_indices']==zeros;equivalent(saved['face_seed'],seed)
    X,y=load_dataset(name);ri,ca,cb=[np.asarray(saved[k]) for k in ('reference_indices','training_indices','validation_indices')]
    A,B=preprocess_fit(X[ri][ca],X[ri][cb]);labels=y[ri][ca]
    if joint['available']:
        assert tuple(saved['fits'])==tuple(saved['fit_reuse'])==ARMS
        for arm in ARMS:
            record=saved['fits'][arm];fit=record['fit'];facial=arm.startswith('face_')
            assert record['features']==fs;equivalent(record['anchor'],anchor)
            assert fit['optimization_eps']==(.01 if arm.endswith('robust') else 0.)
            assert fit['fixed_zero_indices']==(zeros if facial else [])
            assert fit['class_cell']['eta']==.1 and fit['polytope']['retention']==.9 and fit['certificate_eps']==.01
            equivalent(fit['class_cell'],joint['geometry']['cell'])
            equivalent(fit['requested_initial_weights'],seed if facial else anchor)
            assert np.all(np.asarray(record['target'])[zeros]==0.) if facial else True
            reused=saved['fit_reuse'][arm]
            if arm=='full_support_robust':
                assert reused=='audited_capped_anchor_fit_only'
                original=copy.deepcopy(old['inner_training'][0]['fits']['anchor'])
                reconstructed=copy.deepcopy(record)
                for key in ('optimization_eps','certificate_eps','fixed_zero_indices'):
                    del reconstructed['fit'][key]
                for step in reconstructed['fit']['steps']:
                    for trace in step['subproblem_trace']:
                        trace['anchor_blend']=trace.pop('face_seed_blend')
                equivalent(original,reconstructed);counts['already_independently_audited_control_fits_reused']+=1
            elif facial and not zeros:
                control='full_support_'+arm.split('_')[1]
                assert reused==control+'_identical_empty_face'
                equivalent(record,saved['fits'][control]);counts['identical_empty_faces_reused']+=1
            else:
                assert reused is None
                check_fixed_face_fit(A,labels,record,counts)
    else:
        assert saved['fits']=={} and saved['fit_reuse']=={}
        counts['explicit_unavailable_regions']+=1
    metrics={}
    for method in METHODS:
        if method in ARMS:
            w=anchor if not joint['available'] else saved['fits'][method]['target']
            keep=[j for j,v in enumerate(w) if v!=0.]
            metrics[method]={'features':[fs[j] for j in keep],'weights':[w[j] for j in keep]}
        else:
            src='joint_pair_transition' if method=='joint_pair' else method
            metrics[method]=previous['metrics'][src]
        equivalent(metrics[method],saved['metrics'][method])
        local=Counter();replay_final(A,labels,B,y[ri][cb],{'metric':metrics[method],**saved['results'][method]},local)
        counts['independent_inner_validation_predictions']+=local.pop('independent_outer_predictions');counts.update(local)
    scores={m:saved['results'][m]['evaluation']['metrics'] for m in METHODS}
    row={'dataset':name,'d':saved['d'],'available':saved['available'],'fixed_deletions':zeros,'metrics':scores,
        'actual_feature_counts':{m:len(metrics[m]['features']) for m in METHODS},
        'optimizer_failures':{a:saved['fits'][a]['fit']['subproblem_failures'] if joint['available'] else None for a in ARMS},
        'fit_seconds':{a:saved['fits'][a]['fit_seconds'] if joint['available'] else None for a in ARMS},'fit_reuse':saved['fit_reuse']}
    report={'passed':True,'dataset':name,'row':row,'counts':dict(counts),'sha256':digest(path),
        'auditor_sha256':digest(Path(__file__)),'fingerprints':hashes,'goal_achieved':False}
    (FOLDER/(name+'.audit.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(name,'training/fixed-face/full prediction independently checked',dict(counts),flush=True)
    return report


def complete():
    names=sorted(prerequisites()['frs-joint-zero-face']['artifact_hashes']);summary=json.loads((FOLDER/'complete.json').read_text())
    reports=[];counts=Counter()
    for name in names:
        p=FOLDER/(name+'.audit.json')
        if p.exists():
            report=json.loads(p.read_text());assert report['passed'] and report['sha256']==digest(FOLDER/(name+'.json'))
            assert report['fingerprints']==frozen() and report['auditor_sha256']==digest(Path(__file__))
        else:report=audit(name)
        reports.append(report);counts.update(report['counts'])
    rows=[r['row'] for r in reports];equivalent(rows,summary['rows'])
    assert len(rows)==summary['datasets']==15 and counts['datasets']==15
    means={m:{k:float(np.mean([r['metrics'][m][k] for r in rows])) for k in rows[0]['metrics'][m]} for m in METHODS}
    equivalent(means,summary['means'])
    for facial,controls in summary['contrasts'].items():
        for control,expected in controls.items():
            d=[100*(r['metrics'][facial]['clean_BA']-r['metrics'][control]['clean_BA']) for r in rows]
            equivalent({'mean_BA_delta_pp':float(np.mean(d)),'wins':sum(v>1e-10 for v in d),
                'losses':sum(v< -1e-10 for v in d),'ties':sum(abs(v)<=1e-10 for v in d)},expected)
    equivalent({m:float(np.mean([r['actual_feature_counts'][m] for r in rows])) for m in METHODS},summary['mean_actual_feature_counts'])
    assert counts['independent_inner_validation_predictions']==165 and counts['independent_noise_predictions']==2640
    hashes={r['dataset']:r['sha256'] for r in reports};equivalent(hashes,summary['artifact_hashes'])
    out={'passed':True,'counts':dict(counts),'artifact_hashes':hashes,'summary_sha256':digest(FOLDER/'complete.json'),
        'auditor_sha256':digest(Path(__file__)),'classification_superiority_established':False,'goal_achieved':False}
    (FOLDER/'complete-audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset');args=p.parse_args()
    if args.dataset:audit(args.dataset)
    else:print(json.dumps(complete(),indent=2),flush=True)
