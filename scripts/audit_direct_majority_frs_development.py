"""Original training geometry, actual dynamic-rank and ALL15 prediction replay."""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_direct_majority_frs_development import ROOT,PARENTS,FOLDER,ARMS,METHODS,frozen,prerequisites,digest
from audit_fixed_face_majority_milp import check_fit
from audit_forest_compact_transfer import equivalent
from independent_geometric_evaluation import replay_final
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit


def audit(name):
    threadpool_limits(1);parents=prerequisites();hashes=frozen();counts=Counter(datasets=1);path=FOLDER/(name+'.json')
    saved=json.loads(path.read_text());assert saved['complete'] and saved['fingerprints']==hashes and saved['fold']==0
    assert not saved['query_labels_used_for_new_endpoint_selection'] and not saved['new_outer_evaluation_performed']
    assert not saved['goal_achieved'] and not saved['classification_superiority_established']
    for n,p in PARENTS.items():assert saved['source_hashes'][n]==parents[n]['artifact_hashes'][name]==digest(p/(name+'.json'))
    joint=json.loads((PARENTS['frs-joint-zero-face']/(name+'.json')).read_text())
    previous=json.loads((PARENTS['frs-joint-zero-face-prediction']/(name+'.json')).read_text())
    assert saved['available']==joint['available'] and saved['d']==joint['d']
    for k in ('reference_indices','training_indices','validation_indices'):assert saved[k]==previous[k]
    assert tuple(saved['results'])==tuple(saved['metrics'])==METHODS
    fs=joint['original_support'];anchor=joint['original_anchor'];available=joint['available']
    zeros=joint['eliminations']['pair_transition']['exact_zero_indices'] if available else []
    seed=joint['eliminations']['pair_transition']['target'] if available else anchor
    assert saved['fixed_zero_indices']==zeros;equivalent(saved['face_seed'],seed)
    X,y=load_dataset(name);ri,ca,cb=[np.asarray(saved[k]) for k in ('reference_indices','training_indices','validation_indices')]
    A,B=preprocess_fit(X[ri][ca],X[ri][cb]);labels=y[ri][ca]
    if available:
        assert tuple(saved['fits'])==tuple(saved['fit_reuse'])==ARMS
        for arm in ARMS:
            fit=saved['fits'][arm];facial=arm=='direct_face'
            assert fit['features']==fs;equivalent(fit['anchor'],anchor);equivalent(fit['seed'],seed if facial else anchor)
            assert fit['fixed_zero_indices']==(zeros if facial else []) and fit['time_limit']==120. and fit['margin']==.02 and fit['reg']==.01
            equivalent(fit['class_cell'],joint['geometry']['cell'])
            if facial and not zeros:
                assert saved['fit_reuse'][arm]=='identical_empty_face';equivalent(fit,saved['fits']['direct_control'])
                counts['identical_empty_faces_reused']+=1
            else:
                assert saved['fit_reuse'][arm] is None;check_fit(A,labels,fit,counts)
    else:assert saved['fits']=={} and saved['fit_reuse']=={}
    metrics={}
    for method in METHODS:
        if method in ARMS:
            w=saved['fits'][method]['weights'] if available else anchor;keep=[j for j,v in enumerate(w) if v!=0.]
            metrics[method]={'features':[fs[j] for j in keep],'weights':[w[j] for j in keep]}
        else:metrics[method]=previous['metrics']['joint_pair_transition' if method=='joint_pair' else method]
        equivalent(metrics[method],saved['metrics'][method]);local=Counter()
        replay_final(A,labels,B,y[ri][cb],{'metric':metrics[method],**saved['results'][method]},local)
        counts['independent_inner_validation_predictions']+=local.pop('independent_outer_predictions');counts.update(local)
    row={'dataset':name,'d':saved['d'],'available':available,'fixed_deletions':zeros,
        'metrics':{m:saved['results'][m]['evaluation']['metrics'] for m in METHODS},
        'actual_feature_counts':{m:len(metrics[m]['features']) for m in METHODS},
        'training':{a:None if not available else {k:saved['fits'][a][k] for k in
            ('accepted_descent','status','mip_status','mip_gap_solver_reported','initial_objective','final_objective','fit_seconds')} for a in ARMS}}
    report={'passed':True,'dataset':name,'row':row,'counts':dict(counts),'sha256':digest(path),
        'auditor_sha256':digest(Path(__file__)),'fingerprints':hashes,'goal_achieved':False}
    (FOLDER/(name+'.audit.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(name,'independent dynamic-majority training and actual prediction audit',dict(counts),flush=True);return report


def complete():
    names=sorted(prerequisites()['frs-joint-zero-face']['artifact_hashes']);summary=json.loads((FOLDER/'complete.json').read_text())
    reports=[];counts=Counter()
    for name in names:
        p=FOLDER/(name+'.audit.json')
        if p.exists():
            r=json.loads(p.read_text());assert r['passed'] and r['sha256']==digest(FOLDER/(name+'.json'))
            assert r['fingerprints']==frozen() and r['auditor_sha256']==digest(Path(__file__))
        else:r=audit(name)
        reports.append(r);counts.update(r['counts'])
    rows=[r['row'] for r in reports];equivalent(rows,summary['rows'])
    means={m:{k:float(np.mean([r['metrics'][m][k] for r in rows])) for k in rows[0]['metrics'][m]} for m in METHODS};equivalent(means,summary['means'])
    for control,expected in summary['contrasts'].items():
        d=[100*(r['metrics']['direct_face']['clean_BA']-r['metrics'][control]['clean_BA']) for r in rows]
        equivalent({'mean_BA_delta_pp':float(np.mean(d)),'wins':sum(v>1e-10 for v in d),'losses':sum(v< -1e-10 for v in d),'ties':sum(abs(v)<=1e-10 for v in d)},expected)
    equivalent({m:float(np.mean([r['actual_feature_counts'][m] for r in rows])) for m in METHODS},summary['mean_actual_feature_counts'])
    assert len(rows)==summary['datasets']==15 and counts['datasets']==15
    assert counts['independent_inner_validation_predictions']==135 and counts['independent_noise_predictions']==2160
    hashes={r['dataset']:r['sha256'] for r in reports};equivalent(hashes,summary['artifact_hashes'])
    out={'passed':True,'counts':dict(counts),'artifact_hashes':hashes,'summary_sha256':digest(FOLDER/'complete.json'),
        'auditor_sha256':digest(Path(__file__)),'classification_superiority_established':False,'goal_achieved':False}
    (FOLDER/'complete-audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8');return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset');args=p.parse_args()
    if args.dataset:audit(args.dataset)
    else:print(json.dumps(complete(),indent=2),flush=True)
