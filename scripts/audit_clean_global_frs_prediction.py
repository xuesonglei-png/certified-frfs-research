"""Independent endpoints/native choices and all first-inner prediction replays."""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_clean_global_frs_prediction import ROOT,SOURCE,NATIVE,FOLDER,frozen,prerequisites,digest,METHODS
from audit_clean_global_prediction_design import reconstruct_metrics
from independent_geometric_evaluation import replay_final
from audit_forest_compact_transfer import equivalent
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit


def independent_row(saved,metrics):
    scores={m:saved['results'][m]['evaluation']['metrics'] for m in METHODS}
    return {'dataset':saved['dataset'],'d':saved['d'],'metrics':scores,
        'actual_feature_counts':{m:len(metrics[m]['features']) for m in METHODS},
        'nominal_minus_robust_BA_pp':100*(scores['nominal_training']['clean_BA']-scores['robust_training']['clean_BA']),
        'nominal_minus_robust_noise_BA_pp':100*(scores['nominal_training']['mean_perturbed_BA']-scores['robust_training']['mean_perturbed_BA']),
        'source_global_region_available':saved['source_global_region_available']}


def audit(name):
    threadpool_limits(1);parents=prerequisites();hashes=frozen();counts=Counter(datasets=1)
    path=FOLDER/(name+'.json');saved=json.loads(path.read_text())
    assert saved['complete'] and saved['fingerprints']==hashes and saved['classification_scored'] and saved['fold']==0
    assert not saved['query_labels_used_for_new_endpoint_selection'] and not saved['new_outer_evaluation_performed']
    assert not saved['classification_superiority_established'] and not saved['goal_achieved']
    for folder in (SOURCE,NATIVE):
        assert saved['source_hashes'][folder.name]==parents[folder.name]['artifact_hashes'][name]==digest(folder/(name+'.json'))
    training=json.loads((SOURCE/(name+'.json')).read_text());native=json.loads((NATIVE/(name+'.json')).read_text())
    metrics,choices=reconstruct_metrics(training,native);equivalent(metrics,saved['metrics']);equivalent(choices,saved['native_choices'])
    assert saved['reference_indices']==training['reference_indices'] and saved['training_indices']==training['training_indices']
    assert saved['validation_indices']==native['inner'][0]['validation_indices']
    assert saved['source_global_region_available']==(False if training['block'] is None else training['block']['control']['available'])
    assert tuple(saved['results'])==METHODS and saved['d']==training['d']
    X,y=load_dataset(name);ri,ca,cb=[np.asarray(saved[k]) for k in ('reference_indices','training_indices','validation_indices')]
    A,B=preprocess_fit(X[ri][ca],X[ri][cb])
    for method in METHODS:
        local=Counter();replay_final(A,y[ri][ca],B,y[ri][cb],{'metric':metrics[method],**saved['results'][method]},local)
        counts['independent_inner_validation_predictions']+=local.pop('independent_outer_predictions');counts.update(local)
    print(name,'all8 nominal/robust/native arms independently replayed',dict(counts),flush=True)
    return {'dataset':name,'row':independent_row(saved,metrics),'counts':dict(counts),'sha256':digest(path)}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset');args=p.parse_args()
    if args.dataset:
        r=audit(args.dataset);(FOLDER/(args.dataset+'.smoke-audit.json')).write_text(json.dumps({'passed':True,**r},indent=2),encoding='utf-8')
    else:
        summary=json.loads((FOLDER/'complete.json').read_text());names=sorted(prerequisites()[SOURCE.name]['artifact_hashes']);reports=[];counts=Counter()
        assert [r['dataset'] for r in summary['rows']]==names
        for name in names:
            r=audit(name);reports.append(r);counts.update(r['counts'])
        rows=[r['row'] for r in reports];equivalent(rows,summary['rows'])
        means={m:{k:float(np.mean([r['metrics'][m][k] for r in rows])) for k in rows[0]['metrics'][m]} for m in METHODS}
        equivalent(means,summary['means'])
        for method,delta in summary['nominal_clean_BA_differences_pp'].items():
            equivalent(float(np.mean([100*(r['metrics']['nominal_training']['clean_BA']-r['metrics'][method]['clean_BA']) for r in rows])),delta)
            wins=sum(r['metrics']['nominal_training']['clean_BA']-r['metrics'][method]['clean_BA']>1e-12 for r in rows)
            losses=sum(r['metrics'][method]['clean_BA']-r['metrics']['nominal_training']['clean_BA']>1e-12 for r in rows)
            assert summary['nominal_wins_losses'][method]=={'wins':wins,'losses':losses}
        equivalent({m:float(np.mean([r['actual_feature_counts'][m] for r in rows])) for m in METHODS},summary['mean_actual_feature_counts'])
        assert counts['datasets']==summary['datasets']==15
        assert counts['independent_inner_validation_predictions']==120 and counts['independent_noise_predictions']==1920
        hashes={r['dataset']:r['sha256'] for r in reports};equivalent(hashes,summary['artifact_hashes'])
        out={'passed':True,'counts':dict(counts),'artifact_hashes':hashes,'summary_sha256':digest(FOLDER/'complete.json'),
            'auditor_sha256':digest(Path(__file__)),'new_outer_evaluation_performed':False,'classification_superiority_established':False,'goal_achieved':False}
        (FOLDER/'complete-audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2),flush=True)
