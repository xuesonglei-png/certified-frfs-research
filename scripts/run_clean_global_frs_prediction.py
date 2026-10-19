"""ALL15 first-inner clean/query/noise predictions under native full controls."""
import argparse
import json
import time
import numpy as np
from threadpoolctl import threadpool_limits
from run_simplex_soft5_knn_pilot import ROOT,digest
from continue_soft_topk_knn_pilot import verify
from run_frs_zero_face_repair_experiment import builtin
from clean_global_prediction_design import metrics_from_training,METHODS
from geometric_vote_evaluation import final_evaluation
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit

SOURCE=ROOT/'results/clean-global-frs-training'
NATIVE=ROOT/'results/majority-hinge-canonical-benchmark'
FOLDER=ROOT/'results/clean-global-frs-prediction'
FREEZE=ROOT/'research/clean-global-frs-prediction-source-freeze.json'
REVIEW=ROOT/'research/clean-global-FRS-training-review.json'


def prerequisites():
    audits={p.name:verify(p) for p in (SOURCE,NATIVE)};review=json.loads(REVIEW.read_text())
    assert review['audit_sha256']==digest(SOURCE/'complete-audit.json')
    assert len(audits[SOURCE.name]['artifact_hashes'])==len(review['all_dataset_rows'])==15
    assert set(audits[SOURCE.name]['artifact_hashes'])==set(audits[NATIVE.name]['artifact_hashes'])
    return audits


def frozen():
    hashes=json.loads(FREEZE.read_text())['fingerprints']
    for rel,sha in hashes.items():
        assert digest(ROOT/rel)==sha,rel
    return hashes


def row(saved):
    scores={m:r['evaluation']['metrics'] for m,r in saved['results'].items()}
    sizes={m:len(r['features']) for m,r in saved['metrics'].items()}
    return {'dataset':saved['dataset'],'d':saved['d'],'metrics':scores,'actual_feature_counts':sizes,
        'nominal_minus_robust_BA_pp':100*(scores['nominal_training']['clean_BA']-scores['robust_training']['clean_BA']),
        'nominal_minus_robust_noise_BA_pp':100*(scores['nominal_training']['mean_perturbed_BA']-scores['robust_training']['mean_perturbed_BA']),
        'source_global_region_available':saved['source_global_region_available']}


def run(name):
    threadpool_limits(1);parents=prerequisites();hashes=frozen();FOLDER.mkdir(exist_ok=True);path=FOLDER/(name+'.json')
    if path.exists():
        saved=json.loads(path.read_text());assert saved['complete'] and saved['fingerprints']==hashes;return
    sources={p.name:digest(p/(name+'.json')) for p in (SOURCE,NATIVE)}
    for folder in (SOURCE,NATIVE):
        assert sources[folder.name]==parents[folder.name]['artifact_hashes'][name]
    training=json.loads((SOURCE/(name+'.json')).read_text());native=json.loads((NATIVE/(name+'.json')).read_text())
    metrics,choices=metrics_from_training(training,native);first=native['inner'][0]
    ri=np.asarray(training['reference_indices']);ca=np.asarray(training['training_indices']);cb=np.asarray(first['validation_indices'])
    X,y=load_dataset(name);A,B=preprocess_fit(X[ri][ca],X[ri][cb]);timer=time.perf_counter()
    results={m:final_evaluation(A,y[ri][ca],B,y[ri][cb],metrics[m]) for m in METHODS}
    saved={'dataset':name,'d':training['d'],'fold':0,'reference_indices':ri.tolist(),'training_indices':ca.tolist(),
        'validation_indices':cb.tolist(),'metrics':metrics,'native_choices':choices,'source_hashes':sources,'results':results,
        'fingerprints':hashes,'prediction_seconds':time.perf_counter()-timer,
        'source_global_region_available':False if training['block'] is None else training['block']['control']['available'],
        'complete':True,'classification_scored':True,'query_labels_used_for_new_endpoint_selection':False,'new_outer_evaluation_performed':False,
        'classification_superiority_established':False,'goal_achieved':False,
        'scope':'all15_exposed_first_inner_nominal_versus_robust_loss_predictive_screen_not_confirmation'}
    path.write_text(json.dumps(saved,default=builtin,separators=(',',':')),encoding='utf-8')
    print(name,'nominal/robust/full-native first-inner BA',{m:round(r['evaluation']['metrics']['clean_BA'],6) for m,r in results.items()},flush=True)


def summarize(names):
    rows=[row(json.loads((FOLDER/(n+'.json')).read_text())) for n in names]
    means={m:{k:float(np.mean([r['metrics'][m][k] for r in rows])) for k in rows[0]['metrics'][m]} for m in METHODS}
    deltas={m:100*(means['nominal_training']['clean_BA']-means[m]['clean_BA']) for m in METHODS if m!='nominal_training'}
    wl={m:{'wins':sum(r['metrics']['nominal_training']['clean_BA']>r['metrics'][m]['clean_BA']+1e-12 for r in rows),
        'losses':sum(r['metrics']['nominal_training']['clean_BA']<r['metrics'][m]['clean_BA']-1e-12 for r in rows)} for m in deltas}
    out={'rows':rows,'datasets':len(names),'means':means,'nominal_clean_BA_differences_pp':deltas,'nominal_wins_losses':wl,
        'mean_actual_feature_counts':{m:float(np.mean([r['actual_feature_counts'][m] for r in rows])) for m in METHODS},
        'artifact_hashes':{n:digest(FOLDER/(n+'.json')) for n in names},'independent_audit_required':True,
        'classification_scored':True,'classification_superiority_established':False,'new_outer_evaluation_performed':False,'goal_achieved':False,
        'limits':['First inner validation folds are already exposed development data.',
                  'Native controls retain entire original candidate families and training-selected recipes.',
                  'Nominal loss radius0 changes optimization only; original certifying radius .01 unchanged.',
                  'No score-based method promotion or test-set superiority claim.']}
    (FOLDER/'complete.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps({'means':means,'deltas_pp':deltas},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset');args=p.parse_args()
    if args.dataset:
        run(args.dataset)
    else:
        names=sorted(prerequisites()[SOURCE.name]['artifact_hashes'])
        for name in names:
            run(name)
        summarize(names)
