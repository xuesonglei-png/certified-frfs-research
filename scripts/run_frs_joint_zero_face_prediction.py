"""Actual first-inner-fold clean/noisy predictions under matched full controls."""
import argparse
import json
import time
import numpy as np
from threadpoolctl import threadpool_limits
from run_simplex_soft5_knn_pilot import ROOT,digest
from continue_soft_topk_knn_pilot import verify
from run_frs_zero_face_repair_experiment import builtin
from geometric_vote_evaluation import final_evaluation
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit

JOINT=ROOT/'results/frs-joint-zero-face'
NATIVE=ROOT/'results/majority-hinge-canonical-benchmark'
FOLDER=ROOT/'results/frs-joint-zero-face-prediction'
FREEZE=ROOT/'research/frs-joint-zero-face-prediction-source-freeze.json'
NATIVE_ARMS=('fgs_native120','fgs_weighted_native120','sfrip_native6','sfrip_weighted_native6')
REGIMES=('majority_only','stable_cell_only','one_sided','pair_transition')
METHODS=('anchor','full_uniform')+tuple('joint_'+r for r in REGIMES)+NATIVE_ARMS


def prerequisites():
    parents={p.name:verify(p) for p in (JOINT,NATIVE)}
    assert set(parents[JOINT.name]['artifact_hashes'])==set(parents[NATIVE.name]['artifact_hashes'])
    assert len(parents[JOINT.name]['artifact_hashes'])==15
    return parents


def frozen():
    hashes=json.loads(FREEZE.read_text())['fingerprints']
    for rel,sha in hashes.items():
        assert digest(ROOT/rel)==sha,rel
    return hashes


def alignment(joint,native):
    assert joint['complete'] and native['complete'] and joint['fold']==0
    assert joint['reference_indices']==native['reference_indices']
    first=native['inner'][0]
    assert joint['training_indices']==first['reference_indices']
    assert not set(first['reference_indices'])&set(first['validation_indices'])
    base=first['bases']['proposed'][0]
    assert base['features']==joint['original_support'] and base['weights']==joint['original_anchor']
    return first


def metrics_from_training(joint,native):
    first=alignment(joint,native);d=joint['d']
    metrics={'anchor':{'features':joint['original_support'],'weights':joint['original_anchor']},
        'full_uniform':{'features':list(range(d)),'weights':(np.ones(d)/np.sqrt(d)).tolist()}}
    for r in REGIMES:
        weights=joint['eliminations'][r]['target'] if joint['available'] else joint['original_anchor']
        keep=[i for i,w in enumerate(weights) if w!=0.]
        metrics['joint_'+r]={'features':[joint['original_support'][i] for i in keep],
                            'weights':[weights[i] for i in keep]}
    choices={}
    for m in NATIVE_ARMS:
        selected=native['selection'][m]['selected_index'];options=first['options'][m]
        assert len(options)==(1680 if m=='fgs_weighted_native120' else
            120 if m=='fgs_native120' else 84 if m=='sfrip_weighted_native6' else 6)
        original=options[selected]['metric'];metrics[m]={k:original[k] for k in ('features','weights')}
        choices[m]={'selected_index':selected,'candidate_count':len(options),
            'recipe':native['selection'][m]['options'][selected]}
    return metrics,choices


def row(saved):
    metrics={m:r['evaluation']['metrics'] for m,r in saved['results'].items()}
    sizes={m:len(v['features']) for m,v in saved['metrics'].items()}
    return {'dataset':saved['dataset'],'d':saved['d'],'metrics':metrics,'actual_feature_counts':sizes,
        'unit_cost_removed_fractions':{m:1-sizes[m]/saved['d'] for m in METHODS},
        'prediction_seconds':saved['prediction_seconds'],
        'pair_minus_anchor_BA_pp':100*(metrics['joint_pair_transition']['clean_BA']-metrics['anchor']['clean_BA']),
        'pair_minus_one_sided_BA_pp':100*(metrics['joint_pair_transition']['clean_BA']-metrics['joint_one_sided']['clean_BA'])}


def run(name):
    threadpool_limits(1);parents=prerequisites();hashes=frozen();FOLDER.mkdir(exist_ok=True)
    path=FOLDER/(name+'.json')
    if path.exists():
        old=json.loads(path.read_text());assert old['complete'] and old['fingerprints']==hashes;return
    source_hashes={p.name:digest(p/(name+'.json')) for p in (JOINT,NATIVE)}
    for p in (JOINT,NATIVE):
        assert source_hashes[p.name]==parents[p.name]['artifact_hashes'][name]
    joint=json.loads((JOINT/(name+'.json')).read_text());native=json.loads((NATIVE/(name+'.json')).read_text())
    first=alignment(joint,native);metrics,choices=metrics_from_training(joint,native)
    X,y=load_dataset(name);ri=np.asarray(joint['reference_indices']);ca=np.asarray(first['reference_indices']);cb=np.asarray(first['validation_indices'])
    A,B=preprocess_fit(X[ri][ca],X[ri][cb]);timer=time.perf_counter()
    results={m:final_evaluation(A,y[ri][ca],B,y[ri][cb],metrics[m]) for m in METHODS}
    saved={'dataset':name,'d':joint['d'],'fold':0,'reference_indices':ri.tolist(),
        'training_indices':ca.tolist(),'validation_indices':cb.tolist(),'metrics':metrics,
        'native_choices':choices,'source_hashes':source_hashes,'results':results,'fingerprints':hashes,
        'prediction_seconds':time.perf_counter()-timer,'complete':True,'classification_scored':True,
        'query_labels_used_for_joint_selection':False,'new_outer_evaluation_performed':False,
        'classification_superiority_established':False,'goal_achieved':False,
        'scope':'all15_exposed_firstinnerfold_jointreduction_predictive_diagnostic_not_confirmation'}
    path.write_text(json.dumps(saved,default=builtin,separators=(',',':')),encoding='utf-8')
    print(name,'first-inner validation balanced accuracies',{m:round(v['evaluation']['metrics']['clean_BA'],6) for m,v in results.items()},flush=True)


def summarize(names):
    rows=[row(json.loads((FOLDER/(n+'.json')).read_text())) for n in names]
    means={m:{k:float(np.mean([r['metrics'][m][k] for r in rows])) for k in rows[0]['metrics'][m]} for m in METHODS}
    deltas={m:100*(means['joint_pair_transition']['clean_BA']-means[m]['clean_BA']) for m in METHODS if m!='joint_pair_transition'}
    counts={m:{'wins':sum(r['metrics']['joint_pair_transition']['clean_BA']>r['metrics'][m]['clean_BA']+1e-12 for r in rows),
        'losses':sum(r['metrics']['joint_pair_transition']['clean_BA']<r['metrics'][m]['clean_BA']-1e-12 for r in rows)} for m in deltas}
    out={'rows':rows,'datasets':len(names),'means':means,'pair_clean_BA_differences_pp':deltas,'pair_wins_losses':counts,
        'mean_actual_feature_counts':{m:float(np.mean([r['actual_feature_counts'][m] for r in rows])) for m in METHODS},
        'artifact_hashes':{n:digest(FOLDER/(n+'.json')) for n in names},'independent_audit_required':True,
        'classification_scored':True,'classification_superiority_established':False,'new_outer_evaluation_performed':False,'goal_achieved':False,
        'limits':['Repeatedly exposed first inner validation folds are development only.',
                  'Native controls retain entire original candidate families and inner selections.',
                  'Nominal unit feature counts are not measured monetary or end-to-end costs.',
                  'No method choice or promotion is made on these prediction scores.']}
    (FOLDER/'complete.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({'means':means,'pair_clean_BA_differences_pp':deltas},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset');args=p.parse_args()
    if args.dataset:
        run(args.dataset)
    else:
        names=sorted(prerequisites()[JOINT.name]['artifact_hashes'])
        for name in names:
            run(name)
        summarize(names)
